"""Synthetic ground truth tests; no microscope images are checked into Git."""
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from main import SEMStitcher


def texture(size=320, seed=42):
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.uniform(30, 220, (size, size)).astype(np.float32), (0, 0), 1.2)
    return cv2.normalize(image, None, 20, 235, cv2.NORM_MINMAX).astype(np.uint8)


class AlignmentBlendingTests(unittest.TestCase):
    def setUp(self):
        cv2.setRNGSeed(42)
        self.stitcher = SEMStitcher()
        self.stitcher.set_progress_callback(lambda message: None)

    def test_phase_recovers_known_shifts_without_double_counting_fraction(self):
        source = texture()
        for dx, dy in [(0.35, -0.6), (32.35, -21.6), (-30.35, 20.6), (20.0, -12.0)]:
            with self.subTest(shift=(dx, dy)):
                reference = cv2.warpAffine(source, np.float32([[1, 0, dx], [0, 1, dy]]),
                                           source.shape[::-1], borderMode=cv2.BORDER_REFLECT)
                coarse = (dx + 0.3, dy - 0.3)
                correction = self.stitcher._fine_align_phase(source, reference, *coarse)
                np.testing.assert_allclose(np.array(coarse) + correction, (dx, dy), atol=0.2)

    def test_real_pair_registration_handles_odd_fft_overlap_sizes(self):
        scene = texture(480)
        source = scene[:320, :320]
        y, x = np.mgrid[:320, :320].astype(np.float32)
        for dx, dy in [(96.35, 12.6), (95.35, 95.6), (63.7, 16.2), (50, 95)]:
            with self.subTest(shift=(dx, dy)):
                reference = cv2.remap(scene, x + dx, y + dy, cv2.INTER_LINEAR)
                actual_x, actual_y, quality = self.stitcher._try_match_pair(source, reference)
                self.assertGreaterEqual(quality, 8)
                self.assertLess(np.hypot(actual_x + dx, actual_y + dy), 0.2)

    def test_phase_rejects_unreliable_results_and_featureless_overlap(self):
        source = texture()
        for shift, response in [((float('nan'), 0), 1), ((0, 0), float('nan')),
                                ((0, 0), 0.01), ((100, 0), 1)]:
            with self.subTest(shift=shift, response=response):
                with patch('main.cv2.phaseCorrelate', return_value=(shift, response)):
                    self.assertEqual(self.stitcher._fine_align_phase(source, source, 0.3, 0.3), (0, 0))
        flat = np.full((100, 100), 120, np.uint8)
        self.assertEqual(self.stitcher._fine_align_phase(flat, flat, 0, 0), (0, 0))
        self.assertEqual(self.stitcher._fine_align_phase(source, source, 300, 0), (0, 0))

    def test_pair_applies_refinement(self):
        image = texture()
        with patch.object(self.stitcher, '_compute_translation', return_value=(20.4, -5.2, 30, 40)), \
             patch.object(self.stitcher, '_fine_align_phase', return_value=(-0.2, 0.3)):
            dx, dy, quality = self.stitcher._try_match_pair(image, image)
        np.testing.assert_allclose((dx, dy), (20.2, -4.9))
        self.assertEqual(quality, 30)

    def test_two_image_api_uses_same_refined_registration(self):
        image = texture(100)
        with patch.object(self.stitcher, '_try_match_pair', return_value=(20.25, -5.5, 30)):
            result = self.stitcher.stitch_two(image, image)
        np.testing.assert_array_equal(result, self.stitcher._blend_images(image, image, 20.25, -5.5))

    def test_single_tile_keeps_every_border_pixel(self):
        image = texture(64)
        np.testing.assert_array_equal(self.stitcher._compose_images([image], [(0, 0)]), image)

    def test_four_way_junction_has_no_holes_or_dark_borders(self):
        images = [np.full((64, 64), 173, np.uint8) for _ in range(4)]
        result = self.stitcher._compose_images(images, [(0, 0), (32, 0), (0, 32), (32, 32)])
        self.assertEqual(result.shape, (96, 96))
        np.testing.assert_array_equal(result, np.full((96, 96), 173, np.uint8))

    def test_blending_is_independent_of_tile_order(self):
        images = [np.full((40, 40), value, np.uint8) for value in (80, 120, 180, 220)]
        positions = [(0, 0), (20.25, 0), (0, 20.5), (20.25, 20.5)]
        expected = self.stitcher._compose_images(images, positions)
        for order in itertools.permutations(range(4)):
            actual = self.stitcher._compose_images([images[i] for i in order], [positions[i] for i in order])
            np.testing.assert_array_equal(actual, expected)

    def test_fractional_warp_preserves_constant_intensity_at_borders(self):
        image = np.full((40, 40), 42000, np.uint16)
        result = self.stitcher._compose_images([image], [(-0.375, -0.625)])
        self.assertEqual(result.dtype, np.uint16)
        np.testing.assert_array_equal(result, np.full(result.shape, 42000, np.uint16))

    def test_subpixel_composition_is_more_accurate_than_rounded_positions(self):
        # Analytic signal sampled independently on both tile grids provides a
        # reference without reusing the compositor's interpolation algorithm.
        y, x = np.mgrid[:128, :128]
        signal = lambda x, y: 120 + 45 * np.sin(x * 0.22) + 40 * np.cos(y * 0.19)
        dx, dy = 20.4, 12.4
        a = np.rint(signal(x, y)).astype(np.uint8)
        b = np.rint(signal(x + dx, y + dy)).astype(np.uint8)
        result = self.stitcher._compose_images([a, b], [(0, 0), (dx, dy)])
        rounded = self.stitcher._compose_images([a, b], [(0, 0), (round(dx), round(dy))])
        region = np.s_[25:110, 35:110]
        truth = signal(x, y)[region]
        fine_error = np.mean((result[region] - truth) ** 2)
        coarse_error = np.mean((rounded[region] - truth) ** 2)
        self.assertLess(fine_error, coarse_error * 0.2)

    def test_16bit_feature_image_does_not_change_source(self):
        image = texture().astype(np.uint16) * 257
        original = image.copy()
        edge = self.stitcher._extract_edge_map(image)
        self.assertEqual(edge.dtype, np.uint8)
        np.testing.assert_array_equal(image, original)

    def test_reject_mixed_depth_and_nonfinite_positions(self):
        image = np.ones((20, 20), np.uint8)
        with self.assertRaisesRegex(ValueError, 'same bit depth'):
            self.stitcher._compose_images([image, image.astype(np.uint16)], [(0, 0), (0, 0)])
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.stitcher._compose_images([image], [(np.nan, 0)])

    def test_integer_grid_stitches_end_to_end_without_losing_pixels(self):
        scene = texture(480)
        positions = [(0, 0), (160, 0), (0, 160), (160, 160)]
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, (x, y) in enumerate(positions):
                path = Path(directory) / f'tile_{index}.tif'
                self.assertTrue(cv2.imwrite(str(path), scene[y:y + 320, x:x + 320]))
                paths.append(str(path))
            result = self.stitcher.stitch_sequence(paths, grid_shape=(2, 2))
        # Real SIFT, phase correlation, graph solve and blend are all exercised.
        self.assertLessEqual(abs(result.shape[0] - scene.shape[0]), 2)
        self.assertLessEqual(abs(result.shape[1] - scene.shape[1]), 2)
        # Anchor round-off and small real phase errors can pad the origin by 1px.
        best_mae = min(np.mean(np.abs(result[oy:oy + 478, ox:ox + 478].astype(float) - scene[:478, :478]))
                       for oy in (0, 1) for ox in (0, 1))
        self.assertLess(best_mae, 2.0)

    def test_16bit_tiff_pipeline_preserves_depth_and_subpixel_positions(self):
        scene = texture(100).astype(np.uint16) * 257
        with tempfile.TemporaryDirectory() as directory:
            paths = [str(Path(directory) / f'{i}.tif') for i in range(2)]
            for path in paths:
                self.assertTrue(cv2.imwrite(path, scene))
            with patch.object(self.stitcher, '_try_match_pair', return_value=(-20.375, 0, 20)):
                result = self.stitcher.stitch_sequence(paths)
            output = str(Path(directory) / 'result.tif')
            self.assertTrue(cv2.imwrite(output, result))
            loaded = cv2.imread(output, cv2.IMREAD_UNCHANGED)
        expected = self.stitcher._compose_images([scene, scene], [(0, 0), (20.375, 0)])
        np.testing.assert_array_equal(loaded, expected)
        self.assertEqual(loaded.dtype, np.uint16)


if __name__ == '__main__':
    unittest.main()
