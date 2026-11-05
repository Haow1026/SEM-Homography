import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from inspection import (PairDiagnostic, StitchInspection, Viewport, canvas_geometry,
                        display_gray, display_range, file_signature, pair_frame, sample_view)


class InspectionPixelsTests(unittest.TestCase):
    def test_100_percent_samples_original_pixels_without_altering_16bit_source(self):
        image = (np.arange(40 * 50, dtype=np.uint16).reshape(40, 50) * 31)
        original = image.copy()
        view = Viewport((50, 40), (10, 8))
        view.origin[:] = (13, 7)
        sampled, valid = sample_view(image, (0, 0), view)
        np.testing.assert_array_equal(sampled, image[7:15, 13:23])
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(image, original)

    def test_subpixel_tile_offset_has_correct_interpolated_coordinates(self):
        image = np.tile(np.arange(10, dtype=np.uint16) * 100, (10, 1))
        view = Viewport((20, 20), (4, 4))
        view.origin[:] = (3, 3)
        sampled, mask = sample_view(image, (1.5, 1.25), view)
        np.testing.assert_array_equal(sampled, np.tile([150, 250, 350, 450], (4, 1)))
        self.assertTrue(mask.all())

    def test_panorama_wider_than_opencv_limit_renders_only_viewport(self):
        image = np.full((8, 40000), 30000, np.uint16)
        view = Viewport((40000, 8), (20, 5))
        view.origin[:] = (39000, 1)
        sampled, mask = sample_view(image, (0, 0), view)
        self.assertEqual(sampled.shape, (5, 20))
        self.assertTrue(mask.all())
        np.testing.assert_array_equal(sampled, np.full((5, 20), 30000))

    def test_outside_tile_is_masked_and_not_filled_with_clamped_edge_pixels(self):
        view = Viewport((20, 20), (6, 6))
        sampled, valid = sample_view(np.full((4, 4), 200, np.uint8), (2, 2), view)
        self.assertEqual(int(valid.sum()), 16)
        self.assertEqual(int(sampled[:2].sum()), 0)

    def test_overlay_aligned_tiles_are_gray_and_alternation_matches_each_source(self):
        a = np.full((8, 8), 100, np.uint8)
        b = np.full((8, 8), 180, np.uint8)
        view = Viewport((8, 8), (8, 8))
        same = pair_frame([a, a], [(0, 0), (0, 0)], view, 'overlay', (0, 255))
        np.testing.assert_array_equal(same, np.full((8, 8, 3), 100))
        overlay = pair_frame([a, b], [(0, 0), (0, 0)], view, 'overlay', (0, 255))
        np.testing.assert_array_equal(overlay[0, 0], [100, 180, 180])
        for mode, alternate_b, expected in [('a', False, 100), ('b', False, 180),
                                            ('alternate', False, 100), ('alternate', True, 180)]:
            actual = pair_frame([a, b], [(0, 0), (0, 0)], view, mode, (0, 255), alternate_b)
            np.testing.assert_array_equal(actual, np.full((8, 8, 3), expected))

    def test_display_scaling_uses_one_range_for_both_16bit_sources(self):
        images = [np.array([[1000, 2000]], np.uint16), np.array([[3000, 5000]], np.uint16)]
        limits = display_range(images)
        self.assertEqual(limits, (1000, 5000))
        shown = display_gray(images[0].astype(float), np.ones((1, 2), bool), limits)
        np.testing.assert_array_equal(shown, [[0, 64]])
        self.assertEqual(images[0].dtype, np.uint16)

    def test_zoom_keeps_the_cursor_on_the_same_image_coordinate(self):
        view = Viewport((2000, 2000), (200, 200))
        view.origin[:] = (300, 400)
        point = np.array([50, 70])
        before = view.origin + point / view.zoom
        view.zoom_at(2.0, point)
        np.testing.assert_allclose(view.origin + point / view.zoom, before)
        view.pan(20, -40)
        np.testing.assert_allclose(view.origin, [315, 455])

    def test_canvas_geometry_keeps_report_and_blend_origins_consistent(self):
        positions, size = canvas_geometry([(20, 30), (20, 30)], [(-0.25, 0), (10.125, 2.5)])
        np.testing.assert_allclose(positions, [(0.75, 0), (11.125, 2.5)])
        self.assertEqual(size, (42, 23))


class InspectionSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = tuple(str(Path(self.directory.name) / f'{i}.tif') for i in range(2))
        for path in self.paths:
            cv2.imwrite(path, np.full((30, 40), 12000, np.uint16))
        self.report = StitchInspection(self.paths, ((30, 40), (30, 40)), ((0, 0), (20, 0)),
                                       tuple(file_signature(p) for p in self.paths),
                                       (PairDiagnostic(0, 1, (-20, 0), (-20, 0), 50, 0.0, False),))

    def test_reload_only_selected_pair_and_preserve_bit_depth(self):
        pair, images = self.report.load_pair(0)
        self.assertEqual((pair.i, pair.j), (0, 1))
        self.assertTrue(all(image.dtype == np.uint16 for image in images))

    def test_changed_and_missing_sources_have_explicit_errors(self):
        cv2.imwrite(self.paths[0], np.ones((15, 20), np.uint8))
        with self.assertRaisesRegex(ValueError, 'Source image changed'):
            self.report.load_pair(0)
        Path(self.paths[0]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.report.load_pair(0)

    def test_report_reordering_preserves_pair_direction_and_source_identity(self):
        report = self.report.reordered([1, 0])
        self.assertEqual(report.paths, tuple(reversed(self.paths)))
        self.assertEqual((report.pairs[0].i, report.pairs[0].j), (1, 0))
        self.assertEqual(report.pairs[0].predicted, (-20, 0))
        pair, images = report.load_pair(0)
        self.assertEqual(pair.i, 1)
        self.assertTrue(all(image.dtype == np.uint16 for image in images))


if __name__ == '__main__':
    unittest.main()
