import threading
import unittest
from unittest.mock import patch

import numpy as np

from main import SEMStitcher, StitchingCancelled
from tools.blend_reference import compose_full_canvas


class ChunkedBlendingTests(unittest.TestCase):
    def setUp(self):
        self.stitcher = SEMStitcher()
        self.stitcher.set_progress_callback(lambda message: None)

    def test_matches_previous_compositor_across_blocks_depths_and_layouts(self):
        rng = np.random.default_rng(81)
        shapes = [(53, 65), (57, 60), (50, 69), (55, 61)]
        layouts = [
            [(-8, -3), (24, -3), (-8, 28), (24, 28)],
            [(-8.375, -3.625), (24.25, -3.125), (-8.5, 28.75), (24.125, 28.375)],
            [(0, 0), (79.9, 0.02), (0.01, 71.99), (80, 72)],  # gaps / partial blocks
            [(0, 0)] * 4,  # complete overlap
        ]
        for dtype in (np.uint8, np.uint16):
            images = [rng.integers(0, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
                      for shape in shapes]
            originals = [image.copy() for image in images]
            for positions in layouts:
                expected = compose_full_canvas(images, positions)
                for block_size in (1, 7, 16, 31, 64, 1024):
                    with self.subTest(dtype=dtype, positions=positions, block_size=block_size):
                        actual = self.stitcher._compose_images(images, positions, block_size=block_size)
                        np.testing.assert_array_equal(actual, expected)
            for image, original in zip(images, originals):
                np.testing.assert_array_equal(image, original)

    def test_interpolation_rounding_at_half_bins_is_consistent_at_block_boundaries(self):
        rng = np.random.default_rng(311)
        images = [rng.integers(0, 65536, (17, 23), dtype=np.uint16) for _ in range(2)]
        for fraction in (1/64 - 1e-6, 1/64, 1/64 + 1e-6,
                         31/64, 33/64, 63/64, 0.99999):
            positions = [(0, 0), (4+fraction, -3-fraction)]
            with self.subTest(fraction=fraction):
                expected = compose_full_canvas(images, positions)
                actual = self.stitcher._compose_images(images, positions, block_size=8)
                np.testing.assert_array_equal(actual, expected)

    def test_constant_16bit_junction_has_no_new_seams_at_block_boundaries(self):
        images = [np.full((133, 141), 42000, np.uint16) for _ in range(4)]
        positions = [(-0.375, -0.625), (65.25, -0.625), (-0.375, 63.5), (65.25, 63.5)]
        actual = self.stitcher._compose_images(images, positions, block_size=32)
        np.testing.assert_array_equal(actual, np.full(actual.shape, 42000, np.uint16))

    def test_default_blocks_preserve_pixels_of_a_tile_larger_than_a_block(self):
        image = np.random.default_rng(33).integers(0, 65536, (1031, 1049), dtype=np.uint16)
        actual = self.stitcher._compose_images([image], [(0, 0)])
        np.testing.assert_array_equal(actual, image)
        self.assertEqual(self.stitcher.last_metrics['blend_blocks'], 4)
        self.assertEqual(self.stitcher.last_metrics['blend_accumulator_bytes'], 2*1024*1024*8)

    def test_default_blocks_match_previous_subpixel_compositor_on_large_tiles(self):
        rng = np.random.default_rng(77)
        images = [rng.integers(0, 65536, (1031, 1049), dtype=np.uint16) for _ in range(2)]
        positions = [(-0.375, -0.625), (517.28125, 17.71875)]
        actual = self.stitcher._compose_images(images, positions)
        np.testing.assert_array_equal(actual, compose_full_canvas(images, positions))

    def test_fractional_tile_wider_than_opencv_remap_limit(self):
        # Cropping before warp also avoids OpenCV's 32767-pixel remap limit.
        image = np.full((3, 32780), 191, np.uint8)
        actual = self.stitcher._compose_images([image], [(0.375, 0.625)], block_size=512)
        np.testing.assert_array_equal(actual, np.full(actual.shape, 191, np.uint8))

    def test_one_pixel_sources_gaps_and_black_pixels(self):
        for dtype in (np.uint8, np.uint16):
            images = [np.array([[0]], dtype=dtype), np.array([[np.iinfo(dtype).max]], dtype=dtype)]
            for positions in ([(0, 0), (4.375, 3.25)], [(0, 0), (0.375, 0.25)]):
                actual = self.stitcher._compose_images(images, positions, block_size=1)
                np.testing.assert_array_equal(actual, compose_full_canvas(images, positions))

    def test_cancel_between_blocks_of_a_single_large_tile(self):
        cancel = threading.Event()
        self.stitcher.set_cancel_callback(cancel.is_set)
        accumulate = self.stitcher._blend_onto_canvas

        def first_block(*args):
            accumulate(*args)
            cancel.set()

        image = np.ones((80, 80), np.uint8)
        with patch.object(self.stitcher, '_blend_onto_canvas', side_effect=first_block) as blend:
            with self.assertRaises(StitchingCancelled):
                self.stitcher._compose_images([image], [(0.25, 0.5)], block_size=32)
        self.assertEqual(blend.call_count, 1)
        self.assertEqual(self.stitcher.last_metrics['blend_blocks'], 0)

    def test_invalid_block_size_is_rejected(self):
        for block_size in (0, -1, 2.5, True, '64'):
            with self.subTest(block_size=block_size):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    self.stitcher._compose_images([np.ones((2, 2), np.uint8)], [(0, 0)],
                                                  block_size=block_size)


if __name__ == '__main__':
    unittest.main()
