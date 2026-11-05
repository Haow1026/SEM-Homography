from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from image_io import read_image, write_image


class ImageIOTests(unittest.TestCase):
    def test_unicode_file_api_round_trip_tiff_png_both_depths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / '中文と日本語'
            folder.mkdir()
            for dtype in (np.uint8, np.uint16):
                image = np.random.default_rng(17).integers(0, np.iinfo(dtype).max + 1,
                                                          (35, 47), dtype=dtype)
                for extension in ('.tif', '.png'):
                    path = folder / ('图像' + extension)
                    # Exercise the wide-path implementation on every OS, then
                    # the normal dispatch (actual wide-path IO on Windows CI).
                    with self.subTest(dtype=dtype, extension=extension):
                        with patch('image_io._needs_unicode_io', return_value=True):
                            self.assertTrue(write_image(path, image))
                            np.testing.assert_array_equal(read_image(path), image)
                        np.testing.assert_array_equal(read_image(path), image)

    def test_missing_empty_and_corrupt_unicode_files_return_no_image(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('image_io._needs_unicode_io', return_value=True):
            path = Path(directory) / '图像.tif'
            self.assertIsNone(read_image(path))
            path.write_bytes(b'')
            self.assertIsNone(read_image(path))
            path.write_bytes(b'invalid TIFF data')
            self.assertIsNone(read_image(path))

    def test_ascii_dispatch_still_uses_opencv_file_api(self):
        image = np.ones((2, 2), np.uint8)
        with patch('image_io.cv2.imread', return_value=image) as read, \
             patch('image_io.cv2.imwrite', return_value=True) as write:
            self.assertIs(read_image('input.tif'), image)
            self.assertTrue(write_image('output.tif', image))
        read.assert_called_once()
        write.assert_called_once()

    def test_failed_encode_does_not_create_a_destination(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('image_io._needs_unicode_io', return_value=True), \
             patch('image_io.cv2.imencode', return_value=(False, None)):
            path = Path(directory) / '结果.tif'
            self.assertFalse(write_image(path, np.ones((2, 2), np.uint8)))
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
