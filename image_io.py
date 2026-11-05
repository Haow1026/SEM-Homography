"""Preserve OpenCV image data while supporting Unicode Windows file paths."""
import os

import cv2
import numpy as np


def _needs_unicode_io(path):
    return os.name == 'nt' and not path.isascii()


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    path = os.fspath(path)
    if not _needs_unicode_io(path):
        return cv2.imread(path, flags)
    # Python opens the wide-character Windows path; OpenCV only sees bytes.
    try:
        with open(path, 'rb') as source:
            encoded = np.fromfile(source, dtype=np.uint8)
    except OSError:
        return None
    if not encoded.size:
        return None
    return cv2.imdecode(encoded, flags)


def write_image(path, image):
    path = os.fspath(path)
    if not _needs_unicode_io(path):
        return cv2.imwrite(path, image)
    success, encoded = cv2.imencode(os.path.splitext(path)[1], image)
    if not success:
        return False
    with open(path, 'wb') as destination:
        destination.write(memoryview(encoded))
    return True
