"""Frozen full-canvas compositor from 6c5e0bd, for regression/benchmarks only.

Keep this independent of the blocked compositor so a shared cropping/weighting
bug cannot make the A/B check pass. Do not use it for production large images.
"""
import cv2
import numpy as np

from inspection import canvas_geometry


def compose_full_canvas(images, positions):
    dtype = images[0].dtype
    positions, (width, height) = canvas_geometry([image.shape for image in images], positions)
    total = np.zeros((height, width), np.float64)
    total_weight = np.zeros_like(total)
    for image, (x, y) in zip(images, positions):
        h, w = image.shape
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = int(np.ceil(x + w)), int(np.ceil(y + h))
        x_dist = np.minimum(np.arange(w) + 1, w - np.arange(w))
        y_dist = np.minimum(np.arange(h) + 1, h - np.arange(h))
        weights = np.minimum(y_dist[:, None], x_dist[None, :]).astype(np.float64)
        weighted = image.astype(np.float64) * weights
        fx, fy = x - x0, y - y0
        if fx != 0.0 or fy != 0.0:
            transform = np.array([[1, 0, fx], [0, 1, fy]], np.float64)
            size = (x1-x0, y1-y0)
            weighted = cv2.warpAffine(weighted, transform, size, flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            weights = cv2.warpAffine(weights, transform, size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        total[y0:y1, x0:x1] += weighted
        total_weight[y0:y1, x0:x1] += weights
    np.divide(total, total_weight, out=total, where=total_weight > 0)
    return np.clip(np.rint(total), 0, np.iinfo(dtype).max).astype(dtype)
