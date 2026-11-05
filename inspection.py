"""Read-only stitch diagnostics and viewport sampling, independent of Tk."""
from dataclasses import dataclass, replace
import os

import numpy as np

from image_io import read_image


def file_signature(path):
    try:
        info = os.stat(path)
        return info.st_size, info.st_mtime_ns
    except OSError:
        return None


@dataclass(frozen=True)
class PairDiagnostic:
    i: int
    j: int
    measured: tuple
    predicted: tuple
    inliers: int
    residual: float
    excluded: bool


@dataclass(frozen=True)
class StitchInspection:
    paths: tuple
    shapes: tuple
    positions: tuple
    signatures: tuple
    pairs: tuple

    def reordered(self, old_to_new):
        """Map algorithm tile indices back to the GUI acquisition-file order."""
        count = len(self.paths)
        if sorted(old_to_new) != list(range(count)):
            raise ValueError('Expected a permutation of tile indices')
        def reorder(values):
            result = [None] * count
            for old, new in enumerate(old_to_new):
                result[new] = values[old]
            return tuple(result)
        return StitchInspection(
            reorder(self.paths), reorder(self.shapes), reorder(self.positions), reorder(self.signatures),
            tuple(replace(p, i=old_to_new[p.i], j=old_to_new[p.j]) for p in self.pairs),
        )

    def load_pair(self, index):
        """Reload only two source tiles; reject changed/missing source files."""
        pair = self.pairs[index]
        images = []
        for tile in (pair.i, pair.j):
            path = self.paths[tile]
            signature = file_signature(path)
            if signature is None:
                raise FileNotFoundError(path)
            if self.signatures[tile] is not None and signature != self.signatures[tile]:
                raise ValueError(f'Source image changed: {path}')
            image = read_image(path)
            if (image is None or image.ndim != 2 or image.shape != self.shapes[tile]
                    or image.dtype not in (np.uint8, np.uint16)):
                raise ValueError(f'Cannot inspect source image: {path}')
            if file_signature(path) != signature:
                raise ValueError(f'Source image changed while reading: {path}')
            images.append(image)
        return pair, tuple(images)


def canvas_geometry(shapes, positions):
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (len(shapes), 2) or not np.isfinite(positions).all():
        raise ValueError('Each image requires a finite (x, y) position')
    rounded = np.rint(positions)
    positions = np.where(np.abs(positions - rounded) < 1e-7, rounded, positions)
    positions = positions - np.floor(positions.min(axis=0))
    sizes = np.array([(shape[1], shape[0]) for shape in shapes])
    size = tuple(np.ceil((positions + sizes).max(axis=0)).astype(int))
    return positions, size


class Viewport:
    """Pan and zoom in panorama pixel coordinates; 1.0 means 100%."""
    def __init__(self, image_size, view_size=(640, 480)):
        self.image_size = image_size
        self.view_size = view_size
        self.origin = np.zeros(2, dtype=float)
        self.zoom = 1.0

    def fit(self):
        self.zoom = min(v / i for v, i in zip(self.view_size, self.image_size))
        self.origin[:] = 0
        self.clamp()

    def clamp(self):
        extent = np.array(self.view_size) / self.zoom
        maximum = np.maximum(0, np.array(self.image_size) - extent)
        self.origin[:] = np.clip(self.origin, 0, maximum)
        if abs(self.zoom - 1.0) < 1e-12:
            self.origin[:] = np.floor(self.origin + 1e-7)

    def zoom_at(self, zoom, point=None):
        point = np.array(point if point is not None else np.array(self.view_size) / 2)
        world = self.origin + point / self.zoom
        self.zoom = min(8.0, max(1e-4, float(zoom)))
        self.origin = world - point / self.zoom
        self.clamp()

    def center(self, point, zoom=1.0):
        self.zoom = zoom
        self.origin = np.array(point) - np.array(self.view_size) / (2 * zoom)
        self.clamp()

    def pan(self, dx, dy):
        self.origin -= np.array([dx, dy]) / self.zoom
        self.clamp()


def sample_view(image, position, viewport):
    """Sample only a viewport, without resizing the full image or canvas.

    NumPy sampling also works with panorama dimensions above OpenCV remap's
    32767-pixel limit. Integer coordinates preserve exact source values.
    """
    width, height = viewport.view_size
    xs = viewport.origin[0] + np.arange(width) / viewport.zoom - position[0]
    ys = viewport.origin[1] + np.arange(height) / viewport.zoom - position[1]
    h, w = image.shape
    valid = ((ys[:, None] >= 0) & (ys[:, None] <= h - 1)
             & (xs[None, :] >= 0) & (xs[None, :] <= w - 1))
    xs, ys = np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (xs - x0).astype(np.float32), (ys - y0).astype(np.float32)
    top = image[y0[:, None], x0].astype(np.float32) * (1 - fx) + image[y0[:, None], x1] * fx
    bottom = image[y1[:, None], x0].astype(np.float32) * (1 - fx) + image[y1[:, None], x1] * fx
    sampled = top * (1 - fy[:, None]) + bottom * fy[:, None]
    sampled[~valid] = 0
    return sampled, valid


def display_range(images):
    if all(image.dtype == np.uint8 for image in images):
        return 0, 255
    return min(int(image.min()) for image in images), max(int(image.max()) for image in images)


def display_gray(sampled, valid, limits):
    low, high = limits
    if high <= low:
        # Preserve visibility of a nonzero constant scientific image.
        shown = np.full(sampled.shape, 128 if high else 0, np.uint8)
    else:
        shown = np.clip(np.rint((sampled - low) * (255.0 / (high - low))), 0, 255).astype(np.uint8)
    shown[~valid] = 0
    return shown


def pair_frame(images, positions, viewport, mode, limits, alternate_b=False):
    a, mask_a = sample_view(images[0], positions[0], viewport)
    b, mask_b = sample_view(images[1], positions[1], viewport)
    a, b = display_gray(a, mask_a, limits), display_gray(b, mask_b, limits)
    if mode == 'overlay':
        return np.dstack((a, b, b))
    frame = b if mode == 'b' or (mode == 'alternate' and alternate_b) else a
    return np.repeat(frame[:, :, None], 3, axis=2)
