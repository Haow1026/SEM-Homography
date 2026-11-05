"""Versioned stitch projects: JSON metadata, verified sources, no embedded images."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import cv2
import numpy as np

from inspection import PairDiagnostic, StitchInspection, canvas_geometry, file_signature
from image_io import read_image
from main import SEMStitcher, StitchingCancelled
from stitching_task import StitchRequest

FORMAT = 'sem-stitch-project'
VERSION = 1
BLEND = 'distance-feather-opencv-linear-v1'
MAX_PROJECT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class RestoredProject:
    request: StitchRequest
    panorama: np.ndarray | None
    inspection: StitchInspection | None
    metrics: dict


def _check(cancelled):
    if cancelled():
        raise StitchingCancelled('Project operation cancelled')


def _digest(path, cancelled):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        while True:
            _check(cancelled)
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _image_digest(image, cancelled):
    # Hash without a panorama-sized bytes copy; also allow cancellation.
    data = memoryview(np.ascontiguousarray(image)).cast('B')
    digest = hashlib.sha256()
    for start in range(0, len(data), 1024 * 1024):
        _check(cancelled)
        digest.update(data[start:start + 1024 * 1024])
    return digest.hexdigest()


def _relative_path(path, parent):
    try:
        return Path(os.path.relpath(path, parent)).as_posix()
    except ValueError:  # Different Windows drives cannot use a relative path.
        return Path(path).as_posix()


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _vector(value, integers=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    if integers:
        return all(_integer(v, 1) for v in value)
    return all(type(v) in (int, float) and math.isfinite(v) for v in value)


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _validate(document, parent):
    """Validate before source IO or allocation; reject future/unsupported formats."""
    if (not isinstance(document, dict) or document.get('format') != FORMAT or 'alignment' not in document
            or type(document.get('version')) is not int or document['version'] != VERSION):
        raise ValueError('Unsupported SEM project format or version')
    sources, scan = document.get('sources'), document.get('scan')
    if not isinstance(sources, list) or len(sources) < 2 or not isinstance(scan, dict):
        raise ValueError('Project requires at least two sources and scan settings')
    paths = []
    for source in sources:
        if (not isinstance(source, dict) or not isinstance(source.get('path'), str)
                or not source['path'] or '\x00' in source['path']
                or not _integer(source.get('size')) or not _hash(source.get('sha256'))):
            raise ValueError('Invalid project source path or fingerprint')
        paths.append(str((parent / source['path']).resolve()))
    if type(scan.get('serpentine')) is not bool or type(scan.get('first_row_rtl')) is not bool:
        raise ValueError('Invalid project scan directions')
    grid = scan.get('grid_shape')
    if grid is not None and not _vector(grid, integers=True):
        raise ValueError('Invalid project grid shape')
    request = StitchRequest(tuple(paths), grid, scan['serpentine'], scan['first_row_rtl'])
    alignment = document.get('alignment')
    if alignment is None:
        return request
    if (not isinstance(alignment, dict) or alignment.get('blend') != BLEND
            or alignment.get('dtype') not in ('uint8', 'uint16')):
        raise ValueError('Unsupported project alignment or bit depth')
    shapes, positions = alignment.get('shapes'), alignment.get('positions')
    if (not isinstance(shapes, list) or len(shapes) != len(paths)
            or not all(_vector(shape, integers=True) for shape in shapes)
            or not isinstance(positions, list) or len(positions) != len(paths)
            or not all(_vector(position) for position in positions)):
        raise ValueError('Invalid project image shapes or positions')
    # The stored origin is already normalized. Avoid overflow in geometry and
    # reject corrupt coordinates before allocating a panorama.
    if any(not 0 <= p[axis] <= np.iinfo(np.int32).max - s[1-axis]
           for p, s in zip(positions, shapes) for axis in (0, 1)):
        raise ValueError('Project coordinates are outside the supported range')
    normalized, size = canvas_geometry(shapes, positions)
    output = alignment.get('output')
    if (not isinstance(output, dict) or not _vector(output.get('shape'), integers=True)
            or tuple(output['shape']) != size[::-1] or not _hash(output.get('sha256'))
            or not np.allclose(normalized, positions, rtol=0, atol=1e-7)):
        raise ValueError('Project panorama geometry or fingerprint is invalid')
    pairs = alignment.get('pairs')
    if not isinstance(pairs, list):
        raise ValueError('Invalid project matching diagnostics')
    for pair in pairs:
        if (not isinstance(pair, dict) or not _integer(pair.get('i')) or not _integer(pair.get('j'))
                or pair['i'] == pair['j'] or pair['i'] >= len(paths) or pair['j'] >= len(paths)
                or not _vector(pair.get('measured')) or not _vector(pair.get('predicted'))
                or not _integer(pair.get('inliers'), 1) or type(pair.get('excluded')) is not bool
                or type(pair.get('residual')) not in (int, float)
                or not math.isfinite(pair['residual']) or pair['residual'] < 0):
            raise ValueError('Invalid project matching diagnostic')
        predicted = np.subtract(positions[pair['i']], positions[pair['j']])
        residual = np.linalg.norm(predicted - pair['measured'])
        if (not np.allclose(predicted, pair['predicted'], rtol=0, atol=1e-5)
                or not math.isclose(float(residual), pair['residual'], rel_tol=1e-6, abs_tol=1e-5)):
            raise ValueError('Project diagnostics disagree with saved positions')
    return request


def save_project(path, request, inspection=None, panorama=None,
                 cancelled=lambda: False, log=lambda message: None):
    """Atomically save inputs alone, or inputs plus a completed alignment."""
    destination = Path(path).resolve()
    paths = tuple(str(Path(p).resolve()) for p in request.paths)
    if os.path.normcase(str(destination)) in {os.path.normcase(p) for p in paths}:
        raise ValueError('A project cannot overwrite a source image')
    if (inspection is None) != (panorama is None):
        raise ValueError('Both panorama and inspection are required for a completed project')
    if inspection is not None and tuple(str(Path(p).resolve()) for p in inspection.paths) != paths:
        raise ValueError('Inspection does not match the acquisition file order')
    document = dict(format=FORMAT, version=VERSION, sources=[],
                    scan=dict(grid_shape=request.grid_shape, serpentine=request.serpentine,
                              first_row_rtl=request.first_row_rtl), alignment=None)
    signatures = []
    for index, source in enumerate(paths):
        _check(cancelled)
        before = file_signature(source)
        if before is None:
            raise FileNotFoundError(f'Source image is missing: {source}')
        if inspection is not None and before != inspection.signatures[index]:
            raise ValueError(f'Source image changed since stitching: {source}')
        digest = _digest(source, cancelled)
        if file_signature(source) != before:
            raise ValueError(f'Source image changed while saving: {source}')
        document['sources'].append(dict(path=_relative_path(source, destination.parent),
                                        size=before[0], sha256=digest))
        signatures.append(before)
        log(f'  Verified source {index + 1}/{len(paths)}: {os.path.basename(source)}')
    if inspection is not None:
        SEMStitcher._validate_images([panorama])
        document['alignment'] = dict(
            blend=BLEND, dtype=str(panorama.dtype), shapes=inspection.shapes,
            positions=inspection.positions, pairs=[asdict(pair) for pair in inspection.pairs],
            output=dict(shape=panorama.shape, sha256=_image_digest(panorama, cancelled)),
            versions=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__))
    # JSON round-trip normalizes tuples and catches non-finite/unsupported data.
    encoded = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if len(encoded) + 1 > MAX_PROJECT_BYTES:
        raise ValueError('Project metadata exceeds 16 MiB')
    _validate(json.loads(encoded), destination.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix='.' + destination.name + '.',
                                         suffix='.tmp', dir=destination.parent, delete=False) as output:
            temporary = output.name
            output.write(encoded + b'\n')
            output.flush()
            os.fsync(output.fileno())
        _check(cancelled)
        if any(file_signature(p) != s for p, s in zip(paths, signatures)):
            raise ValueError('A source image changed while saving the project')
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return str(destination)


def restore_project(path, cancelled=lambda: False, log=lambda message: None):
    """Verify sources and recompose saved positions without feature matching."""
    _check(cancelled)
    project_path = Path(path).resolve()
    with project_path.open('rb') as source:
        encoded = source.read(MAX_PROJECT_BYTES + 1)
    if len(encoded) > MAX_PROJECT_BYTES:
        raise ValueError('Project metadata exceeds 16 MiB')
    try:
        document = json.loads(encoded)
        request = _validate(document, project_path.parent)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise ValueError(f'Cannot read SEM project: {error}') from error
    alignment = document['alignment']
    images, signatures = [], []
    for index, (path, expected) in enumerate(zip(request.paths, document['sources'])):
        _check(cancelled)
        before = file_signature(path)
        if before is None:
            raise FileNotFoundError(f'Source image is missing: {path}')
        if before[0] != expected['size'] or _digest(path, cancelled) != expected['sha256']:
            raise ValueError(f'Source image changed: {path}')
        if alignment is not None:
            image = read_image(path)
            if (image is None or image.ndim != 2 or tuple(image.shape) != tuple(alignment['shapes'][index])
                    or str(image.dtype) != alignment['dtype']):
                raise ValueError(f'Source image shape/bit depth does not match project: {path}')
            images.append(image)
        if file_signature(path) != before:
            raise ValueError(f'Source image changed while restoring: {path}')
        signatures.append(before)
        log(f'  Verified source {index + 1}/{len(request.paths)}: {os.path.basename(path)}')
    _check(cancelled)
    if any(file_signature(p) != s for p, s in zip(request.paths, signatures)):
        raise ValueError('A source image changed while restoring the project')
    if alignment is None:
        return RestoredProject(request, None, None, {})
    stitcher = SEMStitcher()
    stitcher.set_cancel_callback(cancelled)
    stitcher.set_progress_callback(log)
    result = stitcher._compose_images(images, alignment['positions'])
    if _image_digest(result, cancelled) != alignment['output']['sha256']:
        raise ValueError('Restored pixels differ from the saved result; check software versions and stitch again')
    if any(file_signature(p) != s for p, s in zip(request.paths, signatures)):
        raise ValueError('A source image changed while restoring the project')
    pairs = tuple(PairDiagnostic(pair['i'], pair['j'], tuple(pair['measured']), tuple(pair['predicted']),
                                 pair['inliers'], pair['residual'], pair['excluded'])
                  for pair in alignment['pairs'])
    report = StitchInspection(request.paths, tuple(map(tuple, alignment['shapes'])),
                              tuple(map(tuple, alignment['positions'])), tuple(signatures), pairs)
    return RestoredProject(request, result, report, dict(stitcher.last_metrics, restored_project=True))
