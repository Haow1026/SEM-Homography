"""Reproducible cache A/B timings; fresh process per sample, synthetic data by default."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from main import SEMStitcher


def peak_rss_bytes():
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if sys.platform == 'darwin' else rss * 1024)
    except ImportError:
        return None


def worker(paths, cached):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    stitcher = SEMStitcher()
    stitcher.set_progress_callback(lambda message: None)
    result = stitcher.stitch_sequence(paths, use_feature_cache=cached)
    return dict(metrics=stitcher.last_metrics, peak_rss_bytes=peak_rss_bytes(),
                shape=result.shape, dtype=str(result.dtype), sha256=hashlib.sha256(result.tobytes()).hexdigest())


def run(args, paths):
    samples = {False: [], True: []}
    for repeat in range(args.repeats):
        # Alternate execution order to reduce warm-filesystem ordering effects.
        for cached in ([False, True] if repeat % 2 == 0 else [True, False]):
            command = [sys.executable, str(Path(__file__).resolve()), '--worker']
            if cached:
                command.append('--cached')
            command.extend(paths)
            outcome = subprocess.run(command, capture_output=True, text=True, check=True)
            samples[cached].append(json.loads(outcome.stdout))
    baseline = statistics.median(s['metrics']['total_seconds'] for s in samples[False])
    cached = statistics.median(s['metrics']['total_seconds'] for s in samples[True])
    return dict(dataset='user-supplied images' if args.images else 'synthetic blurred random texture, 3x3 grid',
                tile_size=None if args.images else args.tile_size, input_count=len(paths), repeats=args.repeats,
                platform=platform.system(), architecture=platform.machine(), python=platform.python_version(),
                opencv=cv2.__version__, numpy=np.__version__, opencv_threads=1,
                baseline_median_seconds=baseline, cached_median_seconds=cached, speedup=baseline/cached,
                identical_output_hashes=len({s['sha256'] for group in samples.values() for s in group}) == 1,
                baseline=samples[False], cached=samples[True],
                notes='Fresh process per sample. Peak RSS includes interpreter, sources, descriptors and panorama; '
                      'not Python heap alone. Synthetic performance is not an SEM acquisition accuracy guarantee.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--cached', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--tile-size', type=int, default=384)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    parser.add_argument('images', nargs='*', help='optional real grayscale images (not included in the output report)')
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.images, args.cached)))
        return
    if args.repeats < 1 or args.tile_size < 160:
        parser.error('repeats must be positive and tile-size must be at least 160')
    if args.images and len(args.images) < 2:
        parser.error('at least two images are required')
    with tempfile.TemporaryDirectory() as directory:
        paths = args.images
        if not paths:
            size, step = args.tile_size, args.tile_size // 2
            scene = cv2.GaussianBlur(np.random.default_rng(42).uniform(30, 220, (size+2*step, size+2*step)).astype(np.float32), (0, 0), 1.2)
            scene = cv2.normalize(scene, None, 20, 235, cv2.NORM_MINMAX).astype(np.uint8)
            paths = []
            for row in range(3):
                for col in range(3):
                    path = str(Path(directory) / f'{row}_{col}.tif')
                    if not cv2.imwrite(path, scene[row*step:row*step+size, col*step:col*step+size]):
                        raise OSError(path)
                    paths.append(path)
        report = run(args, paths)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')
    print(json.dumps({key: report[key] for key in ('input_count', 'baseline_median_seconds', 'cached_median_seconds', 'speedup', 'identical_output_hashes')}))


if __name__ == '__main__':
    main()
