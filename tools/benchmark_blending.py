"""Measure full-canvas vs blocked composition in fresh processes (no SIFT/IO)."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from main import SEMStitcher
from tools.blend_reference import compose_full_canvas
from tools.benchmark_cache import peak_rss_bytes


def worker(size, depth, backend, block_size):
    cv2.setNumThreads(1)
    rng = np.random.default_rng(822)
    dtype = np.uint8 if depth == 8 else np.uint16
    images = [rng.integers(0, np.iinfo(dtype).max + 1, (size, size), dtype=dtype)
              for _ in range(9)]
    # Negative origin and fractional steps exercise halos, block boundaries,
    # partial last blocks and four-way intersections at 50% overlap.
    positions = [(col * (size // 2 + 0.375) - 0.375,
                  row * (size // 2 + 0.625) - 0.625)
                 for row in range(3) for col in range(3)]
    stitcher = SEMStitcher()
    stitcher.set_progress_callback(lambda message: None)
    started = time.perf_counter()
    if backend == 'full':
        result = compose_full_canvas(images, positions)
    else:
        result = stitcher._compose_images(images, positions, block_size=block_size)
    elapsed = time.perf_counter() - started
    peak = peak_rss_bytes()
    return dict(seconds=elapsed, peak_rss_bytes=peak, shape=list(result.shape), dtype=str(result.dtype),
                sha256=hashlib.sha256(memoryview(result)).hexdigest(),
                input_bytes=sum(image.nbytes for image in images), output_bytes=result.nbytes,
                accumulator_bytes=(16 * result.size if backend == 'full' else
                                   stitcher.last_metrics['blend_accumulator_bytes']),
                blocks=None if backend == 'full' else stitcher.last_metrics['blend_blocks'])


def run(args):
    cases = []
    for size in args.tile_sizes:
        for depth in args.depths:
            samples = {'full': [], 'blocked': []}
            for repeat in range(args.repeats):
                for backend in (['full', 'blocked'] if repeat % 2 == 0 else ['blocked', 'full']):
                    command = [sys.executable, str(Path(__file__).resolve()), '--worker', backend,
                               '--tile-sizes', str(size), '--depths', str(depth),
                               '--block-size', str(args.block_size)]
                    outcome = subprocess.run(command, capture_output=True, text=True, check=True)
                    samples[backend].append(json.loads(outcome.stdout))
            case = dict(tile_size=size, depth=depth, samples=samples,
                        identical_output_hashes=len({sample['sha256'] for group in samples.values()
                                                      for sample in group}) == 1)
            for backend, group in samples.items():
                case[f'{backend}_median_seconds'] = statistics.median(s['seconds'] for s in group)
                peaks = [s['peak_rss_bytes'] for s in group if s['peak_rss_bytes'] is not None]
                case[f'{backend}_median_peak_rss_bytes'] = statistics.median(peaks) if peaks else None
            full_peak, blocked_peak = (case[f'{b}_median_peak_rss_bytes'] for b in ('full', 'blocked'))
            case['peak_rss_reduction_fraction'] = 1 - blocked_peak/full_peak if full_peak else None
            if not case['identical_output_hashes']:
                raise RuntimeError(f'Output mismatch for {size}px / {depth} bit')
            cases.append(case)
            print(json.dumps({key: value for key, value in case.items() if key != 'samples'}), flush=True)
    return dict(dataset='deterministic random grayscale, 3x3 grid, known fractional positions, 50% overlap',
                seed=822, input_count=9, repeats=args.repeats, block_size=args.block_size,
                platform=platform.system(), architecture=platform.machine(), python=platform.python_version(),
                opencv=cv2.__version__, numpy=np.__version__, opencv_threads=1, cases=cases,
                notes='Fresh process per sample; alternating backend order. Times measure composition only. '
                      'Peak RSS includes interpreter, original tiles, panorama and native allocations. '
                      'Matching, feature cache, file IO, GUI and export are excluded. '
                      'Outputs are compared by SHA256, without changing source intensities. '
                      'Synthetic results do not predict full-task memory or real SEM alignment quality. '
                      'Peak RSS is available on macOS/Linux only.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('full', 'blocked'), help=argparse.SUPPRESS)
    parser.add_argument('--tile-sizes', nargs='+', type=int, default=[1024, 2048])
    parser.add_argument('--depths', nargs='+', type=int, choices=(8, 16), default=[8, 16])
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.tile_sizes) < 2 or args.repeats < 1 or not 1 <= args.block_size <= 16384:
        parser.error('tile sizes must be >=2; repeats positive; block-size must be 1..16384')
    if args.worker:
        print(json.dumps(worker(args.tile_sizes[0], args.depths[0], args.worker, args.block_size)))
        return
    report = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
