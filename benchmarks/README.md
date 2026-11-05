# Feature-cache baseline

Measured with `python3 tools/benchmark_cache.py --output benchmarks/synthetic-cache.json`.
The JSON contains all six samples, package versions, phase timings, peak RSS and output hashes.

- Dataset: deterministic blurred random texture, 3×3 grid, 384×384 uint8 tiles, 50% overlap; automatic matching (36 pairs).
- Environment: macOS (Darwin), arm64, Python 3.13, OpenCV 4.13, one OpenCV thread.
- Three samples per mode, fresh interpreter per sample, alternating mode order. Timings include image loading, feature processing, matching, graph solving, composition and diagnostics.

| Metric | Uncached | Cached |
|---|---:|---:|
| Feature extractions | 72 | 9 |
| Cache hits | 0 | 63 |
| Median total seconds | 12.286 | 4.754 |
| Median peak process RSS (MiB) | 127.2 | 148.2 |
| Descriptor buffers retained during matching (MiB) | 0 | 19.3 |

Measured total speedup: **2.58×**. All six output hashes are identical. Peak RSS includes native allocations and the interpreter, rather than Python heap alone. Descriptor-buffer bytes exclude Python keypoint objects and other allocator overhead; caching exchanges memory during matching for fewer feature extractions. The cache is released when a task completes, fails or is cancelled.

This synthetic benchmark does not establish microscope-image accuracy or predict speed on another machine. The benchmark script accepts real image paths for local measurement; paths are omitted from its JSON output. Peak RSS is currently reported on macOS/Linux only. No real SEM acquisitions are included in this repository.
