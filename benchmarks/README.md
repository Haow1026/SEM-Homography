# Performance baselines

## Feature-cache baseline (before blocked composition)

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

Measured total speedup: **2.58×**. All six output hashes are identical. These historical samples were collected for commit `6c5e0bd`, before blocked composition. Peak RSS includes native allocations and the interpreter, rather than Python heap alone. Descriptor-buffer bytes exclude Python keypoint objects and other allocator overhead; caching exchanges memory during matching for fewer feature extractions. The current implementation releases the cache after pairwise matching, before composition, as well as on failure/cancellation.

This synthetic benchmark does not establish microscope-image accuracy or predict speed on another machine. The benchmark script accepts real image paths for local measurement; paths are omitted from its JSON output. Peak RSS is currently reported on macOS/Linux only. No real SEM acquisitions are included in this repository.

## Blocked composition

Measured with `python3 tools/benchmark_blending.py --output benchmarks/synthetic-blending.json`.
The JSON includes all 24 samples, package versions, times, peak RSS, output shapes and SHA256 hashes.

- Dataset: nine deterministic random grayscale tiles, 3×3 grid, known fractional positions, approximately 50% overlap. Negative origin, partial last blocks and four-tile intersections are included. Both uint8 and uint16 are measured at two tile sizes.
- Environment: macOS arm64, Python 3.13, OpenCV 4.13, one OpenCV thread. Three fresh interpreter processes per backend and case, alternating execution order.
- Reference: independent frozen copy of the full-canvas compositor from `6c5e0bd` in `tools/blend_reference.py`. The production path uses blocks up to 1024×1024 pixels.
- Times measure composition only. Peak process RSS includes the interpreter, original tiles, output and native allocations; matching, descriptors, file IO, GUI and export are excluded. This is not a full-stitch peak-memory measurement.

| Tile size / depth | Output size | Full-canvas peak RSS (MiB) | Blocked peak RSS (MiB) | Peak reduction | Full-canvas seconds | Blocked seconds |
|---|---|---:|---:|---:|---:|---:|
| 1024×1024 / 8-bit | 2050×2050 | 243.7 | 135.2 | 44.5% | 0.138 | 0.111 |
| 1024×1024 / 16-bit | 2050×2050 | 260.8 | 149.3 | 42.8% | 0.152 | 0.133 |
| 2048×2048 / 8-bit | 4098×4098 | 775.1 | 191.5 | 75.3% | 0.583 | 0.418 |
| 2048×2048 / 16-bit | 4098×4098 | 843.2 | 243.6 | 71.1% | 0.556 | 0.484 |

Values are medians of three samples. All six output hashes in each case are identical between old and new backends. Source intensities are unchanged; this benchmark isolates blending and does not validate SEM alignment quality. Timings are small and machine-dependent.

Two reusable float64 accumulation buffers occupy at most **16 MiB**, with additional block-sized weights, weighted intensities and interpolation temporaries. The source halo extends at most one pixel past a block. Distance weights are measured against the full source tile edges, preventing new seams at block boundaries. Normalization, rounding and clipping happen in each block without full-canvas float temporaries.

All source tiles and the final uint8/uint16 panorama still remain in RAM. Memory therefore grows with source/output size; this is not an out-of-core pipeline. Feature extraction and TIFF export can still determine a full task's peak memory. RSS values are available only on macOS/Linux; Windows still runs pixel-equivalence and cancellation tests.
