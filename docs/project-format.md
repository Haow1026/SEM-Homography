# SEM project format, version 1

`.semproj` files are UTF-8 JSON documents with `format: "sem-stitch-project"` and integer `version: 1`.
They store metadata and references to original images. They do not embed or export image pixels and do not use pickle.

| Field | Content |
|---|---|
| `sources` | Acquisition-order list of `path`, file byte `size` and lowercase SHA256 `sha256`. Paths are relative to the project directory where possible, including `..`; different Windows drives require absolute paths. |
| `scan` | `grid_shape` (`null` or `[rows, columns]`), boolean `serpentine` and `first_row_rtl`. A grid must match the source count; automatic mode has both scan-direction flags false. |
| `alignment` | `null` for a prepared input-only project; otherwise the completed pure-translation alignment below. |

An alignment contains:

- `blend`: `distance-feather-opencv-linear-v1`. Recompose with original-edge distance weights, premultiplied intensity/weight resampling and OpenCV linear interpolation, preserving uint8/uint16.
- `dtype`: `uint8` or `uint16`.
- `shapes`: source `[height, width]` values in acquisition order.
- `positions`: normalized floating-point `[x, y]` canvas positions in acquisition order, matching the compositor's canvas origin.
- `pairs`: matching diagnostics with zero-based acquisition indices `i`, `j`; measured/predicted displacement vectors, `inliers`, nonnegative pixel `residual` and boolean `excluded`. Indices need not be ascending after mapping snake/RTL scan order. Vectors retain their original direction. Diagnostics must agree with the saved positions.
- `output`: panorama `[height, width]` and SHA256 of the contiguous uncompressed pixel buffer, in its saved dtype.
- `versions`: Python, OpenCV and NumPy versions used when saving, for provenance. Matching is skipped on completed-project restore. The pixel hash is checked even when package versions differ.

Readers reject unsupported format/version/blend, invalid types, nonfinite/out-of-range coordinates, inconsistent geometry/diagnostics and metadata larger than 16 MiB before source decoding or canvas allocation. Unknown metadata fields do not execute code. Missing/modified source files and output pixel mismatches fail restoration.

Saving a completed project checks source size/mtime against the successful stitch report before streaming source hashes. Save/open check for source changes during their operation. Restore uses source file size + content hash, rather than insisting on the original timestamp, so moving/copying an unchanged bundle is supported. Newly restored seam-inspection reports use the sources' current signatures.

Saving writes and fsyncs a temporary file in the destination directory, checks cancellation and source stability, then uses `os.replace`. Failure/cancellation before replacement cleans up the temporary and preserves an existing project. Once replacement commits, a queued save success is reported even if cancellation arrives just afterwards. A project may not overwrite a source image.

Project IO runs on a background worker with cancellation during streaming hashes and between blend blocks. Tk polls a queue and applies a restored session only after verification and worker exit. Failed/cancelled opens preserve the current session; a cancelled open discards queued success. No source arrays are retained by project metadata after restoration.

The original images and final panorama still reside in RAM during composition. An existing session is retained until a new project succeeds, so its panorama may coexist temporarily with the new one. This format does not supply missing-source relinking, embedded images, interrupted-computation checkpoints, or automatic overwrite on close; use explicit Save project and keep original files alongside the project.
