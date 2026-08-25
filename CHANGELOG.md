# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/).

## [0.2.0] - Unreleased

### Added
- `ocr-pipeline` CLI (`ocr_resilience/cli.py`) — single-file, multi-file,
  and directory batch processing, JSON output.
- `OCR` convenience class (`ocr_resilience.OCR`) wrapping `OCRPipeline` to
  match the documented `OCR(engine=..., preprocessing=...).predict(...)`
  usage.
- `OCRPipeline.run_batch()` for processing multiple images against one
  loaded pipeline instance.
- `ocr_resilience/postprocessing.py` — whitespace/unicode normalization,
  confidence filtering, duplicate-detection dedup. `OCRResult` now exposes
  `raw_text` (pre-postprocessing) and `processed_text` (post) separately,
  plus `confidence`, `bounding_boxes`, `engine_used`, `processing_time`,
  and a `to_dict()` serializer.
- Ablation hooks on `OCRPipeline.run()`: `skip_preprocessing`,
  `force_ensemble`, `force_step` — used by the new
  `benchmark/run_ablation.py`, which measures the incremental contribution
  of each pipeline component (deskew, denoise, CLAHE, Sauvola, adaptive
  preprocessing, multi-engine selection) against a no-preprocessing
  baseline. See the README's Ablation Study section for real numbers.
- `benchmark/run_benchmark.py` CLI flags (`--dataset`, `--engines`,
  `--presets`, `--out`), all 11 degradation presets wired in (previously
  8 of 11 ran by default), `summary.csv`/`benchmark.json`/`latency.csv`
  outputs alongside `raw_results.csv`, P95 latency and peak-memory
  (`tracemalloc`) metrics, and graceful mid-run degradation if an engine
  fails at runtime (drops it and continues, rather than crashing).
- `.github/workflows/ci.yml` (lint + test matrix across Python 3.10-3.13,
  build + clean-install validation) and `.github/workflows/publish.yml`
  (PyPI trusted publishing on a version tag).

### Fixed
- **`OCRResult.text` joined every detection with `"\n"` regardless of
  which text line it belonged to.** Word-level detections on the same
  visual line (e.g. Tesseract's default one-box-per-word output) rendered
  as one word per output line instead of one space-joined line — e.g.
  "Hello World 12345" became three separate lines. Invisible until now
  because `benchmark/run_benchmark.py` always bypassed `.text` and
  space-joined `result.detections` directly itself, so it never showed up
  in a CER/WER number; caught only by exercising the CLI end-to-end
  against a real engine. Fixed by reconstructing text from the same
  line-clustering logic `_reading_order` already used for ordering
  (`_group_into_lines`), joining words within a line by space and lines by
  newline.
- **Benchmark seed derivation was not actually reproducible.**
  `hash((path, preset))` used Python's built-in `hash()`, which salts
  string hashing per-process by default (`PYTHONHASHSEED` randomization)
  — the same (image, preset) pair produced a *different* degraded image
  on every run/machine despite `seed=` being passed through, undermining
  the benchmark's own reproducibility claim. Replaced with a `sha256`-based
  `stable_seed()`.
- **Salt-and-pepper noise made Tesseract return completely empty output
  (CER 1.0), and the quality assessor never noticed** — found only after
  widening the benchmark from 8 to all 11 presets (the previous benchmark
  never ran `salt_pepper`). `noise_score` is a median-absolute-deviation
  estimator, a *robust* statistic deliberately insensitive to sparse
  outlier pixels — exactly what salt-and-pepper noise is — so `is_noisy`
  never fired and no denoising ran. Added `QualityReport.impulse_noise_score`
  /`is_impulse_noisy` (fraction of pixels sharply off their local 3x3
  median) as a second, complementary noise signal, and gated a new
  `preprocess.median_denoise()` on it. CER on that preset: 1.000 -> 0.019.
  `router.decide()`'s degradation-flag count now includes this signal too.

## [0.1.0] - Initial release

Adaptive preprocessing (quality-gated deskew/denoise/deblur/contrast),
Tesseract/EasyOCR/PaddleOCR adapters behind a common `OCREngine` protocol,
quality-aware single-engine-vs-ensemble routing, ROVER-style multi-engine
fusion with union-find spatial grouping, CER/WER metrics, and an initial
synthetic benchmark (8 of 11 degradation presets, printed + cursive-font
"handwritten proxy" text).
