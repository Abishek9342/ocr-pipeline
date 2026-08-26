# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/).

## [0.5.0] - Unreleased

Overnight statistical-hardening, real-data-readiness, and routing-v2-prep
pass. See `docs/OVERNIGHT_RESEARCH_REPORT.md` for the full narrative,
`docs/TOMORROW_HANDOFF.md` for the concise pickup point, and
`docs/routing_v2_readiness.md` for the per-condition routing audit.

### Added
- `benchmark/stats_utils.py` extended: `bootstrap_median_ci`,
  `paired_bootstrap_diff_ci`, `paired_effect_size` (Cohen's d),
  `percentile_interval`, `pairwise_comparison_summary`,
  `bonferroni_correction`; `latency_variance_stats` now also reports
  median/P95. All with validation, deterministic seeding, and no silent
  data-dropping.
- `benchmark/run_paired_comparison.py` — paired bootstrap CI (`ours` vs.
  each baseline, matched by `image_id`/`preset`). All four paired 95%
  CIs exclude zero (see `docs/statistical_rigor_report.md`).
- `benchmark/run_statistical_report.py` extended: repeated-run latency
  (cold-start vs. warm) now measured for every engine plus `ours`, not
  just Tesseract.
- `benchmark/analyze_robustness.py` — worst-severity CER, degradation
  slope, catastrophic-onset severity, and normalized AUC per
  (corruption type, system), computed from the existing robustness
  curves without replacing them.
- `benchmark/dataset_schema.py`, `benchmark/dataset_validator.py`,
  `benchmark/failure_taxonomy.py`, `benchmark/run_real_dataset.py` — a
  generic real-dataset schema, manifest validator (broken paths,
  duplicate/missing IDs, split overlap, invalid UTF-8, malformed
  metadata/bounding boxes), a rule-based (not learned) failure-type
  taxonomy, and a CLI scaffold that validates + evaluates against a
  real dataset the moment one exists. No dataset was downloaded or
  fabricated — outbound network access remains sandboxed.
- `run_benchmark.py`'s row schema extended with `image_features` (the
  full `QualityReport`, logged at collection time), `language`/`script`
  (honest per-corpus values, forward-compatible with a real
  multilingual dataset), `calibrated_confidence` (left `None` — no
  calibrator is wired in; see the 0.4.0 rejection), and `failure_type`.
- `docs/routing_v2_readiness.md` — per-condition audit of
  `select_primary_engine`'s coverage. Headline finding: every remaining
  real gap (`smudged`, `light_blur`, `motion_blur`) needs a NEW or
  recalibrated quality signal, not a quick rule using an existing field
  — checked directly against `assess()`'s real output, not assumed.
- `docs/reproduce_denoise_gate_finding.py` — reproduces the controlled
  comparison behind the `denoise()` fix below.

### Fixed
- **`ocr_resilience/preprocess.py`: `build_pipeline()`'s
  `is_noisy -> denoise()` gate was hurting accuracy on the exact
  condition it targeted.** A controlled same-image comparison showed
  PaddleOCR's CER on the `noisy` preset rise from 0.0087 to 0.0217 (2.5x
  worse) when this step ran; consistent with the independent ablation
  finding that forcing `denoise()` unconditionally hurts CER corpus-wide.
  Removed from the default chain. This is a genuinely MIXED result, not
  a clean win — `combo_hard`'s CER measurably worsened (0.0581 ->
  0.0681) because Tesseract's ensemble contribution there specifically
  benefits from denoising even though its `noisy`-alone performance
  doesn't. Kept anyway because the actually-shipped 4-engine pipeline's
  headline CER improved (0.0319 -> 0.0308) and the `noisy` win is large;
  documented precisely in `denoise()`'s own docstring so a future
  contributor doesn't have to re-discover this trade-off. Found during
  this pass's routing-v2 readiness audit, which noticed `ours`
  underperformed Tesseract ALONE on the `noisy` preset with no obvious
  explanation until traced here.
- **Stale generated benchmark artifacts**: `ablation_raw.csv`/
  `ablation_summary.csv` were timestamped before `engine_selection.py`
  existed — the exact class of bug this pass's Section 15 reproducibility
  audit was built to catch. Re-ran; then re-ran AGAIN after the
  `denoise()` fix above changed the numbers a second time.
- **A machine-sleep data-quality issue in this pass's own overnight full-
  benchmark run**: 2 of 1100 raw rows recorded latency_sec of 3-11 hours
  because the host machine slept mid-run (confirmed via matching
  timestamp gaps in the run log). Excluded from latency aggregates only
  (their CER/WER values are unaffected and remain included); documented
  in `benchmark.json`'s `data_quality_note` rather than silently dropped
  or silently left in.
- `benchmark/results/baseline_summary.json` (the CI regression baseline)
  refreshed to the corrected post-fix numbers via
  `scripts/check_regression.py --write-baseline`.

## [0.4.0] - Unreleased

See `docs/NEXT_PHASE_REPORT.md` for the full handoff on adaptive engine
selection + confidence calibration; `docs/engine_selection_report.md`,
`docs/confidence_calibration_report.md`, and `docs/routing_benchmark_report.md`
for the detailed evidence behind it.

### Added
- `benchmark/stats_utils.py` — bootstrap confidence intervals and
  repeated-run latency variance (mission section 21's statistical-rigor
  gap). Applied to the headline CER comparison: `ours`' 95% CI ([0.023,
  0.044]) doesn't overlap PaddleOCR's ([0.080, 0.148]) even at 220
  samples/system — the pipeline's overall advantage is statistically
  distinguishable from the best single baseline, not just a point-
  estimate difference. See `docs/statistical_rigor_report.md`.
- `ocr_resilience/engine_selection.py` — quality-aware single-engine
  selection (`select_primary_engine`), replacing `router.decide()`'s
  previous `available_engines[0]` (pure registration order) with
  interpretable rules keyed on `QualityReport`'s continuous fields,
  derived from this project's own condition-x-engine benchmark evidence.
  Not a trained/learned model — the 20-image corpus is far too small for
  that (see `docs/engineering_backlog.md`'s meta-dataset schema section).
- `rank_fallback_chain()` + a real two-tier confidence-based fallback:
  `OCRPipeline`'s fallback now tries one ranked additional engine before
  escalating to the full ensemble, instead of jumping straight from one
  engine to all of them.
- `ocr_resilience/calibration.py` — binned/histogram confidence
  calibration, reliability curves, Expected Calibration Error, and
  calibrated-confidence fusion, investigated as a candidate fix for the
  cross-engine confidence-scale mismatch documented in
  `docs/failure_analysis.md`. Result: rejected as a default (see below) —
  kept as a real, tested, documented capability, not wired into the
  pipeline by default.
- `benchmark/analyze_routing.py` — condition x engine strength table and
  routing regret analysis (OLD vs. NEW router), consuming
  `benchmark/run_benchmark.py`'s now-enhanced row schema (per-row
  confidence, detection count, routing reason, catastrophic-failure flag).
- `docs/reproduce_calibration_analysis.py` — standalone reproduction
  script for the calibration report's numbers.

### Fixed
- `engine="auto"` resolution (`OCR()`'s documented default usage) never
  included RapidOCR in its priority list — added as a fourth engine in a
  prior phase without updating this separate list, so `OCR()` would
  silently never use it even when installed. Found during this phase's
  Phase 1 router audit.
- **A real bug in this phase's own new analysis code**:
  `analyze_routing.routing_regret()`'s "top-1" computation compared
  engine *names* against whichever engine `pandas.DataFrame.idxmin()`
  returned for the best CER — which resolves ties (common at CER=0.0 on
  easy images) to whichever engine's rows appear first in
  `raw_results.csv` (always Tesseract, by construction). This made the
  new quality-aware router look far worse than it is (28% apparent top-1
  vs. the corrected 60%) until fixed to compare CER *values*, not names.
- **A third reproducibility gap, this time in the confidence-calibration
  research itself**: an inline (not saved-to-a-script) run of the
  raw-vs-calibrated fusion comparison produced numbers that did not
  reproduce when the identical logic was re-run via a saved script —
  attributed to EasyOCR's underlying PyTorch CPU inference not
  guaranteeing deterministic execution order across process invocations.
  The corrected, twice-confirmed-reproducible numbers are what's
  published in `docs/confidence_calibration_report.md`.
- **The `ours`-vs-baselines full benchmark numbers were from a run that
  started before this phase's router code existed** — a background
  process doesn't pick up file edits made after it starts. Re-ran with
  all current code actually in effect; the corrected numbers show a
  larger win than the isolated regret metric alone implied (mean CER
  0.038 -> 0.032, presets won outright 3 -> 5 of 11).
- **The calibration report initially overstated its own finding**:
  "measurably hurts on 3 of 6 presets" was the point-estimate direction,
  but a paired bootstrap 95% CI on each per-preset difference (added
  alongside `stats_utils.py`, above) shows every one includes zero — the
  accurate claim is "no statistically detectable effect," not "shown to
  hurt." Corrected in `docs/confidence_calibration_report.md`.

## [0.3.0] - Unreleased

See `MISSION_REPORT.md` for the full section-by-section status against
this project's research mission brief, including what's honestly not
done yet and why.

### Added
- `benchmark/run_robustness.py` — severity-sweep robustness curves (4
  corruption types x 5 severity levels each), not just one CER number per
  degradation type. Found a real, previously-invisible issue: Tesseract
  has a sharp failure cliff on Gaussian noise between sigma 25 and 50
  (complete failure, CER 1.0), which the main benchmark's fixed-severity
  `noisy` preset (sigma=25) sits just below — see `docs/robustness_curves.md`.
- `scripts/check_regression.py` + `benchmark/results/baseline_summary.json`
  — a CI regression gate comparing a fresh benchmark run's CER/latency
  against a stored baseline, with explicit thresholds. Wired into
  `.github/workflows/ci.yml` as a manual (`workflow_dispatch`-only)
  `benchmark-regression` job — the full benchmark is too slow for every
  push, so this is opt-in rather than blocking, per a tiered CI strategy.
- `ocr_resilience/debug.py` + `OCRPipeline.run(debug_dir=...)` — visual
  debugging export (original/preprocessed/annotated-with-boxes images).
  Off by default, zero cost unless requested.
- `OCRPipeline.run(min_confidence_for_fallback=...)` — a genuine
  confidence-based second-pass escalation to every available engine when
  the first pass's confidence is low, with a real regression test suite
  (`tests/test_confidence_fallback.py`). Off by default (`None`), zero
  overhead unless explicitly enabled.
- `fusion.fuse(weighted=...)` / `OCRPipeline.run(fusion_weighted=...)` —
  unweighted majority voting as an alternative to confidence-weighted
  ROVER voting, added after investigating (and confirming) that
  cross-engine confidence scales aren't reliably comparable. Tested
  directly against the default; kept as an option, not adopted as the
  default (net negative across presets — see `docs/failure_analysis.md`).
- `ocr_resilience/scoring.py` — an optional composite accuracy/latency/
  memory score with named, published weights and a `rank_stability()`
  check for whether a ranking claim survives reasonable reweighting.
  Never a replacement for the underlying per-metric numbers.
- `RapidOCRAdapter` — a fourth engine backend (ONNX-runtime port of the
  same PP-OCR model family PaddleOCR uses), added specifically because it
  has zero PyTorch/PaddlePaddle/TensorFlow dependency and a ~30MB install
  footprint, making it the most CI-friendly neural engine option. Not a
  replacement for PaddleOCR — added alongside it after researching the
  current open-source OCR ecosystem (Surya, docTR/OnnxTR, MMOCR, Kraken,
  TrOCR also evaluated; see `docs/engine_landscape.md` for why they
  weren't chosen — license restrictions, CPU-latency concerns, or
  narrower scope).
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
- **A second, previously-undiscovered reproducibility hole**: `gaussian_noise()`
  and `salt_and_pepper()` in `benchmark/degrade.py` called
  `np.random.normal`/`np.random.randint` directly, reading NumPy's own
  *global* random state and ignoring the `rng` parameter every other
  degradation function respects. So the `noisy`, `salt_pepper`, and
  `combo_hard` presets were silently NOT seed-reproducible, despite
  `stable_seed()`'s entire purpose being reproducibility, and despite an
  earlier fix (above) for a *different* seeding bug. Found while building
  `benchmark/run_robustness.py`. Fixed by deriving a seeded
  `numpy.random.Generator` from the same `rng` object. The existing
  reproducibility test only checked the `skewed` preset (which never
  touched numpy's global state) — now parametrized over every preset.
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
- **PaddleOCR is no longer permanently skipped from the benchmark.** The
  default model version (PP-OCRv5/v6 detector) hits a reproducible upstream
  PaddlePaddle PIR (Paddle Intermediate Representation) attribute-type
  mismatch on model load in this environment. `PaddleOCRAdapter` now
  defaults `ocr_version="PP-OCRv4"` (configurable), whose mobile det/rec
  models don't hit this bug — verified end-to-end on clean and degraded
  images across all 11 presets.
- **PaddleOCR crashed the first time all three engines actually ran
  together in one ensemble** (`ValueError: not enough values to unpack
  (expected 3, got 2)`, inside PaddleX's internal resize step, which does
  `h, w, _ = img.shape`). The pipeline's shared preprocessing always
  outputs grayscale (2D) images; Tesseract and EasyOCR both accept that
  directly, so this was invisible until PaddleOCR was actually wired into
  a real multi-engine run. `PaddleOCRAdapter.recognize()` now converts
  grayscale input back to 3-channel BGR before calling PaddleOCR.

## [0.1.0] - Initial release

Adaptive preprocessing (quality-gated deskew/denoise/deblur/contrast),
Tesseract/EasyOCR/PaddleOCR adapters behind a common `OCREngine` protocol,
quality-aware single-engine-vs-ensemble routing, ROVER-style multi-engine
fusion with union-find spatial grouping, CER/WER metrics, and an initial
synthetic benchmark (8 of 11 degradation presets, printed + cursive-font
"handwritten proxy" text).
