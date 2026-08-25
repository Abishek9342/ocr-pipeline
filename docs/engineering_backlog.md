# Engineering Backlog: What's Not Done, and Why

This project's mission doc explicitly forbids claiming completion that
isn't real (see `MISSION_REPORT.md`, section 28 of the original brief:
"do not create fake superiority"). This file is the other half of that
honesty: an explicit list of what a genuinely complete version of this
project would still need, why it isn't here, and what it would take.

## Requires a real dataset (not buildable from this environment alone)

- **Real scanned documents / receipts / forms / photographed documents /
  screenshots.** Everything benchmarked here is synthetic, rendered text.
  Needs: sourcing a licensed or public-domain document dataset (e.g.
  FUNSD, CORD, SROIE for receipts/forms — all exist and are usable, just
  not yet integrated here) and re-running the full benchmark against it.
- **A real handwriting corpus** (e.g. IAM) — requires registration/license
  acceptance not completed in this environment. The "handwritten_proxy"
  cursive-font style is an explicitly-flagged stand-in, not equivalent.
- **Multilingual test data** — every engine here supports far more than
  English (PaddleOCR: 80+ languages; EasyOCR: 80+; RapidOCR: Chinese-
  optimized with broader model options). None of that has been exercised.
  **Confirmed blocked at the model level, not just the dataset level**:
  checked directly — `tesseract --list-langs` shows only `eng` installed
  (no Hindi/Tamil/etc. `.traineddata`); `~/.EasyOCR/model/` only has
  `english_g2.pth` cached; `~/.paddlex/official_models/` only has
  `en_PP-OCRv4_mobile_rec` (the detection models are language-agnostic,
  but recognition needs a matching language model). Loading any
  non-English model would require a fresh download, and this
  environment has no outbound network access from code execution
  (confirmed separately — DNS resolution fails). Needs: network access
  to fetch language models (or pre-bundled models), plus per-language
  ground-truth text and, ideally, real (not font-rendered) samples per
  script.
- **A held-out "challenge set"** never used for tuning (mission section
  22). With one contributor and no history of parameter-tuning-against-
  the-test-set pressure yet, creating one now would be process theater —
  worth doing once there's a second contributor or a real risk of
  overfitting to the public benchmark.

## Requires compute/scale this environment doesn't have

- **A learned/ML pipeline router** (mission sections 8, 11 tier 4-5,
  section 45's "can a lightweight router predict the best path?"). This
  needs a meta-dataset large enough to train something non-trivial — the
  current corpus (20 images x 11 presets) is two-to-three orders of
  magnitude too small; a model trained on it would memorize, not
  generalize, and reporting that as "learned routing" would itself be the
  kind of fake result this project's own ground rule forbids. Deliberately
  NOT attempted — see the exact schema and prerequisites below instead.

  **Meta-dataset schema** (every field `benchmark/run_benchmark.py`'s
  enhanced row schema already logs, marked ✅; not yet logged, marked ⬜):

  ```text
  image_id                  ✅ (raw_results.csv)
  image_features            ⬜ (QualityReport's fields aren't logged per-row
                                 yet — assess() is re-run in analyze_routing.py
                                 from the same deterministic seed instead)
  engine                    ✅ (system column)
  pipeline / preprocessing  ✅ (available via OCRResult.preprocessing_steps
                                 for "ours" rows; not logged as a raw_results.csv
                                 column yet)
  raw_confidence            ✅ (mean_confidence)
  calibrated_confidence     ⬜ (this phase's confidence-calibration work)
  CER / WER                 ✅
  latency                   ✅
  memory                    ✅ (peak_memory_bytes)
  success / failure         ✅ (catastrophic_failure, plus cer itself)
  failure_type              ⬜ (no automatic categorization — missing text
                                 vs. wrong text vs. complete blank isn't
                                 classified per-row, only investigated by
                                 hand per case in docs/failure_analysis.md)
  routing_decision           ✅ (routing_reason, for "ours" rows)
  ```

  **What would justify moving to learned routing**: (1) a real, non-
  synthetic dataset at least ~500-1000+ images across diverse real
  document types (mission section 14's gap — the actual blocker), (2) the
  `image_features`/`failure_type` columns above actually populated per
  row, not re-derived after the fact, (3) a held-out test set that the
  model's own training never touches (mission section 21/22's gap), and
  (4) a simple baseline (e.g. this phase's rule-based `select_primary_engine`)
  to compare the learned model against — a learned router that doesn't
  beat the interpretable rules isn't worth the loss of interpretability.
- **Newer VLM-based OCR baselines** (PaddleOCR-VL, GOT-OCR2.0, dots.ocr,
  olmOCR, DeepSeek-OCR — see `docs/engine_landscape.md`) are the current
  document-parsing SOTA but are GPU-first in their reference deployments;
  this project's constraint (CPU-only CI) rules them out until that
  constraint changes.

## Buildable, but deliberately deferred this round (scope, not blockers)

- ~~**Statistical rigor**~~ — **done**, see `docs/statistical_rigor_report.md`
  (bootstrap CIs on mean CER, repeated-run latency variance). Correcting
  a speculation this file itself made earlier: this gap's original entry
  guessed "a confidence interval would be wide enough to not change any
  conclusion" — measured, that guess was wrong. `ours`' 95% CI ([0.023,
  0.044]) doesn't overlap PaddleOCR's ([0.080, 0.148]) even at only 220
  samples per system; the pipeline's advantage over the best baseline is
  statistically distinguishable, not just a point-estimate difference.
  Not yet done: CIs on WER (mechanical extension of the same method), and
  repeated-run latency variance for EasyOCR/PaddleOCR/RapidOCR (only
  Tesseract measured so far, to keep the report's own runtime small).
- **Cascading compute tiers as a fully general N-tier system** (mission
  section 11). What exists: quality assessment (tier 0) -> adaptive
  single/multi-engine routing (tiers 1-3) -> confidence-based fallback to
  full ensemble (`OCRPipeline.run(min_confidence_for_fallback=...)`,
  tier 4). A fully general, configurable N-tier framework with per-tier
  cost budgets was not built — the current 4-stage version already
  exercises the "stop as soon as quality is sufficient" idea the mission
  describes, and building further tiers without a case that needs them
  would be exactly the "add complexity nobody asked for" anti-pattern the
  mission itself warns against.
- ~~**Quality-aware single-engine selection**~~ — **done**, see
  `ocr_resilience/engine_selection.py` and `docs/engine_selection_report.md`.
  Covers 4 of 11 benchmarked conditions with explicit rules; the
  remaining 7 still fall through to registration order (which happens to
  coincide with reasonable choices for several of them). Extending
  coverage to the rest is now the cheapest remaining improvement — the
  condition-x-engine evidence for it already exists in
  `benchmark/results/condition_engine_table.csv`.
- **Condition-aware ranked fallback**: `rank_fallback_chain()` (added
  alongside the selection work above) uses one aggregate engine ranking
  for ALL conditions, not a per-condition one. A real, cheap next step —
  same data source as the item above.
- **Research notebooks / additional Kaggle notebooks** (mission sections
  35, 37: 8 research notebooks, 4 Kaggle notebooks). One executable
  notebook exists (`notebooks/ocr_resilience_benchmark.ipynb`, covering
  baseline comparison, preprocessing/ablation experiments, benchmark
  results, and failure analysis in one document) rather than eight
  separate ones — consolidated deliberately since the underlying
  experiments are the same corpus/harness; splitting them into eight
  files would multiply maintenance surface without adding new evidence.
- **Kaggle/PyPI publishing itself**: the notebook and package are
  release-ready (build validated in a clean venv, CLI/API tested) but
  actually uploading to Kaggle or publishing to PyPI requires accounts and
  credentials this session doesn't have — see the repo's git history /
  session record for what was explicitly deferred to the human owner.

## Already done, listed here only so this isn't mistaken for a duplicate gap

Confidence as a first-class concept (`OCRResult.confidence`), a
composite-score module with rank-stability checking (`ocr_resilience/scoring.py`),
confidence-based fallback (`OCRPipeline.run(min_confidence_for_fallback=...)`),
the full failure-case investigation for Priority 6 (`docs/failure_analysis.md`),
PaddleOCR actually working (not skipped), quality-aware single-engine
selection, ranked (not all-at-once) confidence-based fallback escalation,
and binned confidence calibration (investigated and honestly rejected as
a fusion-weighting default — see `docs/confidence_calibration_report.md`)
are all done — see `MISSION_REPORT.md` and `docs/NEXT_PHASE_REPORT.md`
for the full per-section status.
