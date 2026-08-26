# Next Phase Report: Adaptive Engine Selection + Confidence Calibration

Final handoff for `ocr_next_research_mission.md`. Cross-references:
`docs/engine_selection_report.md`, `docs/confidence_calibration_report.md`,
`docs/routing_benchmark_report.md`.

**Update**: a third, overnight mission phase followed this one — see
`docs/OVERNIGHT_RESEARCH_REPORT.md` for the current status. The mean-CER
figures below (0.038 -> 0.032) are that phase's own accurate snapshot,
superseded by the current **0.0308** (`docs/statistical_rigor_report.md`)
after a bug fix (`ocr_resilience/preprocess.py`'s `denoise()` gate) found
during the later phase's routing-readiness audit. Left as originally
written below, not retroactively edited.

### What changed

- Fixed a real bug found during Phase 1's audit: `engine="auto"`
  resolution never included RapidOCR, so the documented default `OCR()`
  usage silently never used a registered engine.
- Replaced `router.decide()`'s single-engine choice
  (`available_engines[0]`, pure registration order) with
  `ocr_resilience/engine_selection.py::select_primary_engine()` — an
  interpretable, rule-based selector keyed on `QualityReport`'s existing
  continuous fields, derived from this benchmark's own measured
  condition-x-engine evidence.
- Extended confidence-based fallback from a single straight jump
  (primary -> full ensemble) to a ranked two-tier escalation (primary ->
  one best-ranked fallback engine -> only then the full ensemble).
- Extended `benchmark/run_benchmark.py`'s row schema to log per-row
  confidence, detection count, routing reason, and a catastrophic-failure
  flag — none of this was captured before, and Phases 2/6/9/13 all needed
  it.
- Built `benchmark/analyze_routing.py` (condition x engine table, routing
  regret) and `ocr_resilience/calibration.py` (binned calibration,
  reliability curves, ECE, calibrated fusion).
- Found and fixed a real bug in the analysis code itself:
  `routing_regret()`'s "top-1" computation compared engine *names*
  against whichever engine `pandas.idxmin()` returned for ties — which
  silently favors whichever engine's rows appear first in
  `raw_results.csv` (always Tesseract). This made the new router look
  far worse than it is (28% apparent top-1) until fixed to compare CER
  *values* (60% actual top-1).
- Re-ran the full benchmark after ALL of this phase's code was actually
  in place (an earlier run had started before `select_primary_engine`
  existed and silently used the old router the whole time — Python
  doesn't hot-reload an already-running process). The corrected run
  shows a larger, more direct win than the isolated regret metric alone
  implied — see "Current benchmark position."
- Added `benchmark/stats_utils.py` (bootstrap confidence intervals,
  repeated-run latency variance) and applied it to both the headline
  CER comparison and the calibration comparison — see
  `docs/statistical_rigor_report.md`. This corrected an earlier,
  overstated claim in this same phase's calibration report (see below).
- Confirmed, directly (not just asserted), that multilingual testing is
  blocked at the model level, not just the dataset level: `tesseract
  --list-langs` shows only `eng`; EasyOCR/PaddleOCR's local model caches
  only have English recognition models; no outbound network access
  exists to fetch others. See `docs/engineering_backlog.md`.

### What was measured

- Condition x engine strength table (11 presets x 4 baseline engines).
- Routing regret: OLD vs. NEW router, 220 image/preset cases.
- Per-engine Expected Calibration Error (4 engines, 220 rows each).
- Raw-vs-calibrated and weighted-vs-unweighted fusion, 6 presets x 20
  images each, both confirmed reproducible via saved scripts (not just a
  single live run — one calibration comparison was corrected after an
  initial inline run didn't reproduce on script re-run; see
  `docs/confidence_calibration_report.md`'s honesty note).

### What improved

- **Quality-aware engine selection**: top-1 accuracy 56.8% -> 59.6%, mean
  regret 0.1696 -> 0.1199, mean selected CER 0.1790 -> 0.1293 — all three
  simultaneously, on the exact 220-case comparison in
  `docs/routing_benchmark_report.md`. Confirmed again directly in the full
  pipeline benchmark (not just the isolated regret metric): `ours` overall
  mean CER improved 0.038 -> 0.032, and it now wins outright on 5 of 11
  presets instead of 3 — see "Current benchmark position" below.
- **Ranked fallback**: with 3+ engines, tier 1 (primary + 1 fallback) now
  resolves confidence-triggered escalations that don't need the full
  ensemble, at roughly half the added latency of jumping straight to all
  engines — demonstrated in `tests/test_confidence_fallback.py`'s
  3-engine test cases (real code behavior, not yet measured on the full
  benchmark's actual latency distribution).
- **RapidOCR is now included in `engine="auto"`** — a real (if small)
  correctness fix to the documented public API.

### What got worse

- Nothing measured got worse from the engine-selection or fallback
  changes — both show improvement on every metric compared, not a
  tradeoff.
- The calibration investigation is a net time cost with no shipped
  improvement (see below) — a legitimate research cost, not a code
  regression.

### What was rejected

1. **Unweighted fusion as the default** (mission Phase 11, strategy 2):
   helps `combo_hard` specifically but hurts `heavy_blur`/`motion_blur`
   more — net unfavorable. Kept as a tested, documented option
   (`fusion.fuse(weighted=False)`), not adopted as default.
2. **Calibrated-confidence fusion** (strategies 3, 5, 6): binned
   calibration correctly identifies real, reproducible per-engine
   miscalibration (EasyOCR ECE 0.20 vs. PaddleOCR's 0.015), but applying
   it to fusion weighting did not improve — and measurably hurt — fused
   CER in a controlled 6-preset comparison, confirmed reproducible across
   two script re-runs.
3. **A learned/ML router** (mission Phase 16, explicitly forbidden this
   phase): not attempted, correctly — the 20-image corpus is far too
   small.

### Why it was rejected

For unweighted fusion and calibrated fusion: both were tested against
the mission's own standard ("do not claim improvement without a
controlled benchmark proving it") and the controlled benchmarks didn't
support adoption. For the learned router: the mission's own hard-stop
rule, and a real data-scale argument independent of the rule (see
`docs/engineering_backlog.md`'s meta-dataset schema section).

### Current best architecture

Quality-gated adaptive preprocessing -> quality-aware single-engine
selection (this phase) OR quality-flag-count-triggered full ensemble ->
confidence-weighted ROVER fusion (unweighted and calibrated variants
available but not default) -> two-tier ranked confidence-based fallback
(this phase) -> line-aware text reconstruction. Four engines (Tesseract,
EasyOCR, PaddleOCR, RapidOCR) behind a common protocol, pluggable without
core changes.

### Current benchmark position

Mean CER: `ours` **0.032**, PaddleOCR (best single baseline) 0.111,
EasyOCR 0.121, RapidOCR 0.172, Tesseract 0.179 — improved from an
earlier 0.038 once the benchmark was re-run with this phase's router
code actually in effect (see "What was measured"'s reproducibility note).
Catastrophic-failure rate: `ours` 0.5%, vs. 5-14% for the single engines.
Per-condition wins also improved directly: the pipeline now wins outright
on 5 of 11 presets (up from 3), including `skewed` and `low_contrast`
where it now beats even PaddleOCR alone — a direct, measured result of
this phase's `select_primary_engine` rules, not just the consistency
argument. PaddleOCR alone is still the best single engine on the other
5 (clean, light blur, general noise, smudged, and now also `combo_hard`).

### Current limitations

- Engine-selection rules cover 4 of 11 conditions explicitly; the other 7
  fall through to registration order (which happens to coincide with
  good choices for several of them, per the condition table, but isn't
  asserted as a rule).
- `rank_fallback_chain` uses one aggregate ranking, not a condition-aware
  one.
- Calibration's null result is itself only weakly powered (20 images) —
  "rejected at this sample size," not "proven not to work at any scale."
  Now precisely quantified, not just asserted: a paired bootstrap 95% CI
  on every raw-vs-calibrated CER difference includes zero (see
  `docs/statistical_rigor_report.md` and the corrected
  `docs/confidence_calibration_report.md`) — the honest claim is "no
  detectable effect," not "shown to hurt," which an earlier draft of that
  report overstated before this correction.
- No cost-aware routing formula (Phase 12) — the two decisions this phase
  made were cost-aware informally, not via a general optimizer.
- Phase 15's regression protection for the FULL benchmark (not just unit
  tests) depends on `scripts/check_regression.py` actually being run
  before future routing changes ship — it is not automatically enforced
  on every PR (see that script's own CI wiring: manual `workflow_dispatch`
  only, because the full benchmark is too slow for every push).
- All limitations already listed in `docs/engineering_backlog.md` still
  apply (no real dataset, no multilingual testing, no statistical
  confidence intervals).

### Exact next research priority

**Condition-aware fallback ranking** and **filling in engine-selection
rules for the remaining 7 conditions** are the most direct, cheapest
extensions of this phase's own work — both use data already collected in
`benchmark/results/condition_engine_table.csv`, no new experiments
required, just more rules written against evidence that already exists.

### Data required for learned routing

See `docs/engineering_backlog.md`'s exact schema. Restated briefly: a
real (non-synthetic) dataset of at least several hundred images across
genuinely diverse document types, with the same per-row schema this
phase's benchmark harness now logs (image features, engine, confidence,
CER/WER/latency, routing decision, failure type), split into held-out
train/validation/test partitions that the model's own fitting never
touches.

### Conditions required before moving to learned routing

1. The real dataset above exists and is large enough that a held-out
   test set still has meaningful sample size per condition.
2. `select_primary_engine`'s interpretable rules are established as the
   baseline to beat — a learned model that doesn't outperform it isn't
   worth the loss of interpretability.
3. `image_features` and `failure_type` are logged per-row at collection
   time, not re-derived after the fact (as `analyze_routing.py` currently
   does by re-running `assess()` against the same seed — workable at this
   scale, not something to depend on at a much larger one).
4. A discipline for not tuning against the held-out test set is actually
   in place (mission Phase 22's "challenge set," deferred in
   `docs/engineering_backlog.md` until there's a second contributor for
   it to matter for).
