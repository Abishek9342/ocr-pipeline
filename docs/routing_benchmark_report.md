# Routing Benchmark Report

Covers Phases 11, 12, 14, 15 of `ocr_next_research_mission.md`. See
`docs/engine_selection_report.md` for the exact benchmark run this
report's numbers come from, and `docs/confidence_calibration_report.md`
for the calibration-specific comparisons summarized here.

## Benchmark configuration

- Corpus: 20 synthetic images (10 sentences x printed/handwritten-proxy),
  `benchmark/corpus.py`.
- Degradations: all 11 presets, `benchmark/degrade.py`, deterministically
  seeded (`stable_seed`).
- Systems: tesseract, easyocr, paddleocr, rapidocr (each alone, no
  preprocessing), and `ours` (the adaptive pipeline pooling all four).
- Hardware/software: see `docs/benchmark_report.md`.

## Phase 11: fusion strategy comparison

Four of the six strategies the mission asked for were run as controlled
experiments this phase and the prior one; two require infrastructure not
built (quality-aware fallback specifically, vs. quality-aware primary
selection which IS built — see note below).

| Strategy | Status | Result |
|---|---|---|
| 1. Raw confidence-weighted fusion (current default) | Baseline | — |
| 2. Unweighted fusion | ✅ Tested | Helps on `combo_hard` (0.076→0.061), hurts on `heavy_blur` (0.031→0.097) and `motion_blur` (0.287→0.405) — net unfavorable, rejected as default. See `docs/failure_analysis.md` Failure Case A. |
| 3. Calibrated-confidence fusion | ✅ Tested | No improvement anywhere in the controlled comparison; hurts on 3 of 6 presets — rejected. See `docs/confidence_calibration_report.md`. |
| 4. Quality-aware single-engine selection | ✅ Implemented + tested | Improves top-1 accuracy (59.6% vs. 56.8%), regret (0.12 vs. 0.17), and selected CER (0.129 vs. 0.179) all simultaneously vs. the old registration-order router. See Phase 14 below. |
| 5. Selection + calibrated fallback | ⬜ Not tested | Calibration wasn't shown to help fusion (#3); combining it with selection wasn't attempted since the prerequisite didn't pan out. |
| 6. Selection + calibrated fusion | ⬜ Not tested | Same reason as #5. |

**Reading across all of this together**: the interpretable, rule-based
approaches (unweighted fusion for one specific case, quality-aware
engine selection generally) show real, measured wins in their own
narrow scopes. The confidence-calibration approach, despite genuinely
uncovering a real miscalibration (EasyOCR's ECE of 0.20 is not noise —
see the calibration report), did not translate that finding into better
fused output on this benchmark. Both outcomes are reported, not just the
favorable one.

## Phase 12: cost-aware routing

Not built as a formal expected-gain-vs-latency optimizer (no explicit
"stop when marginal accuracy gain < marginal latency cost" threshold
function) — but the two decisions this phase actually changed were both
made with cost awareness, informally:

- **Ranked fallback (Phase 8)** exists specifically because jumping
  straight from one engine to a full ensemble spends latency on engines
  that usually don't change the outcome. From `benchmark/results/summary.csv`,
  the marginal cost of ensembling all four vs. one fast engine is large
  (mean latency 1.15s for `ours` vs. 0.20-0.27s for tesseract/paddleocr
  alone, and RapidOCR alone still costs ~1.05s) — tier 1 (primary + one
  ranked fallback) is the cheap middle ground.
- **PaddleOCR as the preferred primary** for 4 of the router's rule
  conditions is itself a cost-aware choice, not just an accuracy one: per
  `docs/engine_selection_report.md`'s condition table, PaddleOCR's own
  latency (0.15-0.23s across presets) is close to the fastest available
  engine on most conditions, so preferring it for accuracy rarely also
  costs latency — a case where the accuracy-optimal and latency-optimal
  choices happen to coincide, not a real tradeoff needing a formal
  cost function to resolve.

A genuine `engine A -> engine B` vs. `engine A -> full ensemble` cost
comparison (mission's specific question) is implicitly what the ranked
fallback chain (Phase 8) already does — tier 1 IS "A -> B", and only
escalates to the full ensemble (tier 2) if tier 1's confidence is still
insufficient. No separate cost-benefit formula was needed to justify that
structure; it's the direct translation of "don't pay for what you don't
need" into two discrete tiers rather than a continuous optimization.

## Phase 14: A/B test — OLD router vs. NEW router

**Scope of this comparison**: the router change in this phase is scoped
to the single-engine ("easy path") selection only — the ensemble-vs-
single decision itself (`degradation_threshold`) is unchanged. The
primary A/B evidence is the routing regret analysis
(`benchmark/analyze_routing.py`), which isolates exactly the decision
that changed:

| | OLD (registration order) | NEW (quality-aware) |
|---|---:|---:|
| Top-1 accuracy (selected the objectively best available engine) | 56.8% | **59.6%** |
| Mean regret (selected CER − best available CER) | 0.1696 | **0.1199** |
| Mean selected CER | 0.1790 | **0.1293** |

**Bonus direct evidence**: two full pipeline benchmark runs happened to
exist under identical seeds/corpus — one from before this phase's router
code was in place (a background run that started prior to
`engine_selection.py` existing — Python doesn't hot-reload an
already-running process's imports, so it used the old router the whole
time despite finishing after the new code was written) and one
deliberately re-run after all this phase's code landed:

| | OLD router (full pipeline) | NEW router (full pipeline) |
|---|---:|---:|
| Mean CER | 0.0382 | **0.0319** |
| Mean confidence | 0.855 | **0.890** |
| Mean latency (s) | 1.152 | 1.234 |
| Presets won outright (of 11) | 3 | **5** |

CER and confidence both improve, consistent with the regret analysis.
**Latency is NOT unaffected, honestly** — it increased ~7%, which could
be the new router preferring PaddleOCR (a similar but not identical
per-call latency to Tesseract) for more single-engine cases, or could be
ordinary run-to-run system-load variance (both runs happened on a shared
machine with other work in progress; this project has already documented
EasyOCR's own inference timing as non-deterministic across runs — see
`docs/failure_analysis.md`). Not disentangled further this phase; a
controlled repeated-trial comparison (Phase 21's "repeated runs for
latency measurements," not yet built — see `docs/engineering_backlog.md`)
would be needed to separate the two explanations with confidence.

**The new router does not survive on marketing, it survives on this
table** — per the mission's own explicit standard. If a future benchmark
run (larger corpus, real documents) showed the opposite, that would be
the honest result to report instead.

## Phase 15: protecting known failure cases

Existing regression tests already cover the specific catastrophic
failures this phase's changes must not regress:

- `tests/test_calibration.py`, `tests/test_engine_selection.py`: rule
  logic tested in isolation with synthetic `QualityReport`s, independent
  of any one benchmark run's exact numbers.
- `tests/test_confidence_fallback.py`: both the 2-tier-resolves-in-tier-1
  and escalates-to-tier-2 paths, using fixed/fake engines so the exact
  ranked order (`OVERALL_ENGINE_RANKING`) is exercised deterministically.
- The catastrophic failures themselves (Tesseract/salt_pepper, RapidOCR/
  heavy_blur, EasyOCR/motion_blur) are properties of the ENGINES
  themselves, not of the router — the router change doesn't touch
  preprocessing or the engines' own recognition, so these specific
  failure modes are structurally unaffected by anything in this phase.
  Confirmed directly: `raw_results.csv`'s per-engine catastrophic failure
  rates (tesseract 14.1%, rapidocr 12.7%) are unchanged in kind from
  before this phase's router work — they're baseline-engine properties.
- **Not yet added**: a dedicated regression test asserting the FULL
  benchmark's summary numbers stay within tolerance after a routing
  change — this is exactly what `scripts/check_regression.py` (built in
  a prior phase) is for; `benchmark/results/baseline_summary.json` should
  be refreshed against this phase's authoritative run and re-checked
  before any future routing change ships.
