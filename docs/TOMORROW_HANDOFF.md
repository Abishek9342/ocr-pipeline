# Tomorrow Handoff

**CURRENT VERSION**: 0.5.0 (bumped from 0.4.0 this pass)

**LATEST COMMIT**: see `git log -1` at the time this was written — this
overnight pass's work is committed as a single commit on top of `8b85a34`.

**CURRENT BENCHMARK**: `ours` mean CER **0.0308** (95% CI [0.0214,
0.0432]), beats every baseline with all 4 paired-bootstrap CIs excluding
zero. Wins outright on 5/11 presets, ties 1, PaddleOCR wins the other 5.
Catastrophic-failure rate 0.45%.

**TEST COUNT**: 186 passing (up from 137).

## What changed tonight

- Extended `benchmark/stats_utils.py` (paired bootstrap, effect size,
  median CI, percentile interval, pairwise comparison, Bonferroni).
- Built real-dataset infrastructure end-to-end (schema, validator,
  failure taxonomy, CLI harness) — no real dataset used or fabricated.
- Extended per-row logging (`image_features`, `failure_type`,
  `language`/`script`) and per-engine repeated-latency measurement.
- Added robustness summary statistics and artifact versioning.
- **Found and fixed a real bug**: `is_noisy -> denoise()` was hurting
  the condition it targeted. Fixing it improved the headline number but
  measurably worsened `combo_hard` — a genuine, documented trade-off.
- **Found and corrected** a machine-sleep latency-measurement artifact in
  this pass's own benchmark run (2 rows, hours-long recorded latency).
- Ran the full measurement chain (benchmark/ablation/robustness/paired-
  comparison/statistical-report) TWICE — once, then again after the bug
  fix above — and refreshed every downstream doc + the CI regression
  baseline to match.

## What is definitively solved

- Statistical rigor on the headline claim (bootstrap CI, now also
  paired, both exclude zero for `ours` vs. every baseline).
- Real-dataset harness is fully built and tested — plug in a real
  manifest, zero code changes needed.
- The `denoise()` bug — was silently hurting accuracy on the condition it
  targeted; now removed, with the trade-off it introduced elsewhere
  fully documented rather than hidden.
- Ablation/robustness staleness — re-run and now version-stamped so
  future staleness is mechanically detectable, not just found by luck.

## What remains open

- `combo_hard`'s regression from the `denoise()` fix (understood, not
  fixed — needs a per-engine-preprocessing architecture change).
- `smudged` and `motion_blur`'s routing gaps — confirmed to need a NEW
  quality signal each, not a rule using an existing field.
- `light_blur`'s routing gap — real, small, no cheap fix found.
- Every "Blocked externally" item in `docs/engineering_backlog.md` is
  still blocked (real dataset, multilingual models, learned routing's
  data-scale requirement, GPU-first VLM baselines, Kaggle/PyPI
  credentials) — nothing new attempted or claimed there.

## Top 3 next tasks

1. **Design and validate a new quality signal for `smudged`** (highest
   measured routing gap: 0.027 achieved CER vs. 0.008 achievable) — the
   condition-engine evidence already exists in
   `benchmark/results/condition_engine_table.csv`; the missing piece is
   a signal that actually correlates with this specific degradation,
   confirmed NOT to be any existing `QualityReport` field.
2. **Directional-blur detection for `motion_blur`** — same category of
   work, second-highest priority (0.233 achieved vs. 0.120 achievable on
   the ~35% of images that don't reach the full ensemble).
3. **Decide whether the `combo_hard` trade-off is worth a per-engine-
   preprocessing architecture change**, or should be accepted and
   documented as a permanent, understood limitation — this is a design
   decision, not an engineering task, and shouldn't be resolved by
   default in either direction without deliberately choosing.

## First command to run tomorrow

```bash
python -m pytest tests/ -q && python -m benchmark.run_benchmark --engines tesseract,easyocr,paddleocr,rapidocr,ours --presets all
```

Confirms the test suite and headline benchmark are exactly where this
handoff says they are before building on top of them.

## Important files

`docs/OVERNIGHT_RESEARCH_REPORT.md` (full narrative), `docs/routing_v2_readiness.md`
(the per-condition audit behind tomorrow's top 3), `docs/statistical_rigor_report.md`,
`docs/engineering_backlog.md` (Blocked/Buildable/Deferred), `ocr_resilience/preprocess.py`'s
`denoise()` docstring (the mixed-result bug fix, in full), `benchmark/results/benchmark.json`
(now version-stamped — check `git_commit` against HEAD before trusting a cached result).

## Known caveats

- Tesseract's repeated-run latency CV has now been measured at 3
  different values (6.6%, 27.8%, 8.4%) on this shared machine — never
  quote it as a fixed constant.
- The `combo_hard` CER number (0.0681) is worse than an earlier draft of
  this same overnight pass's numbers (0.0581) — this is the correct,
  final number, not a regression to chase down; it's the documented cost
  of the `denoise()` fix.
- This machine went to sleep mid-run during this pass's overnight full
  benchmark — if a future run shows implausible latency (seconds becomes
  hours), check for the same artifact before trusting it.
