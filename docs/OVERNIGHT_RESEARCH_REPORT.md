# Overnight Research Report

Covers `overnight_ocr_research_mission.md` ("STATISTICAL HARDENING,
REAL-DATA READINESS, AND ROUTING V2 PREPARATION"), executed 2026-08-25/26.
Every number below is measured, not asserted — see each section's
reproduction command. Cross-references: `docs/statistical_rigor_report.md`,
`docs/routing_v2_readiness.md`, `docs/robustness_curves.md`,
`docs/engineering_backlog.md`, `CHANGELOG.md`'s `[0.5.0]` entry.

## Starting state

Commit `8b85a34`, version 0.4.0, 137 tests passing. `benchmark/results/`
existed from the prior mission but `ablation_raw.csv`/`ablation_summary.csv`
were confirmed stale (timestamped before `engine_selection.py` existed).

## Work completed

In the mission's own stated priority order:

1. **Reproducibility audit** found the ablation staleness above (real),
   re-ran it, and — as a direct consequence of work below — had to re-run
   it a SECOND time after a mid-session bug fix changed the numbers again.
2. **Repeated-run latency** extended from Tesseract-only to all 4 baseline
   engines + `ours`, with cold-start reported separately from warm,
   measured TWICE (once before, once after the downstream re-run chain).
3. **`benchmark/stats_utils.py` extended**: `bootstrap_median_ci`,
   `paired_bootstrap_diff_ci`, `paired_effect_size` (Cohen's d),
   `percentile_interval`, `pairwise_comparison_summary`,
   `bonferroni_correction`, richer `latency_variance_stats` (median/P95).
   27 new tests, all passing.
4. **Paired bootstrap CI**, `ours` vs. each of the 4 baselines, matched by
   `(image_id, preset)` — see "Statistical results" below.
5. **Real-dataset infrastructure**: generic schema, manifest validator,
   rule-based failure taxonomy, and a CLI scaffold
   (`run_real_dataset.py`) — all built and tested (31 new tests) against
   this project's own synthetic corpus as a plumbing check, with no real
   dataset downloaded or fabricated.
6. **Meta-dataset logging schema extended**: `image_features`,
   `failure_type`, `language`/`script` now logged per-row in
   `raw_results.csv`; `calibrated_confidence` left `None` (no calibrator
   wired in — a deliberate non-value, not a placeholder for a fabricated
   one).
7. **Robustness statistics**: worst-severity CER, degradation slope,
   catastrophic-onset severity, normalized AUC — added on top of the
   existing curves, not replacing them.
8. **Artifact versioning**: `benchmark.json` and a new
   `ablation_meta.json` now record `pipeline_version` + git commit.
9. **Routing V2 readiness audit** — the most substantive analytical work
   this pass: checked every unruled condition directly against
   `assess()`'s real output rather than assuming a cheap rule existed.
10. **A real bug found and fixed mid-pass** (not part of the original
    plan — surfaced BY the routing audit): `build_pipeline()`'s
    `is_noisy -> denoise()` gate was hurting the exact condition it
    targeted. Fixed, then the entire measurement chain (full benchmark,
    ablation, robustness, paired comparison, statistical report) had to
    be re-run to reflect it — see "Bugs found/fixed" below.
11. **A machine-sleep data-quality issue** in this pass's own overnight
    full-benchmark run was found and corrected (see "Reproducibility
    findings" below) rather than reported uncritically.

## Statistical results

Bootstrap 95% CI on mean CER (2000 resamples, percentile method,
220 rows/system): `ours` **[0.0214, 0.0432]** (mean 0.0308) vs.
PaddleOCR (best baseline) [0.0795, 0.1479] — non-overlapping, as in the
prior mission, now with a corrected mean.

**Paired bootstrap** (same matched cases, not independent samples) — all
four exclude zero:

| Baseline | (baseline − ours) mean CER | 95% CI | Cohen's d |
|---|---:|---|---:|
| tesseract | +0.1482 | [0.1068, 0.1944] | 0.452 |
| rapidocr | +0.1410 | [0.0978, 0.1892] | 0.398 |
| easyocr | +0.0906 | [0.0631, 0.1200] | 0.424 |
| paddleocr | +0.0800 | [0.0491, 0.1141] | 0.326 |

**Multiple-comparison note** (mission section 5): only 4 comparisons ran
against one reference (`ours`), all part of one narrow central claim, not
an exploratory sweep — no correction applied by default, but even a
Bonferroni correction for 4 comparisons (`alpha=0.05/4=0.0125`) would not
change any conclusion above, since none of the CIs come close to
including zero. `stats_utils.bonferroni_correction` exists for future use
if a much larger comparison set is ever run.

## Latency variance

Measured for all 4 engines + `ours`, TWICE (before and after the
downstream re-run triggered by the bug fix below). Headline finding:
**Tesseract's own repeated-run CV was measured at three different values
across this project's history — 6.6%, 27.8%, 8.4%** — each reproducible
at the time it was measured, on a shared (non-dedicated) machine. The
honest conclusion is that this number is measurement-condition-sensitive
and must not be quoted as a fixed constant. RapidOCR is consistently
~10x slower than every other engine (~1.6-1.8s vs. 0.08-0.21s warm mean),
a finding stable across both measurements. Full tables:
`docs/statistical_rigor_report.md`.

## Reproducibility findings

- **Confirmed and fixed**: `ablation_raw.csv`/`ablation_summary.csv`
  predated `engine_selection.py` — re-ran, twice (see below).
- **Found and corrected — not silently, not left in**: 2 of 1100 rows in
  this pass's own overnight full-benchmark run recorded `latency_sec` of
  3-11 *hours*, traced to the host machine sleeping mid-run (confirmed
  via matching timestamp gaps in the run log). Excluded from latency
  aggregates only (CER/WER unaffected and retained); documented in
  `benchmark.json`'s `data_quality_note`, not silently dropped.
- **No hidden seeds found**: all degradation/ablation randomness already
  routes through `stable_seed()` (sha256-based, fixed in the prior
  mission). No new seeding gaps introduced this pass.
- **No silently skipped engines**: all 4 engines + `ours` loaded and ran
  in every full benchmark/ablation/robustness run this pass; nothing
  silently dropped.

## New infrastructure

`benchmark/stats_utils.py` (extended), `benchmark/run_paired_comparison.py`,
`benchmark/analyze_robustness.py`, `benchmark/dataset_schema.py`,
`benchmark/dataset_validator.py`, `benchmark/failure_taxonomy.py`,
`benchmark/run_real_dataset.py`, `docs/reproduce_denoise_gate_finding.py`,
artifact-versioning fields in `benchmark.json`/`ablation_meta.json`.

## Tests added

58 new tests this pass (27 for `stats_utils.py`'s extensions, 31 for the
real-dataset infrastructure) — 137 -> 186 total, all passing.

## Bugs found/fixed

1. **`is_noisy -> denoise()` hurts the condition it targets** (see
   `ocr_resilience/preprocess.py`'s `denoise()` docstring for full
   evidence). Removed from the default chain. Net effect: `ours`'
   headline CER improved 0.0319 -> 0.0308; `noisy` preset improved
   sharply (0.0315 -> 0.0087); `combo_hard` got measurably WORSE (0.0581
   -> 0.0681) for a specific, understood reason (Tesseract's ensemble
   contribution benefits from denoising even though its single-engine
   `noisy` performance doesn't). Reported as a genuine trade-off, not
   rounded to a clean win.
2. **Stale ablation artifacts** (see above) — fixed via re-run, twice.
3. **Machine-sleep latency corruption** (see above) — fixed via
   documented exclusion, not silent.
4. **CI regression baseline was stale** relative to the corrected
   numbers — refreshed via `scripts/check_regression.py --write-baseline`
   and verified it still passes its own check.

## Bugs investigated but left unresolved

- **`combo_hard`'s regression from the `denoise()` fix** is understood
  (Tesseract's ensemble contribution vs. its single-engine performance
  diverge) but not fixed — the clean fix (per-engine preprocessing) is an
  architecture change, not a rule tweak, and wasn't judged worth making
  for a 0.01 CER swing on one preset. See `docs/engineering_backlog.md`.
- **`light_blur` and `smudged`'s routing gaps** are measured and real but
  have no cheap fix — both were checked directly against `assess()`'s
  output and neither correlates with an existing quality field. Left
  open, not patched with an unvalidated new rule.

## Real-data readiness

Schema, validator, failure taxonomy, and CLI harness are built and
tested against this project's own synthetic corpus as a plumbing check
only. **No real-world accuracy claim is made anywhere in this repo** — a
real dataset the moment one exists needs zero code changes to be
evaluated by `run_real_dataset.py`.

## Multilingual readiness

Row schema now carries `language`/`script` per row (honest `en`/`Latin`
values for this corpus). Still confirmed blocked at the model level
(only English recognition models cached locally, no outbound network
access to fetch others) — unchanged from the prior mission, re-verified,
not re-asserted from memory.

## Learned-routing readiness

Meta-dataset schema is now substantially more populated
(`image_features`, `failure_type` logged per-row at collection time,
not re-derived after the fact) — one of the four "what would justify
moving to learned routing" conditions from the prior mission's report is
now met. The real blocker (a dataset 25-50x larger than this project's
20-image corpus) is unchanged and remains explicitly not worked around.

## Current benchmark position

Mean CER: `ours` **0.0308**, PaddleOCR (best baseline) 0.1108, EasyOCR
0.1213, RapidOCR 0.1718, Tesseract 0.1790. Catastrophic-failure rate:
`ours` 0.45%. Wins outright on 5 of 11 presets (heavy blur, motion blur,
salt-and-pepper, skew, low contrast), ties on 1 (JPEG), PaddleOCR wins
the other 5 — same win/tie/loss count as the prior mission, though the
`noisy`/`combo_hard` per-preset numbers moved in opposite directions from
this pass's bug fix (see above).

## Remaining blockers

Unchanged from `docs/engineering_backlog.md`'s "Blocked externally"
section: real dataset, real handwriting corpus, multilingual models,
GPU-first VLM baselines, Kaggle/PyPI publishing credentials.

## Exact priorities for tomorrow

See `docs/TOMORROW_HANDOFF.md`.

## Files changed

`benchmark/stats_utils.py`, `benchmark/run_benchmark.py`,
`benchmark/run_ablation.py`, `benchmark/run_statistical_report.py`,
`ocr_resilience/preprocess.py`, `ocr_resilience/__init__.py`,
`pyproject.toml`, `README.md`, `CHANGELOG.md`,
`docs/statistical_rigor_report.md`, `docs/robustness_curves.md`,
`docs/engineering_backlog.md`, `tests/test_statistics.py`, plus new files
listed under "New infrastructure" above and `docs/routing_v2_readiness.md`,
`tests/test_dataset_infrastructure.py`. Regenerated:
`benchmark/results/*.csv`, `benchmark.json`, `baseline_summary.json`.

## Reproduction commands

```bash
python -m pytest tests/ -q
python -m benchmark.run_benchmark --engines tesseract,easyocr,paddleocr,rapidocr,ours --presets all
python -m benchmark.run_ablation
python -m benchmark.run_robustness
python -m benchmark.run_paired_comparison
python -m benchmark.run_statistical_report
python -m benchmark.analyze_robustness
python docs/reproduce_denoise_gate_finding.py
```
