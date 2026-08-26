# Statistical Rigor Report

Addresses mission section 21 and the routing mission's Phase 21 gap:
every benchmark number published in this repo up to now has been a
single-run point estimate with no uncertainty quantification. This adds
two things, using the simplest defensible method for each
(`benchmark/stats_utils.py`):

1. **Bootstrap confidence intervals** on each system's mean CER — is the
   difference between two systems' means actually distinguishable, or
   could it be sampling noise at this corpus size (220 rows per system)?
2. **Repeated-run latency variance** — how much does timing itself
   fluctuate run-to-run, independent of image-to-image differences (which
   the main benchmark's median/P95 already covers)?

## Bootstrap CI on mean CER (95%, 2000 resamples, percentile method)

| System | Mean CER | 95% CI |
|---|---:|---|
| **ours** | **0.0308** | **[0.0214, 0.0432]** |
| paddleocr | 0.1108 | [0.0795, 0.1479] |
| easyocr | 0.1213 | [0.0952, 0.1501] |
| rapidocr | 0.1718 | [0.1292, 0.2175] |
| tesseract | 0.1790 | [0.1338, 0.2256] |

(`ours`' mean/CI moved slightly from an earlier version of this report,
0.0319 → 0.0308, after `ocr_resilience/preprocess.py`'s `denoise()`
gate was removed from the default chain — see README.md's Ablation Study
section and `docs/routing_v2_readiness.md` for why. The baselines are
identical: they call each engine directly with no preprocessing, so this
change couldn't affect them.)

**This is the single most statistically meaningful confirmation of the
pipeline's advantage in this project so far**: `ours`' entire 95% CI
(up to 0.0442) sits below PaddleOCR's — the best single baseline —
entire CI (down to 0.0795). At 220 samples per system, this isn't a
borderline result close enough to worry about noise; the intervals don't
even come close to overlapping. Every OTHER pairwise comparison among
the four baseline engines DOES have overlapping intervals (e.g.
PaddleOCR's upper bound 0.148 vs. EasyOCR's lower bound 0.095) — so "the
four baselines are hard to statistically distinguish from each other on
this corpus, but the pipeline is distinguishable from all of them" is the
honest, precise claim, not "everything here is definitively different
from everything else."

## Paired bootstrap: ours vs. each baseline, matched by case (2026-08-25 overnight addition)

The CI comparison above treats each system's 220 rows as independent
samples — correct as a first check, but it throws away the fact that
`ours` and every baseline were measured on the exact same 220
`(image_id, preset)` cases. `benchmark/run_paired_comparison.py` instead
computes, per matched case, `baseline_engine_cer - ours_cer` and
bootstraps the CI on that paired difference directly
(`stats_utils.pairwise_comparison_summary`) — the statistically correct
method for "is `ours` really better than PaddleOCR on the SAME images,"
since pairing cancels out case-to-case variance that both systems share:

| Baseline engine | (baseline − ours) mean CER | 95% CI | Distinguishable from 0? | Cohen's d |
|---|---:|---|---|---:|
| tesseract | +0.1482 | [0.1068, 0.1944] | Yes | 0.452 |
| rapidocr | +0.1410 | [0.0978, 0.1892] | Yes | 0.398 |
| easyocr | +0.0906 | [0.0631, 0.1200] | Yes | 0.424 |
| paddleocr | +0.0800 | [0.0491, 0.1141] | Yes | 0.326 |

All four paired 95% CIs exclude zero — `ours` measurably beats every
baseline, including PaddleOCR, on the matched-case comparison, not just
in the weaker independent-sample sense above. Effect sizes are small-to-
medium by Cohen's convention (0.32–0.45), consistent with a real but not
enormous per-case advantage that adds up over the corpus rather than a
handful of cases dominating the result. Only 4 comparisons were run here
(one central claim tested 4 ways against the same reference), so no
multiple-comparison correction was applied — see
`stats_utils.bonferroni_correction` if a stricter, corrected threshold is
ever needed; at `alpha=0.05` corrected for 4 comparisons
(`bonferroni_correction(0.05, 4) = 0.0125`), all four would still clear a
95%+ bar since none of the CIs come close to including zero. Reproduce:
`python -m benchmark.run_paired_comparison` (writes
`benchmark/results/paired_comparison.csv`).

## Repeated-run latency variance (all engines + ours, 2026-08-25 extension)

The same degraded image, run 10 times in immediate succession per
system (isolates timing noise from image-to-image variance). The
engine/pipeline object is constructed once before timing starts, but the
FIRST `recognize()`/`run()` call is still reported separately from the
remaining 9 ("cold" vs. "warm") since lazy initialization inside the
first inference call (thread-pool spin-up, ONNX/Paddle session warmup)
is a real, distinct cost from steady-state timing — collapsing the two
would hide a real effect inside "noise":

| System | Cold start (s) | Warm mean (s) | Warm std (s) | Warm CV | Warm median (s) | Warm P95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| tesseract | 0.657 | 0.083 | 0.007 | **8.4%** | 0.081 | 0.102 |
| paddleocr | 0.353 | 0.126 | 0.005 | 3.6% | 0.126 | 0.134 |
| easyocr | 0.236 | 0.170 | 0.011 | 6.6% | 0.163 | 0.188 |
| rapidocr | 1.598 | 1.644 | 0.145 | 8.8% | 1.692 | 1.879 |
| **ours** | 0.745 | 0.127 | 0.006 | 4.6% | 0.124 | 0.137 |

(This is the SECOND time this exact 10-repeat measurement was run — the
table's own reproduce command was simply re-executed the next day as
part of the same overnight pass's downstream re-runs, not a special
re-measurement.)

Three findings that update earlier, narrower claims:

- **Tesseract's own repeated-run CV has now been measured at three
  different values in this project — 6.6%, 27.8%, and 8.4%** — across a
  standalone script run, a longer multi-system run, and a second
  multi-system run the next day, respectively. All three are real and
  each was reproducible AT THE TIME it was measured. **The only honest
  conclusion left standing is that Tesseract's repeated-run latency
  variance is measurement-condition-sensitive on this (non-dedicated,
  shared) machine and its CV must never be quoted as a fixed constant**
  — a direct, now twice-repeated application of this project's own data-
  integrity rule: don't preserve a prettier earlier number once a new
  measurement contradicts it, and don't assume the SECOND number is the
  "real" one either just because it came later — report the pattern
  (condition-sensitive), not a cherry-picked point estimate.
- **RapidOCR is dramatically and consistently slower than every other
  engine on this machine** (~1.6-1.8s vs. 0.08–0.21s warm mean for the
  others, stable across both measurement occasions) — an order of
  magnitude difference not visible in any previous report, since no prior
  benchmark isolated single-engine latency this directly. This is
  consistent with RapidOCR being invoked through a fresh ONNXRuntime
  session per adapter instance rather than any inherent slowness of the
  ONNX format itself; not root-caused further here (see
  `docs/engineering_backlog.md`).
- **`ours`' warm mean (0.127s, both measurements) sits close to
  PaddleOCR's (0.126s)**, well below a naive expectation that a multi-
  engine pipeline must be slower than every single engine — on this
  specific (`clean`-preset) test image, routing evidently selects a
  single fast engine rather than paying for the full ensemble every time,
  consistent with the adaptive-routing design intent, though this
  one-image measurement is not a claim about the pipeline's latency
  distribution across the whole corpus (see
  `benchmark/results/summary.csv`'s median/P95 columns for that).

## What this does and doesn't change

- Every benchmark table elsewhere in this repo remains a point estimate;
  this report doesn't retroactively add CIs to all of them, only
  demonstrates the method and applies it to the claims that most needed
  it (the headline `ours`-vs-best-baseline comparison, both independent
  and now paired).
- The correctness proxy / CER metric itself is unchanged — this only
  quantifies uncertainty in the ESTIMATE of the mean, not a new accuracy
  claim.
- Latency variance is now measured for all 4 baseline engines plus
  `ours`, each on the same single test image — a real extension of the
  earlier Tesseract-only measurement, but still one image, not a claim
  about variance across the whole corpus.

## Reproducing this report

```bash
python -m benchmark.run_statistical_report
```

Requires `benchmark/results/raw_results.csv` to already exist (run
`benchmark/run_benchmark.py` first if not). Writes
`cer_confidence_intervals.csv` and `latency_variance.csv` to
`benchmark/results/`.
