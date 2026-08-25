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
| **ours** | **0.0319** | **[0.0229, 0.0442]** |
| paddleocr | 0.1108 | [0.0795, 0.1479] |
| easyocr | 0.1213 | [0.0952, 0.1501] |
| rapidocr | 0.1718 | [0.1292, 0.2175] |
| tesseract | 0.1790 | [0.1338, 0.2256] |

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

## Repeated-run latency variance

Tesseract, the same degraded image, run 10 times in immediate succession
(isolates timing noise from image-to-image variance):

| Mean (s) | Std (s) | CV | Min (s) | Max (s) |
|---:|---:|---:|---:|---:|
| 0.0852 | 0.0056 | 6.6% | 0.0819 | 0.1013 |

A 6.6% coefficient of variation for Tesseract's own repeated timing is
modest — consistent with it being a single-threaded classical algorithm
with little opportunity for the kind of thread-scheduling non-determinism
already documented for EasyOCR's PyTorch backend (see
`docs/failure_analysis.md`'s honesty note and
`docs/confidence_calibration_report.md`'s reproducibility correction —
neither measured EasyOCR's OWN repeated-run variance directly, only
observed that two separate full-benchmark runs disagreed). **Measuring
EasyOCR's repeated-run CV directly, the same way, is the natural next
step** — not done here to keep this report's own runtime small (a single
Tesseract measurement takes ~1s for 10 repeats; the equivalent for
EasyOCR would need its model loaded first, a heavier one-time cost worth
doing deliberately rather than folding into this quick report).

## What this does and doesn't change

- Every benchmark table elsewhere in this repo remains a point estimate;
  this report doesn't retroactively add CIs to all of them, only
  demonstrates the method and applies it to the headline claim (`ours`
  vs. the best baseline) that most needed it.
- The correctness proxy / CER metric itself is unchanged — this only
  quantifies uncertainty in the ESTIMATE of the mean, not a new accuracy
  claim.
- Latency variance was measured for one engine (Tesseract) on one image.
  Generalizing this to "how noisy is latency measurement across this
  whole benchmark" would need the same treatment for EasyOCR/PaddleOCR/
  RapidOCR too — flagged above as the natural next step, not done here.

## Reproducing this report

```bash
python -m benchmark.run_statistical_report
```

Requires `benchmark/results/raw_results.csv` to already exist (run
`benchmark/run_benchmark.py` first if not). Writes
`cer_confidence_intervals.csv` and `latency_variance.csv` to
`benchmark/results/`.
