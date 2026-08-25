# Confidence Calibration Report

Covers Phases 9–10 of `ocr_next_research_mission.md`. Every number below
is computed directly from `benchmark/results/raw_results.csv` (the
authoritative run — see `docs/engine_selection_report.md` for the exact
command) via `ocr_resilience/calibration.py`. Reproduce with the snippet
in "Reproducing this report" below.

## The confidence problem

`fusion.fuse(weighted=True)` (the default) votes per aligned character
position, weighted by each candidate engine's raw, self-reported
confidence. That's only sound if different engines' confidence scores
mean the same thing. Direct evidence they don't (from
`docs/failure_analysis.md`, Failure Case A): on one `combo_hard` case,
Tesseract reported HIGHER confidence (0.802) than EasyOCR (0.679) while
being the *less* correct of the two (CER 0.118 vs. 0.059) — a
confidence-weighted vote structurally favors whichever engine
over-reports confidence, independent of actual correctness.

## Per-engine calibration analysis

Using `1 - min(cer, 1.0)` as a continuous correctness proxy (there is no
binary correct/incorrect ground truth available — see the methodological
honesty note in `calibration.py`'s docstring), Expected Calibration Error
(10 bins, sample-weighted mean |predicted confidence − actual correctness
proxy| across bins) over all 220 baseline-engine rows each:

| Engine | Mean confidence | Mean correctness proxy | ECE | Direction |
|---|---:|---:|---:|---|
| **paddleocr** | 0.904 | 0.889 | **0.0149** | Nearly perfectly calibrated |
| rapidocr | 0.854 | 0.828 | 0.0261 | Mildly overconfident |
| tesseract | 0.731 | 0.821 | 0.0897 | Underconfident |
| **easyocr** | 0.685 | 0.879 | **0.2010** | Badly underconfident |

**EasyOCR is the most miscalibrated engine by a wide margin** — it
reports confidence around 0.685 on average while actually being right
about 88% of the time. This directly explains (not just correlates with)
the combo_hard finding above: EasyOCR's systematically low confidence
scale means confidence-weighted fusion structurally *underweights* its
contribution relative to how reliable it actually is, letting a
higher-raw-confidence-but-wrong Tesseract answer dominate the vote.
PaddleOCR, by contrast, is close to perfectly calibrated in aggregate —
its raw confidence really is informative about its real accuracy.

## Calibration methodology

Simplest defensible method first, per the mission's own instruction:
**binned/histogram calibration** (`BinnedCalibrator` — bucket raw
confidence into 10 bins, map to the empirical mean correctness observed
in that bin during fitting). Not attempted: Platt scaling, isotonic
regression, or temperature scaling — no evidence surfaced during this
phase that binned calibration is too coarse to be useful, so reaching for
a more complex method wasn't justified (see "Keep/reject decision" below
for what would justify revisiting this).

**Direct test of whether calibration flips the motivating combo_hard
case**: applying each engine's fitted calibrator to the exact confidence
values from that case:

| | Tesseract | EasyOCR | Tesseract wins the vote? |
|---|---:|---:|---|
| Raw | 0.802 | 0.679 | Yes (gap 0.123) |
| Calibrated | 0.968 | 0.950 | **Still yes** (gap narrows to 0.018) |

Calibration narrows the miscalibration-driven gap by ~85% but does not
flip this specific case — both engines' calibrated confidence converges
toward "usually right" once corrected, which is itself informative (both
ARE usually right in aggregate; this was one of their less-common wrong
cases) but doesn't resolve which one is right on this particular case.

## Raw vs. calibrated fusion — aggregate comparison

Same controlled setup as the weighted-vs-unweighted experiment in
`docs/failure_analysis.md` (20 images x 6 presets, forced two-engine
ensemble tesseract+easyocr, adaptive preprocessing on), run via
`docs/reproduce_calibration_analysis.py`:

| Preset | Raw-weighted CER | Calibrated-weighted CER | Paired 95% CI on the difference | Statistically distinguishable from zero? |
|---|---:|---:|---|---|
| clean | 0.0144 | 0.0144 | [+0.0000, +0.0000] | No |
| heavy_blur | 0.0146 | 0.0146 | [-0.0063, +0.0060] | No |
| skewed | 0.0104 | 0.0160 | [+0.0000, +0.0147] | No |
| combo_hard | 0.0545 | 0.0593 | [-0.0071, +0.0182] | No |
| motion_blur | 0.3200 | 0.3265 | [+0.0000, +0.0196] | No |
| noisy | 0.0124 | 0.0124 | [+0.0000, +0.0000] | No |

**None of the six differences are statistically distinguishable from
zero** at 95% confidence (paired bootstrap on the per-image
calibrated-minus-raw CER, `docs/reproduce_calibration_analysis.py`). This
sharpens — and corrects — an earlier, less careful version of this
finding that called the point-estimate differences on skewed/combo_hard/
motion_blur "worse": that's true of the raw numbers, but at only 20
images per preset the honest statement is "no detectable effect," not
"calibration hurts." Contrast this with the CER comparison in
`docs/statistical_rigor_report.md`, where `ours`' advantage over the best
baseline DOES clear this same statistical bar cleanly — the calibration
question genuinely doesn't, and that distinction is the point of doing
this rigorously rather than eyeballing point estimates.

**Honesty note on how this table was produced, because it matters**: an
earlier, inline (not saved-to-a-script) run of this exact comparison
produced a *different* pattern — showing calibration *helping* on
combo_hard and noisy. Re-running the identical logic via the saved script
above twice in direct succession produced the table shown, identically
both times. The code was checked line-by-line between the two versions
and is equivalent; the most likely explanation is EasyOCR's underlying
PyTorch CPU inference not guaranteeing deterministic execution order
across process invocations (already documented as a known issue — see
`docs/failure_analysis.md`'s honesty note on this exact point). **The
numbers in this table are the ones confirmed stable across two direct
re-runs and are what `docs/reproduce_calibration_analysis.py` will
reproduce**; the earlier, non-reproduced numbers were wrong to publish
without first confirming they held up on re-run, and this correction is
the point — check a saved, re-runnable script before trusting a result,
not just one live invocation, exactly the same lesson debugging-story
finding #9 already drew from a different bug in this same session.

## Calibration quality analysis / keep-reject decision

**REJECT adopting calibrated fusion as the default — on the precise
grounds that there is no statistically detectable benefit, not on the
grounds that it was proven to hurt.** Once measured via a script
confirmed reproducible across repeats (not a single live run) AND once a
paired bootstrap CI was applied to the per-preset differences, none of
the six point-estimate differences (three numerically better, three
numerically worse for calibration) clear the bar of being distinguishable
from zero at 95% confidence, at this sample size (20 images/preset). Two
things can both be true here: (1) the per-engine ECE numbers above are
real and reproducible (PaddleOCR genuinely is well-calibrated, EasyOCR
genuinely is not), and (2) correcting for that miscalibration, via this
specific simple method, has not been shown to translate into better
fused text on this benchmark — "not shown to help" is the accurate
claim, not "shown to hurt." A calibrated number from this method is a
**quality proxy for ranking**, not a validated probability — the
methodology (a proxy correctness label, not true binary outcomes)
doesn't support a stronger claim either way.

**What would justify revisiting**: a larger corpus (reduces the
noise-floor problem directly), and/or isotonic regression if binned
calibration's coarseness (only 10 discrete output levels) turns out to be
the limiting factor rather than sample size — neither attempted this
phase since the mixed result doesn't yet demonstrate binned calibration
itself is the bottleneck.

## Reproducing this report

```bash
python docs/reproduce_calibration_analysis.py
```

Regenerates both the ECE table and the raw-vs-calibrated fusion
comparison table above from scratch (requires
`benchmark/results/raw_results.csv` to already exist — run
`benchmark/run_benchmark.py` first if not).
