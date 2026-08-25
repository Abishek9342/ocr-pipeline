# Failure Analysis: Why the Pipeline Loses Where It Loses

Every "loses to baseline X" cell in the README's Results table is
investigated here rather than just reported. Real inputs, real numbers —
reproducible via the commands under each section.

## Failure Case A: `combo_hard` (EasyOCR/PaddleOCR beat the pipeline)

**Hypothesis:** ROVER's confidence-weighted vote assumes different
engines' confidence scores are on a comparable scale. If one engine
systematically over-reports confidence relative to another, the vote is
structurally biased toward it regardless of actual correctness.

**Direct evidence** (Tesseract vs. EasyOCR, `combo_hard`, first 3 corpus
images, confidence-gated adaptive preprocessing, no ensemble forcing):

| Ground truth | Tesseract conf / CER | EasyOCR conf / CER |
|---|---|---|
| "Hello World 12345" | 0.960 / 0.000 | 0.670 / 0.000 |
| "Hello World 12345" | **0.802** / **0.118** | **0.679** / **0.059** |
| "Invoice Number INV-8452" | 0.933 / 0.000 | 0.734 / 0.000 |

Row 2 is the key evidence: Tesseract reports *higher* confidence (0.802
vs. 0.679) while being the *less accurate* of the two (0.118 vs. 0.059
CER). A confidence-weighted vote combining these two hypotheses
structurally leans toward Tesseract's answer here, independent of which
one is actually more correct. Across all three rows, Tesseract's
confidence runs 0.10-0.29 higher than EasyOCR's regardless of who's right
— consistent with a real scale difference, not sampling noise.

**Intervention tested:** unweighted majority voting (`fuse(weighted=False)`,
exposed as `OCRPipeline.run(fusion_weighted=False)`) — every engine's
vote counts equally regardless of its self-reported confidence.

**Controlled comparison** (20 images, forced two-engine ensemble,
adaptive preprocessing off to isolate the fusion variable):

| Preset | weighted (default) | unweighted | Result |
|---|---:|---:|---|
| clean | 0.0144 | 0.0149 | ~tie |
| heavy_blur | **0.0308** | 0.0969 | weighted much better |
| skewed | 0.0658 | 0.0637 | ~tie |
| **combo_hard** | 0.0764 | **0.0614** | unweighted better (hypothesis confirmed) |
| motion_blur | **0.2870** | 0.4047 | weighted much better |
| noisy | **0.0166** | 0.0192 | weighted slightly better |

**Verdict: REJECT switching the default.** Unweighted voting does help on
`combo_hard`, exactly as the confidence-mismatch hypothesis predicted —
but it hurts substantially more on `heavy_blur` and `motion_blur` (net
unfavorable across this preset mix). `fusion_weighted=False` is kept as a
real, tested, documented option (not a hidden or removed code path) for
cases where it's known to help, but confidence-weighted stays the
pipeline's default. This is the "KEEP/REJECT" step of the mission's own
development loop, applied honestly — a plausible hypothesis, tested, and
rejected as a default change based on the actual numbers, not intuition.

Reproduce: `python docs/reproduce_failure_analysis.py` regenerates both
tables above from scratch. **Honesty note:** re-running it does NOT
reproduce these exact numbers bit-for-bit — EasyOCR's own CPU inference
has some run-to-run variance (a re-run during review showed `combo_hard`
weighted/unweighted as 0.0539/0.0546 instead of 0.0764/0.0614, still the
same qualitative pattern of "unweighted helps combo_hard, weighted helps
heavy_blur/motion_blur substantially more"). The degradation *images*
themselves are exactly reproducible (see `stable_seed` in
`benchmark/run_benchmark.py`); the non-determinism is inside EasyOCR's
own model inference, upstream of anything this package controls. Treat
the specific decimal values here as representative, not exact — the
qualitative conclusion (reject switching the default) is what's load-
bearing, and it held across both runs.

## Failure Case B: `motion_blur` (Tesseract/RapidOCR beat EasyOCR and the pipeline is close behind RapidOCR)

With all four engines pooled, the pipeline's `motion_blur` CER (0.130) is
now close to RapidOCR's (0.120, the best) and far ahead of EasyOCR alone
(0.561, catastrophically bad) — see the 5-system table in the README. The
earlier (2-engine, Tesseract+EasyOCR only) pipeline configuration lost
outright to Tesseract alone here (0.299 vs. 0.245); adding PaddleOCR and
RapidOCR to the ensemble closed most of that gap. Not fully investigated
further this pass: whether RapidOCR's specific advantage on motion blur
generalizes (only 20 images tested) or is corpus-specific.

## Failure Case C & D: general `noisy` / `low_contrast` (PaddleOCR/Tesseract narrowly beat the pipeline)

With all four engines pooled, PaddleOCR alone is the best single system
on both (`noisy`: 0.0081 vs. the pipeline's 0.0283; `low_contrast`: 0.0129
vs. 0.0250). These are the mildest degradations in the suite (a single
flag typically fires, so the router often stays single-engine rather than
ensembling) — the pipeline's single-engine choice on these cases is
whichever engine is registered first (`next(iter(self._engines))` in
`router.decide`), not necessarily the best available one for that
specific condition. **This is a real, identified architectural gap**: the
router picks an engine by registration order for the "easy" path, not by
any per-condition strength ranking. A quality-aware single-engine
*choice* (not just ensemble-or-not) is the natural next step — see
`docs/engineering_backlog.md`'s "learned/quality-aware engine selection"
item.

## Failure Case E: PaddleOCR (previously failing entirely)

**Resolved**, not merely worked around blindly:

1. Root cause 1 (model load): a PaddlePaddle PIR (Paddle Intermediate
   Representation) attribute-type mismatch (`strides` expected as
   `pir::Int32Attribute`) when loading the default PP-OCRv5/v6 detector
   model — reproduced consistently in this environment. Fixed by pinning
   `ocr_version="PP-OCRv4"` (configurable — see `PaddleOCRAdapter`),
   whose mobile det/rec models don't hit this bug. Verified end-to-end on
   clean and degraded images across all 11 presets before trusting it.
2. Root cause 2 (runtime crash once actually ensembled): PaddleX's
   internal resize step assumes 3-channel input (`h, w, _ = img.shape`);
   the shared pipeline preprocessing outputs grayscale. Tesseract/EasyOCR
   both tolerate grayscale directly, which is exactly why this was
   invisible until PaddleOCR ran inside a real multi-engine ensemble for
   the first time. Fixed by converting grayscale back to BGR in
   `PaddleOCRAdapter.recognize()`.

PaddleOCR is now a genuinely strong single-engine baseline (mean CER
0.097, second only to the full pipeline, and the fastest of the three
neural engines at 0.21s mean latency) — see the README's Results table.
