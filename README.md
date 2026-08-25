# ocr-resilience

A **classical computer-vision preprocessing + multi-engine ensemble layer**
built on top of existing OCR engines (EasyOCR, Tesseract, PaddleOCR) — not
a replacement for them, and not an LLM-based correction pass. No trained
model of its own anywhere in this package: every technique is a
well-established, decades-old CV method (Laplacian-variance blur
detection, Sauvola binarization, CLAHE, ROVER-style multi-hypothesis
voting), applied deliberately and only where a benchmark actually showed
it helps.

## Problem

Off-the-shelf OCR engines are excellent on clean documents but degrade
unevenly — and in different ways from each other — on messy real-world
input: a blurry phone photo, a skewed scan, a smudged receipt, salt-and-
pepper scanner noise, handwriting. Worse, no single engine is uniformly
best across all of that; the benchmark below shows each of Tesseract and
EasyOCR alone has degradation types where it *catastrophically* fails
(complete blank output, or majority-wrong text) while the other engine
handles the same case fine.

## Approach

Quality-aware preprocessing plus consensus across multiple engines can
meaningfully reduce that catastrophic-failure risk, without training a new
OCR model (which would require millions of labeled images and GPU-cluster
compute neither engineering time nor this environment has) and without an
LLM correction step (by explicit design choice — this is a computer-vision
project, not a language-model one). A classical quality assessor decides
*which* preprocessing operators are actually needed for a given image, and
*whether* a second engine is worth the latency — rather than always
running everything on everything.

## Architecture

```
image -> quality.assess()        classical CV metrics: blur, noise,
              |                  impulse noise, contrast, skew angle,
              v                  handwriting-likelihood
      preprocess.build_pipeline() targeted chain: deskew -> median-denoise
              |                   (impulse noise) -> denoise (general) ->
              v                   deblur -> smudge removal -> contrast
                                  enhancement (each step only runs if the
                                  quality report says it's needed)
      router.decide()            clean image -> one fast engine;
              |                  hard image (blur+noise+skew stacked, or
              v                  handwriting-like) -> ensemble everything
    engines.{Tesseract,EasyOCR,PaddleOCR}Adapter
              |
              v
      fusion.fuse()               spatial grouping (overlap-ratio, not
              |                    plain IoU — see below) + ROVER-style
              v                    confidence-weighted character voting
      postprocessing.postprocess_text()   whitespace/unicode normalization
                                           (raw_text preserved separately)
```

## Results

Full run: 20 synthetic corpus images (10 sentences x printed + cursive-
font "handwritten proxy" — see the honesty note below) x **all 11**
degradation presets x {Tesseract, EasyOCR, PaddleOCR, RapidOCR — each
alone, no preprocessing — vs. this pipeline pooling all four}.

**Overall (all styles, all degradations) — `python -m benchmark.run_benchmark --engines tesseract,easyocr,paddleocr,rapidocr,ours --presets all`:**

| System | Mean CER ↓ | Mean WER ↓ | Mean latency (s) ↓ | P95 latency (s) ↓ | Mean peak memory ↓ |
|---|---:|---:|---:|---:|---:|
| **ours** | **0.0319** | **0.1739** | 1.234 | 3.282 | 44.7 MB |
| paddleocr\_alone | 0.1108 | 0.3228 | 0.198 | 0.266 | 3.6 MB |
| easyocr\_alone | 0.1213 | 0.3626 | 0.961 | 1.481 | 3.3 MB |
| rapidocr\_alone | 0.1718 | 0.2382 | 1.212 | 1.496 | 161.0 MB |
| tesseract\_alone | 0.1790 | 0.3205 | 0.200 | 0.665 | 0.1 MB |

**Mean CER by degradation preset:**

| Preset | ours | tesseract | easyocr | paddleocr | rapidocr | Who wins |
|---|---:|---:|---:|---:|---:|---|
| clean | 0.0166 | 0.0166 | 0.0249 | **0.0106** | 0.0544 | paddleocr |
| light\_blur | 0.0143 | 0.0143 | 0.0415 | **0.0043** | 0.0558 | paddleocr |
| heavy\_blur | **0.0065** | 0.0701 | 0.1358 | 0.0695 | 1.0000 (total failure) | **ours** |
| motion\_blur | **0.0870** | 0.2447 | 0.5614 | 0.2327 | 0.1201 | **ours** |
| noisy | 0.0315 | 0.0187 | 0.0452 | **0.0056** | 0.0839 | paddleocr |
| salt\_pepper | **0.0831** | 1.0000 (total failure) | 0.1213 | 0.8051 | 0.2000 | **ours** |
| skewed | **0.0043** | 0.1245 | 0.2009 | 0.0064 | 0.0442 | **ours** |
| low\_contrast | **0.0085** | 0.0209 | 0.0568 | 0.0129 | 0.0562 | **ours** |
| smudged | 0.0270 | 0.0270 | 0.0541 | **0.0081** | 0.0704 | paddleocr |
| jpeg\_compressed | **0.0142** | 0.0142 | 0.0449 | 0.0437 | 0.0819 | tie (ours/tesseract) |
| combo\_hard | 0.0581 | 0.4180 | 0.0478 | **0.0199** | 0.1227 | paddleocr |

**Genuinely closer to "wins about as often as it loses" now, after this
session's quality-aware engine-selection work** (see
`docs/engine_selection_report.md`) — the pipeline wins outright on 5 of
11 presets (heavy blur, motion blur, salt-and-pepper, skew, low contrast),
ties on 1 (JPEG), and single-engine PaddleOCR still wins outright on the
other 5 (clean, light blur, general noise, smudged, and now `combo_hard`
too — PaddleOCR alone beats every other system on that case). PaddleOCR
remains dramatically faster (0.20s vs. 1.23s mean) and is still the
better choice if latency matters more than the last percent of accuracy.
**What the ensemble buys, beyond the categories it wins outright, is
avoiding each single engine's worst-case catastrophic failure**:
Tesseract goes completely blank on salt-and-pepper noise (CER 1.0),
RapidOCR goes completely blank on heavy Gaussian blur (CER 1.0), and
PaddleOCR falls apart on salt-and-pepper (0.81) — three different
engines, three different specific blind spots, none of which the
pipeline inherits. That's also precisely why its mean latency (1.23s)
and memory (44.7MB — RapidOCR's own ONNX runtime allocation alone
averages 161MB when it runs) are much higher than any single engine:
consistency and now genuinely competitive per-condition accuracy are
bought with real, measured compute cost, not free. Whether that trade is
worth it depends on your use case — this table is what lets you decide,
not a headline claim.

**Is the overall mean-CER gap real, or could it be noise at this sample
size?** Bootstrap 95% confidence intervals (`docs/statistical_rigor_report.md`):
`ours` [0.023, 0.044] vs. PaddleOCR (the best baseline) [0.080, 0.148] —
the intervals don't overlap at all, at 220 samples per system. The four
baseline engines are mostly *not* statistically distinguishable from each
other this way (their intervals overlap); the pipeline's overall
advantage over all of them is.

Run `python -m benchmark.run_benchmark --engines tesseract,easyocr,paddleocr,rapidocr,ours --presets all`
yourself; raw per-image results, an aggregated `summary.csv`, per-stage
`latency.csv`, and a machine-readable `benchmark.json` land in
`benchmark/results/`.

## Ablation Study

Which component is actually responsible for the improvement over doing
nothing? `benchmark/run_ablation.py` measures each cell below as an
independent real run (20 images x 5 presets: clean, heavy_blur, skewed,
noisy, combo_hard):

| Variant | Mean CER ↓ | Mean WER ↓ | Mean latency (s) |
|---|---:|---:|---:|
| baseline (no preprocessing, single engine) | 0.0898 | 0.2475 | 0.134 |
| + deskew | 0.0653 | 0.2007 | 0.124 |
| + denoise (non-local means) | 0.1111 | 0.2755 | 0.222 |
| + CLAHE contrast | 0.1602 | 0.3278 | 0.121 |
| + Sauvola binarization | 0.0889 | 0.2223 | 0.117 |
| + adaptive preprocessing (quality-gated chain) | **0.0503** | 0.1925 | 0.153 |
| + multi-engine selection (forced ensemble) | 0.0320 | 0.1837 | 0.399 |
| **full pipeline (adaptive + multi-engine)** | **0.0350** | **0.1813** | 0.309 |

**Honest reading, not a marketing one:**

- **Denoise and CLAHE, forced unconditionally on every image, each made
  results *worse* than doing nothing** (0.111 and 0.160 CER vs. 0.090
  baseline). This is exactly why they're gated on the quality report
  instead of always-on — this table is the evidence for that design
  choice, not an assumption.
- Deskew and Sauvola binarization each help in isolation.
- The **gated adaptive combination (0.050) beats every individual forced
  operator alone**, including the ones that helped individually — the
  components interact, and gating matters as much as the operators do.
- **Multi-engine selection is the single largest lever** (0.032, ~3x lower
  than baseline) but also by far the most expensive (~3x baseline
  latency) — adaptive routing exists specifically to only pay that cost
  on images that need it, which is why the full pipeline's latency (0.31s)
  sits below forced-ensemble-always (0.40s) at nearly the same accuracy.
- **Post-processing (whitespace/unicode normalization) showed no
  measurable CER difference** (0.035 raw vs. 0.035 processed) on this
  rendered synthetic corpus — an honest null result on this corpus
  specifically, not a claim it never helps (this corpus never produces
  the double-spaces/encoding artifacts a real scanned document would).

Run `python -m benchmark.run_ablation` yourself; results land in
`benchmark/results/ablation_summary.csv` and `ablation_raw.csv`.

## Robustness Curves

A single CER number per degradation type (the tables above) can hide a
sharp failure threshold sitting just past the tested severity.
`benchmark/run_robustness.py` sweeps blur/noise/skew/JPEG severity across
5 levels each rather than one fixed point per type. **Headline finding:
Tesseract has a sharp failure cliff on Gaussian noise between sigma 25
and 50** — CER jumps from 0.035 straight to a complete failure (1.0) and
stays there, while EasyOCR and the pipeline degrade gracefully at every
tested severity, including sigma=80. The main benchmark's `noisy` preset
(sigma=25) sits just below that cliff, which is why Tesseract looked
essentially fine there (0.0166 CER) — a single-severity snapshot doesn't
show you where the cliff is. Full curves, including a JPEG-quality sweep
that shows no real trend (an honest null result, not a hidden one), are
in `docs/robustness_curves.md`. Reproduce: `python -m benchmark.run_robustness`.

### The debugging story behind these numbers (worth reading before trusting them)

The first working version of this pipeline was **worse than either engine
alone** (overall CER 0.28 vs. 0.07-0.09) — every one of these was caught by
the benchmark itself, not by code review, and fixed before any number in
this README was written:

1. **Sauvola binarization actively hurt accuracy** when stacked with other
   preprocessing. It's a template-matching-era technique; modern
   LSTM/CRNN-based engines are trained on natural grayscale images and
   their own feature extractors already handle contrast internally.
   Ablation: on one heavy-blur case, deblurring alone got both engines to
   0.000 CER — binarizing on top of that pushed EasyOCR to 0.235, *worse
   than doing no preprocessing at all* (0.118). Removed from the default
   pipeline entirely (still available as a standalone function, and — per
   the ablation table above — helps *on its own*, just not stacked in).
2. **Plain IoU failed on word-vs-line granularity mismatches.** Tesseract
   returns one box per word; EasyOCR often returns one box per line. A
   word box fully contained in a line box scores a tiny IoU (union
   dominated by the much larger box) even though it's 100% the same
   region — so the two engines' outputs never got merged, and the fused
   text was the two outputs concatenated (roughly doubling every
   transcription). Fixed by switching to intersection-over-smaller-area.
3. **Greedy single-link clustering didn't handle transitivity.** Even
   after fixing the overlap metric, a single bridging detection (one
   line box overlapping three separate word boxes that don't overlap
   *each other*) only joined the first matching group it happened to
   check, leaving the rest stranded as singletons. Fixed with proper
   union-find clustering.
4. **The skew-correction angle was inverted.** `deskew()` was rotating
   images the WRONG way — actively increasing skew instead of correcting
   it — since the codebase first existed. Caught only because the
   benchmark showed the "skewed" preset getting *worse* after
   "correction," not by inspecting the rotation math directly. One sign
   flip fixed it; CER on that preset dropped from 0.35 to 0.02.
5. **Fixing bug #2/#3 introduced a new bug.** After the fusion fixes, a
   final reading-order sort was added so multi-line output wouldn't come
   out in incidental engine/dict order. The first version sorted purely
   by `(y_min, x_min)` — which immediately regressed an *undegraded*
   ("clean") test image from ~0.04 CER to 0.83, because cursive/italic
   fonts give words on the SAME visual line noticeably different y_min
   values (ascenders/descenders). Fixed by clustering into lines via
   vertical-center proximity before sorting left-to-right within each
   line.
6. **`OCRResult.text` joined every detection with a newline, regardless of
   which line it actually belonged to.** Tesseract's default one-box-per-
   word output on a single line of text ("Hello World 12345") rendered as
   three separate output *lines* instead of one space-joined line. This
   sat undetected through bugs #2-#5 above because `benchmark/run_benchmark.py`
   always bypassed `.text` and space-joined `result.detections` directly
   itself — it never exercised the property that is now the CLI's and
   `OCRResult.raw_text`'s primary output. Caught only by running the CLI
   end-to-end against a real engine. Fixed by reusing the same line-
   clustering logic `_reading_order` already used for sort order, to also
   join words-on-a-line with spaces and lines with newlines.
7. **Salt-and-pepper noise made Tesseract return completely empty output
   (CER 1.0), and the quality assessor never noticed.** `noise_score` is a
   median-absolute-deviation (MAD) estimator — a *robust* statistic,
   specifically insensitive to sparse outlier pixels, which is exactly
   what salt-and-pepper noise is. So `is_noisy` never fired, no denoising
   ever ran, and Tesseract's segmentation broke outright on the raw noisy
   image. Caught by widening the benchmark from 8 to all 11 presets (the
   original benchmark never ran `salt_pepper` at all). Fixed by adding a
   *second*, complementary noise signal (`impulse_noise_score`: fraction
   of pixels sharply off their local 3x3 median) and gating a median
   filter — the standard, specific fix for impulse noise — on it. CER on
   that preset dropped from 1.000 to 0.019.
8. **PaddleOCR crashed the first time it actually ran inside a real
   multi-engine ensemble**, after its model-loading bug (above) was fixed
   — `ValueError: not enough values to unpack (expected 3, got 2)`, deep
   inside PaddleX's own resize step (`h, w, _ = img.shape`). The shared
   pipeline preprocessing outputs grayscale (2D) images; Tesseract and
   EasyOCR both accept that directly, so this was invisible in every
   prior single/two-engine test — only surfaced once PaddleOCR was one of
   three-plus engines actually pooled together. Fixed by converting
   grayscale input back to 3-channel BGR in `PaddleOCRAdapter.recognize()`.
9. **Adding a confidence-weighted vote assumes different engines'
   confidence scores are on the same scale — investigated directly, and
   they're not.** On `combo_hard`, Tesseract reported *higher* confidence
   than EasyOCR even in a case where Tesseract was the wrong answer (0.802
   vs. 0.679, but EasyOCR's text had lower CER), which structurally biases
   ROVER's weighted vote toward whichever engine over-reports confidence
   regardless of correctness. Tested the obvious fix (unweighted majority
   voting) directly: it helps on `combo_hard` (0.076 -> 0.061 CER) but
   *hurts* on `heavy_blur` (0.031 -> 0.097) and `motion_blur` (0.287 ->
   0.405) — net effect across presets unfavorable, so the default stays
   confidence-weighted (`OCRPipeline.run(fusion_weighted=False)` is kept
   as a real, tested option, not adopted as the default). See
   `docs/failure_analysis.md` for the full experiment.
10. **The benchmark's own reproducibility claim had a second hole in it,
    found while building a new experiment (severity-sweep robustness
    curves) that happened to call the noise-degradation functions
    directly.** `gaussian_noise()` and `salt_and_pepper()` called
    `np.random.normal`/`np.random.randint` directly — reading NumPy's own
    *global* random state, completely ignoring the `rng` parameter every
    other degradation function in the same file already respects.
    Confirmed directly: `apply_degradation(img, "noisy", seed=42)` called
    twice produced two different images. This meant the `noisy`,
    `salt_pepper`, and `combo_hard` presets were silently NOT
    seed-reproducible the whole time — including in every number in this
    README's Results table above them. Fixed by deriving a seeded
    `numpy.random.Generator` from the same `rng` object instead. The
    existing reproducibility regression test only checked the `skewed`
    preset (which never touched numpy's global state, so it never
    caught this) — now parametrized over every preset.

Each of these has a regression test in `tests/` reproducing the exact
failure mode, not just testing the fixed behavior in isolation. #5 and #6
are worth sitting with together: two different "the output text itself is
wrong" bugs, both invisible to the *existing* benchmark/test suite for a
long time because neither ever exercised the exact code path (`.text`)
that a real downstream user (the CLI) actually calls — a reminder that a
benchmark only catches what it actually measures, and it's worth
periodically asking what a benchmark *isn't* exercising, not just trusting
that a passing one means nothing is wrong.

### Honesty notes

- **"Handwritten" test images use a cursive TrueType font (Windows'
  bundled Lucida Handwriting), not real handwriting samples.** Real
  handwriting has far more irregular stroke geometry than any font can
  produce — treat the handwriting numbers here as a lower bound on
  real-world difficulty, not an equivalent test. No real handwriting
  corpus (e.g. IAM) was available in this environment.
- **PaddleOCR now genuinely works, but needed two real fixes to get
  there** (previously skipped entirely — see the debugging story below,
  findings #6 and #7). Its default model version (PP-OCRv5/v6 detector)
  hits a reproducible upstream PaddlePaddle PIR attribute-type mismatch on
  load in this environment; `PaddleOCRAdapter` defaults to the older
  `ocr_version="PP-OCRv4"` (configurable) to avoid it, and separately
  needed its grayscale-input path fixed once it actually ran inside a
  multi-engine ensemble for the first time.
- **RapidOCR** (a fourth engine, the same PP-OCR model family as
  PaddleOCR but exported to ONNX Runtime with no PyTorch/Paddle/TF
  dependency) was added after researching the current open-source OCR
  ecosystem — see `docs/engine_landscape.md` for the full comparison
  (Surya, docTR/OnnxTR, MMOCR, Kraken, TrOCR also evaluated, and why they
  weren't chosen). It's markedly weaker on this benchmark than expected
  from its "lightweight" design (mean CER 0.175, worst of the four single
  engines, and a complete failure on heavy blur) — a genuinely disappointing
  result worth stating plainly rather than glossing over just because it
  was the recommended pick going in.
- This is a **synthetic, rendered-text benchmark**, not real scanned
  documents or photographs — see Limitations below.

## Installation

```bash
pip install ocr-resilience[easyocr,tesseract]   # or [paddleocr] / [all]
```

Tesseract requires the native binary installed separately (not pip-
installable) — see the [UB-Mannheim Windows builds](https://github.com/UB-Mannheim/tesseract/wiki)
or your OS package manager (`apt install tesseract-ocr`, `brew install tesseract`).

## Usage

```python
from ocr_resilience import OCR

ocr = OCR(engine="auto", preprocessing="adaptive", return_boxes=True)
result = ocr.predict("document.jpg")

print(result.text)                 # fused, line-reconstructed text
print(result.confidence)           # mean per-detection confidence
print(result.processed_text)       # whitespace/unicode-normalized (raw_text preserved separately)
print(result.routing.reason)       # why this many engines ran
print(result.preprocessing_pipeline)  # exactly what was applied and why
print(result.to_dict())            # JSON-serializable: raw_text, processed_text, confidence,
                                    # bounding_boxes, engine_used, preprocessing_pipeline, processing_time
```

Lower-level access (specific engine instances, ablation flags) via
`OCRPipeline` directly:

```python
from ocr_resilience import OCRPipeline

pipeline = OCRPipeline.with_engines(["tesseract", "easyocr"])
result = pipeline.run("document.png")
results = pipeline.run_batch(["a.png", "b.png"])
```

### CLI

```bash
ocr-pipeline document.jpg
ocr-pipeline document.jpg --engine auto --output result.json
ocr-pipeline ./scans/ --output results/     # batch: one JSON per input, one bad file doesn't abort the rest
ocr-pipeline document.jpg --debug-dir ./debug   # writes original/preprocessed/annotated images for inspection
```

## Architecture in code

- `quality.py` — classical CV image-quality metrics, no learned weights.
- `preprocess.py` — deskew / median-denoise (impulse noise) / non-local-
  means denoise / unsharp deblur / CLAHE / Sauvola (opt-in) / smudge
  removal; `build_pipeline()` composes only what the quality report says
  is needed.
- `router.py` — quality-aware single-engine-vs-ensemble routing; the
  single-engine choice itself is delegated to...
- `engine_selection.py` — quality-aware single-engine selection
  (`select_primary_engine`, interpretable rules keyed on `QualityReport`'s
  continuous fields, not registration order) and ranked confidence-based
  fallback ordering (`rank_fallback_chain`). See `docs/engine_selection_report.md`.
- `calibration.py` — binned confidence calibration, reliability curves,
  Expected Calibration Error, calibrated fusion — investigated as a fix
  for cross-engine confidence-scale mismatch, real miscalibration found,
  but rejected as a fusion-weighting default (didn't improve fused
  output). See `docs/confidence_calibration_report.md`.
- `engines.py` — `OCREngine` protocol + Tesseract/EasyOCR/PaddleOCR/
  RapidOCR adapters.
- `fusion.py` — ROVER-style multi-engine consensus (spatial grouping via
  overlap-ratio + union-find, then confidence-weighted character voting;
  `weighted=False` for unweighted majority voting — see the debugging
  story finding #9 for why weighted stays the default).
- `postprocessing.py` — whitespace/unicode normalization, confidence
  filtering, dedup — text-level only, never overwrites the raw output.
- `scoring.py` — an optional composite score with named, published
  weights and a `rank_stability()` check for whether a ranking claim
  survives reasonable reweighting — never a replacement for the
  underlying per-metric numbers.
- `pipeline.py` — orchestrates the above; `OCRPipeline.run()` exposes
  `skip_preprocessing` / `force_step` / `force_ensemble` / `fusion_weighted`
  ablation hooks (used by `benchmark/run_ablation.py`) plus
  `min_confidence_for_fallback` for a genuine second-pass escalation to
  every available engine when the first pass's confidence is low.
- `debug.py` — visual debugging export (`OCRPipeline.run(debug_dir=...)`,
  `ocr-pipeline --debug-dir`): original/preprocessed/annotated images.
- `cli.py` — the `ocr-pipeline` command.
- `benchmark/run_robustness.py` — severity-sweep robustness curves (not
  just one CER per degradation type); `scripts/check_regression.py` — a
  CI regression gate against a stored baseline (`benchmark/results/baseline_summary.json`).

## Limitations

- Benchmarked only on **synthetic, rendered text** (10 sentences x
  printed/cursive-font-proxy x 11 controlled degradations) — not real
  scanned documents, receipts/forms/invoices, natural scene text, or
  screenshots. The relative rankings above may not transfer to those.
- The "handwritten" style is a cursive font, not real handwriting (see
  Honesty notes).
- English only — no multilingual testing has been done.
- Single-column reading order only — `_reading_order`'s line-clustering
  heuristic has no multi-column/layout-detection logic.
- No perspective/four-point document correction, no orientation
  (90/180/270°) detection — only sub-degree deskew.
- Single-engine PaddleOCR is the best individual system on 8 of 11
  presets (see `docs/engine_selection_report.md`'s condition table), and
  plain EasyOCR wins the stacked `combo_hard` case — the pipeline's
  advantage is the near-zero catastrophic-failure rate (0.5% vs. 5-14%
  for single engines), not being the single best on every condition.
- `select_primary_engine` has explicit rules for only 4 of 11 conditions;
  the other 7 fall through to registration order (which happens to
  coincide with reasonable choices for several of them, per the evidence,
  but isn't asserted as a rule yet).
- Confidence calibration was investigated (binned/histogram method,
  `calibration.py`) and real per-engine miscalibration was found (EasyOCR
  ECE 0.20 vs. PaddleOCR's 0.015) — but applying it to fusion weighting
  did not improve, and measurably hurt, fused output in a controlled
  comparison. See `docs/confidence_calibration_report.md` for the full
  investigation and why it wasn't adopted despite the real underlying
  finding.
- No real (non-synthetic) dataset, no dev/val/test split, no held-out
  challenge set — see `docs/engineering_backlog.md` for the full list of
  what a "world-class" version of this project would still need and why
  each item isn't done here (resource constraints: real licensed
  datasets, multilingual corpora, GPU budget, or simply scope).

## Roadmap

- A real (non-synthetic) benchmark set — scanned documents, receipts, or
  a licensed handwriting corpus (e.g. IAM) — to validate whether the
  synthetic-benchmark rankings above transfer to real images.
- Extend `select_primary_engine`'s explicit rules to the remaining 7
  conditions, and make `rank_fallback_chain` condition-aware instead of
  one aggregate ranking — both using evidence already collected in
  `benchmark/results/condition_engine_table.csv`, no new experiments
  required.
- Isotonic regression for confidence calibration, if a larger corpus
  shows binned calibration's coarseness (not sample size) was the actual
  limiting factor — not yet demonstrated, so not attempted.
- Perspective/four-point document correction and orientation detection.
- docTR/OnnxTR as a fifth engine (see `docs/engine_landscape.md` — the
  strongest not-yet-added candidate from that research).
- Multilingual benchmarking — every engine here supports far more than
  English; none of that capability has been exercised or measured.
- A held-out challenge set and CI regression gating (see
  `docs/engineering_backlog.md`) once there's more than one contributor
  for "don't tune against the test set" discipline to matter for.

## Running the tests / benchmark yourself

```bash
git clone https://github.com/Abishek9342/ocr-pipeline.git
cd ocr-pipeline
pip install -e ".[all,benchmark,dev]"
pytest tests/                                       # unit + regression tests, no OCR binary required (adapters mocked)
python -m benchmark.run_benchmark --presets all      # full benchmark -> benchmark/results/
python -m benchmark.run_ablation                     # ablation study -> benchmark/results/ablation_*.csv
```

See `CONTRIBUTING.md` for the ground rule on benchmark-backed PRs, and
`CHANGELOG.md` for release history.
