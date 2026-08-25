# Benchmark Reproducibility Report

This is the detailed companion to the README's Results/Ablation Study
sections — enough for another engineer to reproduce the exact experiment,
not just read its conclusion. It documents one specific run (below); the
harness itself (`benchmark/run_benchmark.py`, `benchmark/run_ablation.py`)
is designed to be re-run by anyone, and machine-readable output
(`benchmark/results/benchmark.json`) is written on every run so this
report never has to be trusted blindly — cross-check it against that file.

## Environment (this reference run)

| | |
|---|---|
| Hardware | Intel64 Family 6 Model 183 Stepping 1 (GenuineIntel), 20 logical CPUs |
| OS | Windows-11-10.0.26200-SP0 |
| Python | 3.13.13 |
| opencv-python-headless | 5.0.0 |
| numpy | 2.3.5 |
| pandas | 3.0.5 |
| pytesseract | 0.3.13 |
| Tesseract binary | 5.4.0.20240606 (UB-Mannheim Windows build) |
| easyocr | 1.7.2 |
| paddleocr / paddlepaddle | 3.7.0 / 3.0.0 (fails to load — see below) |
| ocr-resilience | 0.2.0 |

Reproduce this table for your own environment with:
```bash
python --version
python -c "import cv2, numpy, pandas; print(cv2.__version__, numpy.__version__, pandas.__version__)"
pip show pytesseract easyocr paddleocr paddlepaddle
tesseract --version
```

## Dataset

Synthetic, rendered via PIL — see `benchmark/corpus.py`. 10 fixed sentences
(covering digits, currency, mixed case, hyphenated codes) x 2 styles
(`printed`: standard sans-serif font; `handwritten_proxy`: a cursive
TrueType font, an honest *proxy* for handwriting, not equivalent to real
handwritten strokes — see the README's Honesty notes) = 20 base images,
each with exact ground truth by construction (no manual transcription).

11 degradation presets applied on top (`benchmark/degrade.py`): clean,
light/heavy Gaussian blur, motion blur, Gaussian noise, salt-and-pepper
noise, rotation/skew (±8°), low contrast, smudges (inpainted-blob
occlusion), JPEG compression (quality=12), and a stacked `combo_hard`
(rotation + blur + noise + smudge together). Full parameterization is in
`DEGRADATION_PRESETS` in that module.

**Dataset size for the reference run:** 20 images x 11 presets x 3 systems
= 660 measured (image, preset, system) triples.

**Seeding:** each (image path, preset) pair gets a deterministic seed via
`stable_seed()` — a `sha256`-based hash, specifically NOT Python's
built-in `hash()` (which is salted per-process by default via
`PYTHONHASHSEED` randomization; an earlier version of this harness used
`hash()` and was, despite appearances, not actually reproducible across
runs/machines — see the README's debugging story, finding #7's sibling
issue was caught the same way, by re-running and comparing). Re-running
`run_benchmark.py`/`run_ablation.py` with the same code and same corpus
reproduces bit-identical degraded images.

## Methodology

Three systems compared: `tesseract` and `easyocr` called directly with
**no** preprocessing/routing/fusion (the baselines), and `ours` (this
package's `OCRPipeline`, built from whichever engines are requested and
load successfully). `paddleocr` is requested in the reference run's
`--engines` flag but fails to load in this environment (see below) and is
automatically excluded — `benchmark.json`'s `config.systems_skipped`
records the exact error, and no partial/incomplete rows for it are
included in `summary.csv` (a system that fails mid-run has its rows
dropped entirely, not averaged over a smaller, biased sample — see
`run_benchmark.py`'s `_write_outputs` call site).

Exact command:
```bash
python -m benchmark.run_benchmark --engines tesseract,easyocr,paddleocr,ours --presets all
```

Metrics: Character Error Rate (CER) and Word Error Rate (WER), both
normalized Levenshtein edit distance (`ocr_resilience/metrics.py`);
latency via `time.perf_counter()` around each system's call (for `ours`,
also broken into per-stage timings — quality assessment, preprocessing,
each engine call, fusion — in `latency.csv`); peak memory via Python's
`tracemalloc` around each call (Python-level allocations only — this does
NOT capture memory inside compiled OpenCV/PyTorch/PaddlePaddle C++/CUDA
internals, so the absolute numbers understate true peak RSS, especially
for EasyOCR's PyTorch backend; treat these as a *relative*, same-
methodology comparison between systems in this harness, not an absolute
memory budget).

## PaddleOCR load failure (this environment)

```
(InvalidArgument) Type of attribute: strides is not right.
  [Hint: Expected attributes.at("strides").dyn_cast<pir::ArrayAttribute>().at(i).isa<pir::Int32Attribute>()
  == true, but received attributes.at("strides").dyn_cast<pir::ArrayAttribute>().at(i)
  .isa<pir::Int32Attribute>():0 != true:1.] (at paddle\fluid\pir\dialect\operator\ir\pd_op3.cc:24692)
```

Occurs loading the PP-OCRv6 detector model, reproduced consistently
across attempts in this environment (PaddlePaddle 3.0.0, PaddleOCR 3.7.0,
Python 3.13.13, Windows). This is a PIR (Paddle Intermediate
Representation)/model-export attribute-type mismatch upstream, not
something this package's adapter can work around — `ocr_resilience.engines.PaddleOCRAdapter`
is written and verified API-correct (against the actually-installed v3
`.predict()` API via `inspect.signature`) independent of this failure.

## Results

See the README's Results and Ablation Study sections for the full tables
and their honest reading — not duplicated here to avoid the two documents
drifting apart. The underlying data: `benchmark/results/raw_results.csv`
(660 rows, one per (image, preset, system) triple), `summary.csv`
(aggregated), `latency.csv` (per-stage breakdown for `ours`), and
`benchmark.json` (the same summary plus the exact config used, in machine-
readable form — reproduced in this report's Environment section above).

## Error analysis

The two most informative individual failures in this run, both already
covered as regression tests in `tests/`:

- **`tesseract` alone on `salt_pepper`: CER exactly 1.0 across every one
  of the 20 images.** Tesseract's segmentation returns zero detections on
  raw salt-and-pepper noise (confirmed directly: `TesseractAdapter().recognize(degraded)`
  returns `[]`). This is the baseline's expected behavior (no
  preprocessing) — the pipeline (`ours`) fixes this specific case (CER
  0.019) via a purpose-built impulse-noise detector + median filter; see
  the README's debugging story #7.
- **`easyocr` alone on `motion_blur`: mean CER 0.5614**, by far the worst
  single cell in the whole benchmark. Motion blur — a directional,
  elongated point-spread function — is qualitatively different from the
  Gaussian blur EasyOCR's training presumably saw more of; unlike
  Tesseract (0.245) or `ours` (0.299), EasyOCR's recognizer seems to
  degrade sharply rather than gracefully here. Not investigated further in
  this pass — see Roadmap in the README.

## Ablation methodology

`benchmark/run_ablation.py` isolates each pipeline component via
`OCRPipeline.run()`'s three ablation parameters — `skip_preprocessing`
(bypass `build_pipeline` entirely), `force_step` (apply exactly one named
preprocessing operator unconditionally, bypassing the quality gate), and
`force_ensemble` (bypass `router.decide()`, forcing single-engine or
all-engines) — rather than reimplementing pipeline logic separately for
the ablation. Same corpus/seeding as above, restricted to 5 presets
(clean, heavy_blur, skewed, noisy, combo_hard) x 20 images x 8 variants =
800 measured cells, to keep the ablation's runtime reasonable while still
covering a clean case, a pure-blur case, a pure-skew case, a pure-noise
case, and a stacked-degradation case.

## Reproducing this report

```bash
git clone https://github.com/Abishek9342/ocr-pipeline.git
cd ocr-pipeline
pip install -e ".[all,benchmark,dev]"
python -m benchmark.run_benchmark --engines tesseract,easyocr,paddleocr,ours --presets all
python -m benchmark.run_ablation
```

Numbers will differ from this report wherever your environment differs
(OS, Tesseract build/version, EasyOCR/PaddleOCR model versions, CPU) — see
the README's Roadmap for what would need to change to validate these
rankings against a real (non-synthetic) document dataset instead.
