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
| paddleocr / paddlepaddle | 3.7.0 / 3.0.0 (works via `ocr_version="PP-OCRv4"` — see below) |
| rapidocr / onnxruntime | 3.9.2 / 1.29.0 |
| ocr-resilience | 0.2.0 |

Reproduce this table for your own environment with:
```bash
python --version
python -c "import cv2, numpy, pandas; print(cv2.__version__, numpy.__version__, pandas.__version__)"
pip show pytesseract easyocr paddleocr paddlepaddle rapidocr onnxruntime
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

**Dataset size for the reference run:** 20 images x 11 presets x 5 systems
= 1,100 measured (image, preset, system) triples.

**Seeding:** each (image path, preset) pair gets a deterministic seed via
`stable_seed()` — a `sha256`-based hash, specifically NOT Python's
built-in `hash()` (which is salted per-process by default via
`PYTHONHASHSEED` randomization; an earlier version of this harness used
`hash()` and was, despite appearances, not actually reproducible across
runs/machines — see the README's debugging story). Re-running
`run_benchmark.py` with the same code and same corpus reproduces
bit-identical degraded *images*. **Not** bit-identical: EasyOCR's own CPU
inference has measurable run-to-run variance (confirmed directly — see
`docs/failure_analysis.md`'s honesty note); treat CER/WER numbers
involving EasyOCR as representative, not exact to the last decimal.

## Methodology

Five systems compared: `tesseract`, `easyocr`, `paddleocr`, `rapidocr`
called directly with **no** preprocessing/routing/fusion (the baselines),
and `ours` (this package's `OCRPipeline`, pooling all four engines,
adaptively routed). PaddleOCR previously failed to load entirely in this
environment; both root causes are now fixed (see
`docs/failure_analysis.md`, Failure Case E) and it participates fully. If
any engine DOES fail to load or crashes mid-run, `benchmark.json`'s
`config.systems_skipped` records the exact error, and no partial/
incomplete rows for it are included in `summary.csv` (a system that fails
mid-run has its rows dropped entirely, not averaged over a smaller,
biased sample — see `run_benchmark.py`'s `_write_outputs` call site).

Exact command:
```bash
python -m benchmark.run_benchmark --engines tesseract,easyocr,paddleocr,rapidocr,ours --presets all
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

## PaddleOCR: previously failed, now resolved

Full root-cause analysis and fix in `docs/failure_analysis.md` (Failure
Case E) — briefly: the default PP-OCRv5/v6 detector model hits a
PaddlePaddle PIR attribute-type mismatch on load (worked around via
`ocr_version="PP-OCRv4"`), and a second, separate bug crashed it the
first time it actually ran inside a real multi-engine ensemble (grayscale
input assumption in PaddleX's resize step, fixed in
`PaddleOCRAdapter.recognize()`). Both fixes were verified end-to-end
before this reference run.

## Results

See the README's Results and Ablation Study sections for the full tables
and their honest reading — not duplicated here to avoid the two documents
drifting apart. The underlying data: `benchmark/results/raw_results.csv`
(1,100 rows, one per (image, preset, system) triple), `summary.csv`
(aggregated), `latency.csv` (per-stage breakdown for `ours`), and
`benchmark.json` (the same summary plus the exact config used, in machine-
readable form — reproduced in this report's Environment section above).

## Error analysis

The most informative individual failures in this run — full investigation
in `docs/failure_analysis.md`:

- **`tesseract` alone on `salt_pepper`: CER exactly 1.0 across every one
  of the 20 images.** Tesseract's segmentation returns zero detections on
  raw salt-and-pepper noise (confirmed directly: `TesseractAdapter().recognize(degraded)`
  returns `[]`). This is the baseline's expected behavior (no
  preprocessing) — the pipeline (`ours`) fixes this specific case (CER
  0.026) via a purpose-built impulse-noise detector + median filter.
- **`rapidocr` alone on `heavy_blur`: CER exactly 1.0 across every one of
  the 20 images** — the same complete-failure pattern as Tesseract on
  salt-and-pepper, but a different engine and a different degradation
  type. Not investigated further at the root-cause level this pass (the
  pipeline avoids it via ensembling — CER 0.0065 — but *why* RapidOCR's
  specific PP-OCRv6-small models fail this completely on Gaussian blur
  isn't understood, only observed).
- **`easyocr` alone on `motion_blur`: mean CER 0.5614**, the worst
  single-baseline cell in the whole benchmark. Motion blur — a
  directional, elongated point-spread function — is qualitatively
  different from the Gaussian blur EasyOCR's training presumably saw more
  of. RapidOCR (0.120) and the pipeline (0.130) both handle it far better.
- **Single-engine PaddleOCR wins 6 of 11 presets outright** (clean, light
  blur, general noise, skew, low contrast, smudged) — see Failure Case
  C/D in `docs/failure_analysis.md` for why the adaptive pipeline's
  router doesn't currently pick PaddleOCR specifically for these cases
  (it picks the first-registered engine on the "easy" path, not
  necessarily the strongest one for that condition — an identified,
  unfixed gap).

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
python -m benchmark.run_benchmark --engines tesseract,easyocr,paddleocr,rapidocr,ours --presets all
python -m benchmark.run_ablation
```

Numbers will differ from this report wherever your environment differs
(OS, Tesseract build/version, EasyOCR/PaddleOCR model versions, CPU) — see
the README's Roadmap for what would need to change to validate these
rankings against a real (non-synthetic) document dataset instead.
