# Contributing

## Ground rule

**No benchmark claim ships without a number to back it.** If a change
touches preprocessing, routing, fusion, or post-processing, run the
relevant benchmark/ablation before and after, and include both numbers in
the PR description. A PR that changes accuracy-relevant code without a
benchmark run attached will be asked to add one before review.

## Setup

```bash
git clone https://github.com/Abishek9342/ocr-pipeline.git
cd ocr-pipeline
pip install -e ".[all,benchmark,dev]"
```

`tesseract`, `easyocr`, and `paddleocr` are optional — the core test suite
(`pytest tests/`) mocks all three engine adapters at their import boundary
and does not require any of the underlying binaries/models. You only need
a real install to run `benchmark/run_benchmark.py` or
`benchmark/run_ablation.py` against a real engine, or to work on an
adapter itself. Tesseract additionally requires the native binary (not
pip-installable) — see the README's Installation section.

## Before opening a PR

```bash
ruff check .              # lint
pytest tests/ -v          # unit + regression tests
python -m benchmark.run_benchmark --presets clean,heavy_blur,skewed  # quick sanity check, ~1 min
```

## Adding a new OCR engine adapter

Implement the `OCREngine` protocol in `ocr_resilience/engines.py`
(`name: str` + `recognize(image) -> list[Detection]`), register it in
`AVAILABLE_ENGINES`, and add adapter-level tests following the pattern in
`tests/test_engines.py` (mock the underlying library at the `sys.modules`
boundary — no real model download needed to test the adapter's own
data-shaping logic).

## Adding a preprocessing technique

Add the function to `ocr_resilience/preprocess.py`. Do NOT wire it into
`build_pipeline()`'s default chain until an ablation
(`benchmark/run_ablation.py`) shows it measurably helps for some real
image condition — see that module's existing results and `preprocess.py`'s
own docstring for why Sauvola binarization, for example, is implemented
but deliberately excluded from the default chain. A technique that doesn't
help is still worth keeping (as an opt-in function, documented with the
finding) — just not defaulted on.

## Reporting a bug

Please include: the input image (or a minimal repro if it can't be
shared), the exact command/code that produced the wrong result, what you
expected, and what you got. If it's an accuracy regression, a
before/after CER or the specific output text is more useful than a
description of "worse."

## Code style

`ruff check .` is the only enforced style gate (see `[tool.ruff]` in
`pyproject.toml`). No separate formatter is required.
