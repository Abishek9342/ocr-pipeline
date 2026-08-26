"""Reproduces the 2026-08-25 overnight finding that motivated removing
`is_noisy -> denoise()` from `ocr_resilience.preprocess.build_pipeline()`'s
default chain: applying non-local-means denoise specifically to images the
quality report flags as `is_noisy` made CER WORSE for both PaddleOCR and
Tesseract on the `noisy` (Gaussian noise) degradation preset, on this
project's synthetic corpus. See `preprocess.py`'s `denoise()` docstring
and `docs/engineering_backlog.md` for how this was found (a routing-v2
readiness audit noticed `ours`' mean CER on the `noisy` preset was worse
than Tesseract ALONE, which had no obvious explanation until traced here).

Run from the repo root: python docs/reproduce_denoise_gate_finding.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from benchmark.corpus import build_corpus
from benchmark.degrade import apply_degradation
from benchmark.run_benchmark import stable_seed
from ocr_resilience.engines import PaddleOCRAdapter, TesseractAdapter
from ocr_resilience.metrics import cer
from ocr_resilience.preprocess import deblur_unsharp, denoise, deskew
from ocr_resilience.quality import assess

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark", "_corpus_cache")


def main() -> dict:
    manifest = build_corpus(CORPUS_DIR)
    paddle, tess = PaddleOCRAdapter(), TesseractAdapter()
    results = {"with_denoise_paddle": [], "without_denoise_paddle": [],
               "with_denoise_tess": [], "without_denoise_tess": []}

    for item in manifest:
        clean = cv2.imread(item["path"])
        degraded = apply_degradation(clean, "noisy", seed=stable_seed(item["path"], "noisy"))
        truth = item["ground_truth"]
        report = assess(degraded)
        gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)

        g_with = gray.copy()
        if report.is_skewed:
            g_with = deskew(g_with, report.skew_angle_deg)
        if report.is_noisy:
            g_with = denoise(g_with)
        if report.is_blurry:
            g_with = deblur_unsharp(g_with)
        bgr_with = cv2.cvtColor(g_with, cv2.COLOR_GRAY2BGR)

        g_without = gray.copy()
        if report.is_skewed:
            g_without = deskew(g_without, report.skew_angle_deg)
        if report.is_blurry:
            g_without = deblur_unsharp(g_without)
        bgr_without = cv2.cvtColor(g_without, cv2.COLOR_GRAY2BGR)

        for engine, name in [(paddle, "paddle"), (tess, "tess")]:
            text_with = " ".join(d.text for d in engine.recognize(bgr_with))
            text_without = " ".join(d.text for d in engine.recognize(bgr_without))
            results[f"with_denoise_{name}"].append(cer(text_with, truth))
            results[f"without_denoise_{name}"].append(cer(text_without, truth))

    print(f"{'variant':25s} {'mean_cer':>10s} {'n':>4s}")
    for key, vals in results.items():
        print(f"{key:25s} {sum(vals) / len(vals):10.4f} {len(vals):4d}")
    return results


if __name__ == "__main__":
    main()
