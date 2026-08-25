"""Reproduces the exact experiments behind docs/failure_analysis.md's
Failure Case A (confidence-calibration hypothesis for combo_hard). Not
part of the installed package or the test suite — a standalone script for
anyone who wants to re-verify the finding rather than trust the numbers
pasted in the markdown.

Run from the repo root: python docs/reproduce_failure_analysis.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from benchmark.corpus import build_corpus
from benchmark.degrade import apply_degradation
from benchmark.run_benchmark import stable_seed
from ocr_resilience.engines import EasyOCRAdapter, TesseractAdapter
from ocr_resilience.metrics import cer
from ocr_resilience.pipeline import OCRPipeline
from ocr_resilience.preprocess import build_pipeline
from ocr_resilience.quality import assess


def per_engine_confidence_vs_accuracy(preset: str, n_images: int = 3) -> None:
    manifest = build_corpus(os.path.join(os.path.dirname(__file__), "..", "benchmark", "_corpus_cache"))
    tess, easy = TesseractAdapter(), EasyOCRAdapter()

    print(f"=== {preset}: per-engine confidence vs. CER ===")
    for item in manifest[:n_images]:
        clean = cv2.imread(item["path"])
        degraded = apply_degradation(clean, preset, seed=stable_seed(item["path"], preset))
        report = assess(degraded)
        processed, _ = build_pipeline(degraded, report)
        truth = item["ground_truth"]

        for name, adapter in [("tesseract", tess), ("easyocr", easy)]:
            dets = adapter.recognize(processed)
            text = " ".join(d.text for d in dets)
            conf = sum(d.confidence for d in dets) / len(dets) if dets else 0.0
            print(f"  {name:10s} conf={conf:.3f} cer={cer(text, truth):.3f} text={text!r}")


def weighted_vs_unweighted_fusion(presets: list[str]) -> None:
    manifest = build_corpus(os.path.join(os.path.dirname(__file__), "..", "benchmark", "_corpus_cache"))
    pipeline = OCRPipeline.with_engines(["tesseract", "easyocr"])
    results = {p: {"weighted": [], "unweighted": []} for p in presets}

    for item in manifest:
        clean = cv2.imread(item["path"])
        truth = item["ground_truth"]
        for preset in presets:
            degraded = apply_degradation(clean, preset, seed=stable_seed(item["path"], preset))
            for label, w in [("weighted", True), ("unweighted", False)]:
                r = pipeline.run(degraded, skip_preprocessing=True, force_ensemble=True, fusion_weighted=w)
                results[preset][label].append(cer(r.raw_text, truth))

    print(f"\n{'preset':15s} {'weighted':>10s} {'unweighted':>12s}")
    for p in presets:
        w = sum(results[p]["weighted"]) / len(results[p]["weighted"])
        u = sum(results[p]["unweighted"]) / len(results[p]["unweighted"])
        print(f"{p:15s} {w:10.4f} {u:12.4f}")


if __name__ == "__main__":
    per_engine_confidence_vs_accuracy("combo_hard")
    weighted_vs_unweighted_fusion(["clean", "heavy_blur", "skewed", "combo_hard", "motion_blur", "noisy"])
