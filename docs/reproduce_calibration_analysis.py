"""Reproduces docs/confidence_calibration_report.md's raw-vs-calibrated
fusion comparison and the ECE table. Requires benchmark/results/raw_results.csv
to already exist (run benchmark/run_benchmark.py first).

Run from the repo root: python docs/reproduce_calibration_analysis.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pandas as pd

from benchmark.corpus import build_corpus
from benchmark.degrade import apply_degradation
from benchmark.run_benchmark import stable_seed
from ocr_resilience.calibration import apply_calibration, expected_calibration_error, fit_binned_calibrator
from ocr_resilience.engines import EasyOCRAdapter, TesseractAdapter
from ocr_resilience.fusion import fuse
from ocr_resilience.metrics import cer
from ocr_resilience.preprocess import build_pipeline
from ocr_resilience.quality import assess

RAW_RESULTS = os.path.join(os.path.dirname(__file__), "..", "benchmark", "results", "raw_results.csv")
PRESETS = ["clean", "heavy_blur", "skewed", "combo_hard", "motion_blur", "noisy"]


def print_ece_table() -> dict:
    df = pd.read_csv(RAW_RESULTS)
    baseline = df[df["system"] != "ours"]
    calibrators = {}
    print("=== Expected Calibration Error per engine ===")
    for engine in sorted(baseline["system"].unique()):
        sub = baseline[baseline["system"] == engine]
        ece = expected_calibration_error(sub["mean_confidence"].tolist(), sub["cer"].tolist(), n_bins=10)
        print(f"{engine:12s} ECE={ece:.4f}  mean_confidence={sub['mean_confidence'].mean():.3f}  "
              f"mean_correctness_proxy={(1 - sub['cer'].clip(upper=1)).mean():.3f}")
        if engine in ("tesseract", "easyocr"):
            calibrators[engine] = fit_binned_calibrator(sub["mean_confidence"].tolist(), sub["cer"].tolist(), n_bins=10)
    return calibrators


def raw_vs_calibrated_fusion(calibrators: dict) -> None:
    manifest = build_corpus(os.path.join(os.path.dirname(__file__), "..", "benchmark", "_corpus_cache"))
    tess, easy = TesseractAdapter(), EasyOCRAdapter()
    results = {p: {"raw_weighted": [], "calibrated_weighted": []} for p in PRESETS}

    for item in manifest:
        clean = cv2.imread(item["path"])
        truth = item["ground_truth"]
        for preset in PRESETS:
            degraded = apply_degradation(clean, preset, seed=stable_seed(item["path"], preset))
            report = assess(degraded)
            processed, _ = build_pipeline(degraded, report)
            dets = tess.recognize(processed) + easy.recognize(processed)

            raw_text = " ".join(d.text for d in fuse(dets, weighted=True))
            cal_text = " ".join(d.text for d in fuse(apply_calibration(dets, calibrators), weighted=True))

            results[preset]["raw_weighted"].append(cer(raw_text, truth))
            results[preset]["calibrated_weighted"].append(cer(cal_text, truth))

    print(f"\n{'preset':15s} {'raw_weighted':>14s} {'calibrated_weighted':>20s}")
    for p in PRESETS:
        r = sum(results[p]["raw_weighted"]) / len(results[p]["raw_weighted"])
        c = sum(results[p]["calibrated_weighted"]) / len(results[p]["calibrated_weighted"])
        print(f"{p:15s} {r:14.4f} {c:20.4f}")


if __name__ == "__main__":
    calibrators = print_ece_table()
    raw_vs_calibrated_fusion(calibrators)
