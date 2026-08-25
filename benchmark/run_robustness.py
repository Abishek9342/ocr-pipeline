"""Robustness curves (mission section 18): not just one CER number per
degradation TYPE, but CER as a function of SEVERITY within each type —
"how gracefully does performance degrade as the image gets worse?"
Reuses `benchmark/degrade.py`'s parameterized primitives directly
(gaussian_blur(sigma), gaussian_noise(sigma), rotate(angle_deg),
jpeg_artifact(quality)) rather than the fixed-severity DEGRADATION_PRESETS
dict used by run_benchmark.py — this is a genuinely different axis of
investigation, not a rerun of the same experiment.

Run: python -m benchmark.run_robustness
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from benchmark.corpus import build_corpus
from benchmark.degrade import gaussian_blur, gaussian_noise, jpeg_artifact, rotate
from benchmark.run_benchmark import stable_seed
from ocr_resilience.metrics import cer
from ocr_resilience.pipeline import OCRPipeline

# (corruption_type, [(severity_label, severity_value)]) — severity increases left to right.
# jpeg quality is inverted (lower quality = more severe) so the LABEL axis still reads
# increasing-severity left to right, matching the other three curves.
SEVERITY_SWEEPS = {
    "blur_sigma": [("0.0", 0.0), ("0.5", 0.5), ("1.5", 1.5), ("3.0", 3.0), ("5.0", 5.0)],
    "noise_sigma": [("0", 0), ("10", 10), ("25", 25), ("50", 50), ("80", 80)],
    "skew_deg": [("0", 0), ("2", 2), ("5", 5), ("10", 10), ("20", 20)],
    "jpeg_quality": [("95", 95), ("50", 50), ("25", 25), ("12", 12), ("5", 5)],
}

APPLY = {
    "blur_sigma": lambda img, v, rng: img if v == 0 else gaussian_blur(img, sigma=v),
    "noise_sigma": lambda img, v, rng: img if v == 0 else gaussian_noise(img, sigma=v, rng=rng),
    "skew_deg": lambda img, v, rng: img if v == 0 else rotate(img, angle_deg=v),
    "jpeg_quality": lambda img, v, rng: jpeg_artifact(img, quality=v),
}

N_IMAGES = 5  # kept small deliberately — this sweeps 4 types x 5 severities x N images x 3 systems


def run() -> list[dict]:
    corpus_dir = os.path.join(os.path.dirname(__file__), "_corpus_cache")
    manifest = build_corpus(corpus_dir)[:N_IMAGES]

    print("Loading engines (one-time cost)...")
    pipeline_2engine = OCRPipeline.with_engines(["tesseract", "easyocr"])
    from ocr_resilience.engines import EasyOCRAdapter, TesseractAdapter
    tess, easy = TesseractAdapter(), EasyOCRAdapter()

    rows = []
    total = len(SEVERITY_SWEEPS) * 5 * len(manifest)
    done = 0
    for corruption_type, levels in SEVERITY_SWEEPS.items():
        for label, value in levels:
            for item in manifest:
                done += 1
                clean = cv2.imread(item["path"])
                seed = stable_seed(item["path"], corruption_type, str(value))
                degraded = APPLY[corruption_type](clean, value, random.Random(seed))
                truth = item["ground_truth"]

                for system, text in [
                    ("tesseract", " ".join(d.text for d in tess.recognize(degraded))),
                    ("easyocr", " ".join(d.text for d in easy.recognize(degraded))),
                    ("ours", pipeline_2engine.run(degraded).raw_text),
                ]:
                    rows.append({
                        "corruption_type": corruption_type, "severity_label": label,
                        "severity_value": value, "system": system,
                        "cer": cer(text, truth), "seed": seed,
                    })
                print(f"  [{done}/{total}] {corruption_type:12s} severity={label:6s} done", end="\r")

    print()
    _summarize(rows)
    return rows


def _summarize(rows: list[dict]) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "robustness_raw.csv"), index=False)

    curve = df.groupby(["corruption_type", "severity_label", "system"])["cer"].mean().unstack("system").round(4)
    curve.to_csv(os.path.join(out_dir, "robustness_curves.csv"))

    for corruption_type in SEVERITY_SWEEPS:
        print(f"\n=== Robustness curve: {corruption_type} ===")
        sub = curve.loc[corruption_type]
        # re-order rows by the original severity sequence, not alphabetically
        order = [label for label, _ in SEVERITY_SWEEPS[corruption_type]]
        print(sub.reindex(order).to_string())

    print(f"\nFull robustness data written to {out_dir} (robustness_raw.csv, robustness_curves.csv)")


if __name__ == "__main__":
    run()
