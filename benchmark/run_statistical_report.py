"""Statistical rigor report: bootstrap confidence intervals on each
system's mean CER (from the existing benchmark data — no new OCR calls
needed), plus a real repeated-latency-run measurement (the SAME image
run N times, to characterize run-to-run timing noise directly rather than
just asserting it exists).

Run: python -m benchmark.run_statistical_report
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pandas as pd

from benchmark.corpus import build_corpus
from benchmark.degrade import apply_degradation
from benchmark.run_benchmark import stable_seed
from benchmark.stats_utils import bootstrap_ci, latency_variance_stats
from ocr_resilience.engines import TesseractAdapter

RAW_RESULTS = os.path.join(os.path.dirname(__file__), "results", "raw_results.csv")
N_REPEATS = 10


def cer_confidence_intervals() -> pd.DataFrame:
    df = pd.read_csv(RAW_RESULTS)
    rows = []
    for system in sorted(df["system"].unique()):
        cers = df[df["system"] == system]["cer"].tolist()
        ci = bootstrap_ci(cers, n_resamples=2000, seed=0)
        rows.append({"system": system, **ci})
    return pd.DataFrame(rows).set_index("system")


def repeated_latency_measurement(n_repeats: int = N_REPEATS) -> dict:
    """Runs Tesseract on the SAME degraded image `n_repeats` times in a
    row and reports the variance — isolates measurement noise from
    image-to-image variance, which the main benchmark's median/P95
    columns already cover but don't distinguish from this."""
    manifest = build_corpus(os.path.join(os.path.dirname(__file__), "_corpus_cache"))
    item = manifest[0]
    clean = cv2.imread(item["path"])
    degraded = apply_degradation(clean, "clean", seed=stable_seed(item["path"], "clean"))

    tess = TesseractAdapter()
    latencies = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        tess.recognize(degraded)
        latencies.append(time.perf_counter() - t0)

    return latency_variance_stats(latencies)


def main() -> None:
    print("=== Bootstrap 95% CI on mean CER per system (n_resamples=2000) ===")
    ci_table = cer_confidence_intervals()
    print(ci_table.round(4).to_string())

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    ci_table.to_csv(os.path.join(out_dir, "cer_confidence_intervals.csv"))

    print(f"\n=== Repeated-run latency variance: Tesseract, same image x {N_REPEATS} ===")
    stats = repeated_latency_measurement()
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    pd.DataFrame([stats]).to_csv(os.path.join(out_dir, "latency_variance.csv"), index=False)
    print(f"\nWritten to {out_dir}: cer_confidence_intervals.csv, latency_variance.csv")


if __name__ == "__main__":
    main()
