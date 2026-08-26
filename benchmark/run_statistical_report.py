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
from ocr_resilience.engines import AVAILABLE_ENGINES
from ocr_resilience.pipeline import OCRPipeline

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


def _degraded_test_image():
    manifest = build_corpus(os.path.join(os.path.dirname(__file__), "_corpus_cache"))
    item = manifest[0]
    clean = cv2.imread(item["path"])
    return apply_degradation(clean, "clean", seed=stable_seed(item["path"], "clean"))


def repeated_latency_measurement(n_repeats: int = N_REPEATS) -> dict:
    """Runs every available single engine, plus the full `ours` pipeline,
    on the SAME degraded image `n_repeats` times in a row and reports the
    variance — isolates measurement noise from image-to-image variance,
    which the main benchmark's median/P95 columns already cover but don't
    distinguish from this. The engine object is constructed once before
    timing starts (one-time model-load cost paid outside the loop), but
    the FIRST `recognize()`/`run()` call is still reported separately from
    the rest ("cold" vs. "warm") since inference-time lazy initialization
    (thread pools, ONNX/Paddle session warmup, first-call JIT paths) is a
    real, distinct cost from steady-state inference — collapsing the two
    would hide a real cold-start effect inside the "noise.\""""
    degraded = _degraded_test_image()
    results = {}

    for name, adapter_cls in sorted(AVAILABLE_ENGINES.items()):
        adapter = adapter_cls()
        latencies = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            adapter.recognize(degraded)
            latencies.append(time.perf_counter() - t0)
        results[name] = {
            "cold_start_sec": latencies[0],
            "warm": latency_variance_stats(latencies[1:]) if len(latencies) > 1 else None,
            "all_repeats": latency_variance_stats(latencies),
        }

    pipeline = OCRPipeline.with_engines(list(AVAILABLE_ENGINES))
    latencies = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        pipeline.run(degraded)
        latencies.append(time.perf_counter() - t0)
    results["ours"] = {
        "cold_start_sec": latencies[0],
        "warm": latency_variance_stats(latencies[1:]) if len(latencies) > 1 else None,
        "all_repeats": latency_variance_stats(latencies),
    }

    return results


def main() -> None:
    print("=== Bootstrap 95% CI on mean CER per system (n_resamples=2000) ===")
    ci_table = cer_confidence_intervals()
    print(ci_table.round(4).to_string())

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    ci_table.to_csv(os.path.join(out_dir, "cer_confidence_intervals.csv"))

    print(f"\n=== Repeated-run latency variance: every engine + ours, same image x {N_REPEATS} ===")
    stats = repeated_latency_measurement()
    flat_rows = []
    for system, s in stats.items():
        warm = s["warm"] or s["all_repeats"]
        print(f"  {system}: cold_start={s['cold_start_sec']:.4f}s  "
              f"warm_mean={warm['mean']:.4f}s  warm_std={warm['std']:.4f}s  "
              f"warm_cv={warm['coefficient_of_variation']:.4f}  warm_median={warm['median']:.4f}s  "
              f"warm_p95={warm['p95']:.4f}s")
        flat_rows.append({
            "system": system, "cold_start_sec": s["cold_start_sec"],
            **{f"warm_{k}": v for k, v in warm.items()},
        })

    pd.DataFrame(flat_rows).to_csv(os.path.join(out_dir, "latency_variance.csv"), index=False)
    print(f"\nWritten to {out_dir}: cer_confidence_intervals.csv, latency_variance.csv")


if __name__ == "__main__":
    main()
