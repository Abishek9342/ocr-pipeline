"""Paired bootstrap comparison of `ours` against each baseline engine,
matched by (image_id, preset) — mission section 4. `raw_results.csv`
already has 220 rows per system, one row per (image_id, preset) case, so
every system can be aligned on that key. This is the correct method for
"is `ours` really better than PaddleOCR": naive independent-sample
comparison would treat the two systems' 220 rows as unrelated, throwing
away the fact that they were measured on the exact same 220 cases and so
share case-to-case variance that a paired comparison cancels out.

Requires benchmark/results/raw_results.csv to already exist (run
benchmark/run_benchmark.py first).

Run from the repo root: python -m benchmark.run_paired_comparison
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from benchmark.stats_utils import pairwise_comparison_summary

RAW_RESULTS = os.path.join(os.path.dirname(__file__), "results", "raw_results.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "results", "paired_comparison.csv")
BASELINES = ["tesseract", "easyocr", "paddleocr", "rapidocr"]


def main() -> pd.DataFrame:
    df = pd.read_csv(RAW_RESULTS)
    pivot = df.pivot_table(index=["image_id", "preset"], columns="system", values="cer")

    missing = pivot[["ours", *BASELINES]].isna().any(axis=1).sum()
    if missing:
        print(f"WARNING: dropping {missing} of {len(pivot)} cases missing a CER for at least one system")
        pivot = pivot.dropna(subset=["ours", *BASELINES])

    ours = pivot["ours"].tolist()
    others = {name: pivot[name].tolist() for name in BASELINES}

    # pairwise_comparison_summary(baseline_name, baseline_values, others) computes
    # diff = mean(other) - mean(baseline) for each entry in `others`. Passing
    # "ours" as the statistical baseline means diff = baseline_engine_cer -
    # ours_cer, so POSITIVE = that engine's CER is HIGHER than ours (ours wins).
    rows = pairwise_comparison_summary("ours", ours, others, n_resamples=2000, seed=0)
    out = pd.DataFrame(rows).rename(columns={"system": "baseline_engine", "mean_difference": "engine_minus_ours_cer"})
    out = out[["baseline_engine", "engine_minus_ours_cer", "ci_lower", "ci_upper",
               "ci_level", "distinguishable_from_zero", "cohens_d", "n"]]
    out.to_csv(OUT_PATH, index=False)

    print(f"Paired bootstrap: (baseline engine CER) - (ours CER), n={len(pivot)} matched cases each\n")
    print(out.to_string(index=False))
    print(f"\nWritten to {OUT_PATH}")
    print("\nPositive engine_minus_ours_cer = that engine's CER is HIGHER than ours (ours wins on this case set).")
    return out


if __name__ == "__main__":
    main()
