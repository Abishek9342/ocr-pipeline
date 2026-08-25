"""Phase 6 (condition x engine strength) and Phase 13 (routing regret)
analysis. Reuses the exact same deterministic corpus/degradation pipeline
as `run_benchmark.py` so the `QualityReport` recomputed here for each
(image, preset) is identical to what the router actually saw — no new
randomness introduced, no re-running of OCR engines needed (this reads
`raw_results.csv`, it doesn't regenerate it).

Run: python -m benchmark.analyze_routing [--raw benchmark/results/raw_results.csv]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pandas as pd

from benchmark.corpus import build_corpus
from benchmark.degrade import apply_degradation
from benchmark.run_benchmark import stable_seed
from ocr_resilience.engine_selection import select_primary_engine
from ocr_resilience.quality import assess


def condition_engine_table(df: pd.DataFrame, baseline_systems: list[str]) -> pd.DataFrame:
    """Phase 6: for every preset, which baseline engine has the lowest
    mean CER, which is second, and which has the lowest mean latency."""
    baseline_df = df[df["system"].isin(baseline_systems)]
    cer_pivot = baseline_df.groupby(["preset", "system"])["cer"].mean().unstack()
    latency_pivot = baseline_df.groupby(["preset", "system"])["latency_sec"].mean().unstack()

    rows = []
    for preset in cer_pivot.index:
        ranked = cer_pivot.loc[preset].sort_values()
        rows.append({
            "preset": preset,
            "best_engine": ranked.index[0], "best_cer": ranked.iloc[0],
            "second_best_engine": ranked.index[1] if len(ranked) > 1 else None,
            "second_best_cer": ranked.iloc[1] if len(ranked) > 1 else None,
            "fastest_engine": latency_pivot.loc[preset].idxmin(),
            "fastest_latency_sec": latency_pivot.loc[preset].min(),
        })
    return pd.DataFrame(rows).set_index("preset")


def routing_regret(df: pd.DataFrame, baseline_systems: list[str], corpus_dir: str) -> pd.DataFrame:
    """Phase 13: for every (image, preset), what would the OLD router
    (registration-order pick, i.e. always `baseline_systems[0]`) and the
    NEW router (`select_primary_engine`, quality-aware) have selected as
    the single "easy path" engine, versus the actual best-performing
    available engine on that exact case? Regret = selected_engine_CER -
    best_available_engine_CER (0 = picked the best one)."""
    manifest = build_corpus(corpus_dir)
    path_by_image_id = {os.path.basename(m["path"]): m["path"] for m in manifest}

    rows = []
    for (image_id, preset), group in df.groupby(["image_id", "preset"]):
        baseline_rows = group[group["system"].isin(baseline_systems)]
        if baseline_rows.empty or image_id not in path_by_image_id:
            continue

        best_row = baseline_rows.loc[baseline_rows["cer"].idxmin()]
        old_selected = baseline_systems[0]  # the OLD router's registration-order behavior

        clean_img = cv2.imread(path_by_image_id[image_id])
        degraded = apply_degradation(clean_img, preset, seed=stable_seed(path_by_image_id[image_id], preset))
        report = assess(degraded)
        new_selected, new_reason = select_primary_engine(report, baseline_systems)

        def cer_for(engine: str):
            match = baseline_rows[baseline_rows["system"] == engine]
            return match["cer"].iloc[0] if not match.empty else None

        old_cer, new_cer, best_cer = cer_for(old_selected), cer_for(new_selected), best_row["cer"]
        # "top1" = did the selected engine ACHIEVE the best available CER on
        # this exact case — not "does its NAME match whichever engine
        # pandas' idxmin() happened to return first among ties." Ties at
        # CER=0.0 are common on easy images, and idxmin() always resolves
        # them to whichever engine appears first in raw_results.csv's row
        # order (tesseract, by construction) — comparing engine NAMES
        # instead of CER VALUES would silently and unfairly inflate
        # whichever engine happens to run first's apparent top-1 rate.
        rows.append({
            "image_id": image_id, "preset": preset,
            "best_engine": best_row["system"], "best_cer": best_cer,
            "old_router_selected": old_selected, "old_router_cer": old_cer,
            "old_router_regret": (old_cer - best_cer) if old_cer is not None else None,
            "old_router_top1": (old_cer is not None) and (old_cer <= best_cer + 1e-9),
            "new_router_selected": new_selected, "new_router_cer": new_cer,
            "new_router_regret": (new_cer - best_cer) if new_cer is not None else None,
            "new_router_top1": (new_cer is not None) and (new_cer <= best_cer + 1e-9),
            "new_router_reason": new_reason,
        })
    return pd.DataFrame(rows)


def summarize_regret(regret_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "old_router": {
            "top1_accuracy": regret_df["old_router_top1"].mean(),
            "mean_regret": regret_df["old_router_regret"].mean(),
            "mean_selected_cer": regret_df["old_router_cer"].mean(),
        },
        "new_router": {
            "top1_accuracy": regret_df["new_router_top1"].mean(),
            "mean_regret": regret_df["new_router_regret"].mean(),
            "mean_selected_cer": regret_df["new_router_cer"].mean(),
        },
    }).T


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=os.path.join(os.path.dirname(__file__), "results", "raw_results.csv"))
    parser.add_argument("--baseline-systems", default="tesseract,easyocr,paddleocr,rapidocr")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    baseline_systems = [s.strip() for s in args.baseline_systems.split(",") if s.strip()]
    df = pd.read_csv(args.raw)
    corpus_dir = os.path.join(os.path.dirname(__file__), "_corpus_cache")

    print(f"=== Condition x engine strength (Phase 6), baselines={baseline_systems} ===")
    table = condition_engine_table(df, baseline_systems)
    print(table.round(4).to_string())
    table.to_csv(os.path.join(args.out_dir, "condition_engine_table.csv"))

    print("\n=== Routing regret: OLD (registration-order) vs. NEW (quality-aware) router (Phase 13) ===")
    regret_df = routing_regret(df, baseline_systems, corpus_dir)
    regret_df.to_csv(os.path.join(args.out_dir, "routing_regret_raw.csv"), index=False)
    summary = summarize_regret(regret_df)
    print(summary.round(4).to_string())
    summary.to_csv(os.path.join(args.out_dir, "routing_regret_summary.csv"))

    print(f"\nWritten to {args.out_dir}: condition_engine_table.csv, routing_regret_raw.csv, routing_regret_summary.csv")


if __name__ == "__main__":
    main()
