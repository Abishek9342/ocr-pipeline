"""Ablation study: which components of the pipeline actually earn their
place? Every row here is a real measured run against the same synthetic,
labeled corpus + degradation presets used by run_benchmark.py — nothing is
asserted without a number attached (see run_benchmark.py's module
docstring for the same philosophy). This exists because run_benchmark.py
only ever compares fully-assembled systems (tesseract_alone vs. pipeline);
it never isolates which individual piece of "pipeline" is responsible for
the difference.

Each variant reuses `OCRPipeline.run()`'s ablation hooks
(`skip_preprocessing`, `force_step`, `force_ensemble` — see pipeline.py)
rather than duplicating pipeline logic here.

Run: python -m benchmark.run_ablation
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from benchmark.corpus import build_corpus
from benchmark.degrade import apply_degradation
from benchmark.run_benchmark import _git_commit, stable_seed
from ocr_resilience import __version__ as pipeline_version
from ocr_resilience.metrics import cer, wer
from ocr_resilience.pipeline import OCRPipeline

PRESETS_TO_RUN = ["clean", "heavy_blur", "skewed", "noisy", "combo_hard"]

# (label, kwargs passed to OCRPipeline.run) — see pipeline.py's run() docstring
# for what each ablation hook isolates.
VARIANTS = [
    ("baseline", dict(skip_preprocessing=True, force_ensemble=False)),
    ("baseline+deskew", dict(force_step="deskew", force_ensemble=False)),
    ("baseline+denoise", dict(force_step="denoise", force_ensemble=False)),
    ("baseline+clahe", dict(force_step="enhance_contrast", force_ensemble=False)),
    ("baseline+adaptive_threshold(sauvola)", dict(force_step="binarize_sauvola", force_ensemble=False)),
    ("baseline+adaptive_preprocessing", dict(skip_preprocessing=False, force_ensemble=False)),
    ("baseline+multi_engine_selection", dict(skip_preprocessing=True, force_ensemble=True)),
    ("full_pipeline(adaptive+multi_engine)", dict(skip_preprocessing=False, force_ensemble=None)),
]


def run() -> list[dict]:
    corpus_dir = os.path.join(os.path.dirname(__file__), "_corpus_cache")
    manifest = build_corpus(corpus_dir)

    print("Loading engines (one-time cost)...")
    pipeline = OCRPipeline.with_engines(["tesseract", "easyocr"])

    rows = []
    total = len(manifest) * len(PRESETS_TO_RUN) * len(VARIANTS)
    done = 0
    for item in manifest:
        clean_img = cv2.imread(item["path"])
        for preset in PRESETS_TO_RUN:
            degraded = apply_degradation(clean_img, preset, seed=stable_seed(item["path"], preset))
            truth = item["ground_truth"]

            for label, kwargs in VARIANTS:
                done += 1
                t0 = time.perf_counter()
                result = pipeline.run(degraded, **kwargs)
                elapsed = time.perf_counter() - t0

                rows.append({
                    "style": item["style"], "preset": preset, "variant": label,
                    "cer_raw": cer(result.raw_text, truth), "cer_processed": cer(result.processed_text, truth),
                    "wer": wer(result.raw_text, truth), "latency_sec": elapsed,
                })
                print(f"  [{done}/{total}] {item['style']:20s} {preset:12s} {label:38s} done", end="\r")

    print()
    _summarize(rows)
    return rows


def _summarize(rows: list[dict]) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "ablation_raw.csv"), index=False)

    print("\n=== Ablation: mean CER (raw) / WER / latency by variant, in the order defined above ===")
    order = [label for label, _ in VARIANTS]
    summary = df.groupby("variant")[["cer_raw", "wer", "latency_sec"]].mean().round(4).reindex(order)
    summary.to_csv(os.path.join(out_dir, "ablation_summary.csv"))
    print(summary.to_string())

    print("\n=== Post-processing effect: mean CER before vs. after text post-processing (full pipeline only) ===")
    full = df[df["variant"] == "full_pipeline(adaptive+multi_engine)"]
    print(full[["cer_raw", "cer_processed"]].mean().round(4).to_string())

    # Sidecar metadata (mission section 16): records the commit/version this
    # ablation was generated at, so a future staleness check (e.g. "does this
    # CSV predate the router change it's supposed to be measuring?") can be
    # mechanical instead of a manual timestamp comparison — the exact gap that
    # let ablation_raw.csv go stale earlier in this same overnight pass.
    with open(os.path.join(out_dir, "ablation_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"pipeline_version": pipeline_version, "git_commit": _git_commit()}, f, indent=2)

    print(f"\nFull ablation results written to {out_dir}")


if __name__ == "__main__":
    run()
