"""Benchmark: this pipeline vs. calling each OCR engine directly, with no
preprocessing/routing/fusion — across a labeled, synthetically-degraded
corpus with known ground truth. This is the artifact that actually
substantiates (or disproves) any accuracy/speed claim; nothing here is
asserted without a number attached.

Run: python -m benchmark.run_benchmark
  or with explicit options:
    python -m benchmark.run_benchmark --engines tesseract,easyocr,paddleocr,ours --presets all
"""
import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from benchmark.corpus import build_corpus
from benchmark.degrade import DEGRADATION_PRESETS, apply_degradation
from benchmark.failure_taxonomy import classify_failure
from ocr_resilience.engines import AVAILABLE_ENGINES
from ocr_resilience.metrics import cer, wer
from ocr_resilience.pipeline import OCRPipeline
from ocr_resilience.quality import assess

# This corpus is entirely English/Latin-script (see corpus.py) — these are
# recorded per-row, honestly, as facts about THIS corpus, not a claim that
# language/script detection or multilingual evaluation is implemented.
# Section 12's point is that the row schema below can already carry a real
# dataset's language/script metadata unchanged once one exists.
CORPUS_LANGUAGE = "en"
CORPUS_SCRIPT = "Latin"


def _git_commit() -> str:
    """Best-effort short commit hash for artifact versioning (mission
    section 16) — this session found multiple STALE generated-artifact
    bugs (ablation CSVs predating a router change) that a human only
    caught by comparing file timestamps by hand. Recording the commit a
    result was generated at makes that check mechanical: a benchmark.json
    whose commit isn't HEAD is stale by construction, not by inspection.
    Returns 'unknown' rather than raising if git isn't available (e.g. a
    tarball install with no .git directory) — this is diagnostic metadata,
    not something worth failing a benchmark run over."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"

ALL_PRESETS = list(DEGRADATION_PRESETS)
DEFAULT_ENGINES = ["tesseract", "easyocr", "ours"]


def stable_seed(*parts: str) -> int:
    """Deterministic seed derived from a stable hash (sha256), not Python's
    built-in `hash()` — `hash()` of a string is salted per-process by
    default (`PYTHONHASHSEED` randomization), so the same (path, preset)
    pair produced a DIFFERENT degraded image on every run/machine even
    with `seed=` passed through. A real bug: this benchmark's own
    reproducibility claim depended on it, undetected until traced here."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _load_systems(engine_names: list[str]) -> dict:
    """Build the requested comparison systems. 'ours' is this package's
    adaptive pipeline (all successfully-loaded engines registered with
    it); every other name is that engine called directly with NO
    preprocessing/routing/fusion, i.e. the baseline. An engine that fails
    to load (missing binary/model, or — as documented in the README — a
    genuine upstream bug in a given PaddleOCR/PaddlePaddle build) is
    reported and skipped rather than aborting the whole run; which
    engines actually ran is written into benchmark.json so a reader can
    see exactly what a given result set does and doesn't cover."""
    loaded = {}
    skipped = {}
    for name in engine_names:
        if name == "ours":
            continue
        try:
            loaded[name] = AVAILABLE_ENGINES[name]()
        except Exception as exc:  # noqa: BLE001 - engine load failures are data, not a crash
            skipped[name] = str(exc)
    pipeline = None
    if "ours" in engine_names and loaded:
        pipeline = OCRPipeline(engines=dict(loaded))
    return loaded, pipeline, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocr-benchmark", description="Benchmark this pipeline against OCR engine baselines.")
    parser.add_argument(
        "--dataset", default=None,
        help="Directory to cache/read the corpus from (default: benchmark/_corpus_cache, synthetic).",
    )
    parser.add_argument(
        "--engines", default=",".join(DEFAULT_ENGINES),
        help=f"Comma-separated systems to compare, from {{{','.join(AVAILABLE_ENGINES)},ours}} (default: {','.join(DEFAULT_ENGINES)}).",
    )
    parser.add_argument(
        "--presets", default="all",
        help=f"Comma-separated degradation presets, or 'all' (default) for all {len(ALL_PRESETS)}: {','.join(ALL_PRESETS)}.",
    )
    parser.add_argument("--out", default=None, help="Output directory (default: benchmark/results).")
    return parser


def run(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]
    presets = ALL_PRESETS if args.presets == "all" else [p.strip() for p in args.presets.split(",") if p.strip()]
    unknown = set(engine_names) - set(AVAILABLE_ENGINES) - {"ours"}
    if unknown:
        raise ValueError(f"Unknown engine(s) {sorted(unknown)}. Available: {list(AVAILABLE_ENGINES)}, ours")

    corpus_dir = args.dataset or os.path.join(os.path.dirname(__file__), "_corpus_cache")
    manifest = build_corpus(corpus_dir)

    print("Loading engines (one-time cost)...")
    loaded, pipeline, skipped = _load_systems(engine_names)
    if skipped:
        for name, error in skipped.items():
            print(f"  [skipped] {name}: {error}", file=sys.stderr)
    systems = [name for name in engine_names if (name == "ours" and pipeline is not None) or name in loaded]
    if not systems:
        raise RuntimeError("No systems could be loaded — nothing to benchmark.")

    rows, latency_rows = [], []
    total = len(manifest) * len(presets)
    done = 0
    for item in manifest:
        clean_img = cv2.imread(item["path"])
        for preset in presets:
            done += 1
            degraded = apply_degradation(clean_img, preset, seed=stable_seed(item["path"], preset))
            truth = item["ground_truth"]
            # Computed once per (image, preset), reused by every system's row
            # below — mission section 9's "image_features" column, logged at
            # collection time rather than re-derived later from a re-run.
            image_features_json = json.dumps(dataclasses.asdict(assess(degraded)))

            for system in list(systems):  # copy: a system can be dropped mid-loop below
                tracemalloc.start()
                t0 = time.perf_counter()
                try:
                    if system == "ours":
                        result = pipeline.run(degraded)
                        text = result.raw_text
                        stage_timings = dict(result.timing_sec)
                        confidence = result.confidence
                        n_detections = len(result.detections)
                        routing_reason = result.routing.reason
                        engine_used = result.engine_used
                    else:
                        detections = loaded[system].recognize(degraded)
                        text = " ".join(d.text for d in detections)
                        stage_timings = {}
                        confidence = sum(d.confidence for d in detections) / len(detections) if detections else 0.0
                        n_detections = len(detections)
                        routing_reason = ""
                        engine_used = system
                except Exception as exc:  # noqa: BLE001 - a runtime failure mid-benchmark degrades gracefully, not a crash
                    tracemalloc.stop()
                    print(f"\n  [dropping {system}] failed on {item['path']}/{preset}: {exc}", file=sys.stderr)
                    skipped[system] = str(exc)
                    systems.remove(system)
                    continue
                elapsed = time.perf_counter() - t0
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                cer_value, wer_value = cer(text, truth), wer(text, truth)
                rows.append({
                    "image_id": os.path.basename(item["path"]), "style": item["style"],
                    "preset": preset, "system": system,
                    "cer": cer_value, "wer": wer_value,
                    "latency_sec": elapsed, "peak_memory_bytes": peak_bytes,
                    "mean_confidence": confidence, "n_detections": n_detections,
                    "engine_used": engine_used, "routing_reason": routing_reason,
                    "catastrophic_failure": cer_value >= 0.9,
                    "language": CORPUS_LANGUAGE, "script": CORPUS_SCRIPT,
                    "image_features": image_features_json,
                    # No fitted calibrator is wired into this benchmark run (see
                    # docs/confidence_calibration_report.md's keep/reject decision —
                    # calibration showed no statistically detectable benefit here).
                    # Left as None rather than silently populated with an
                    # uncalibrated value under a misleading column name.
                    "calibrated_confidence": None,
                    "failure_type": classify_failure(text, truth, cer_value).value,
                })
                stage_timings["total"] = elapsed
                for stage, seconds in stage_timings.items():
                    latency_rows.append({"style": item["style"], "preset": preset, "system": system, "stage": stage, "seconds": seconds})

            print(f"  [{done}/{total}] {item['style']:20s} {preset:15s} done", end="\r")

    print()
    # A system dropped mid-run leaves partial, presumably-easier-cases-only
    # rows behind (it only failed once it hit whatever image tripped it) —
    # an average over that partial set would misleadingly look like a real
    # score. Drop its rows entirely; `skipped` still records why.
    rows = [r for r in rows if r["system"] not in skipped]
    latency_rows = [r for r in latency_rows if r["system"] not in skipped]
    _write_outputs(rows, latency_rows, systems, skipped, presets, args.out)


def _write_outputs(rows, latency_rows, systems, skipped, presets, out_arg) -> None:
    import pandas as pd

    out_dir = out_arg or os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)
    pd.DataFrame(latency_rows).to_csv(os.path.join(out_dir, "latency.csv"), index=False)

    def _p95(s):
        return s.quantile(0.95)

    summary = df.groupby("system").agg(
        mean_cer=("cer", "mean"), mean_wer=("wer", "mean"),
        mean_latency_sec=("latency_sec", "mean"), median_latency_sec=("latency_sec", "median"),
        p95_latency_sec=("latency_sec", _p95), mean_peak_memory_bytes=("peak_memory_bytes", "mean"),
        mean_confidence=("mean_confidence", "mean"), catastrophic_failure_rate=("catastrophic_failure", "mean"),
    ).round(6)
    summary.to_csv(os.path.join(out_dir, "summary.csv"))

    print("\n=== Overall: mean CER / WER / latency / P95 latency / memory by system ===")
    print(summary.to_string())

    print("\n=== Mean CER by system x degradation preset ===")
    print(df.pivot_table(index="preset", columns="system", values="cer", aggfunc="mean").round(4).to_string())

    print("\n=== Mean CER by system x text style ===")
    print(df.pivot_table(index="style", columns="system", values="cer", aggfunc="mean").round(4).to_string())

    from ocr_resilience import __version__ as pipeline_version

    manifest_json = {
        "config": {
            "systems": systems, "systems_skipped": skipped, "presets": presets,
            "python_version": platform.python_version(), "platform": platform.platform(),
            "pipeline_version": pipeline_version, "git_commit": _git_commit(),
        },
        "summary": json.loads(summary.reset_index().to_json(orient="records")),
    }
    with open(os.path.join(out_dir, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_json, f, indent=2)

    print(f"\nFull results written to {out_dir} (raw_results.csv, summary.csv, latency.csv, benchmark.json)")


if __name__ == "__main__":
    run()
