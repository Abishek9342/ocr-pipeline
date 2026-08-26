"""CLI scaffold for evaluating against a real (non-synthetic) dataset
once one exists (mission section 11). This project has NO real dataset
today — outbound network access is sandboxed, so nothing gets
downloaded, and nothing gets fabricated to fill the gap (this project's
own hard limit). What this script provides is the harness that would
consume one immediately: point it at a manifest file (JSON Lines, one
`dataset_schema.DatasetRow`-shaped dict per line), and it validates,
then evaluates, then writes a reproducible, schema-consistent output —
the same row schema (`image_id`, `system`, `cer`, `wer`, `mean_confidence`,
`engine_used`, `routing_reason`, `catastrophic_failure`, plus a
`failure_type` column from `failure_taxonomy.classify_failure` and,
where available, `image_features`/`language`/`script`) that
`run_benchmark.py` already uses for the synthetic corpus, so downstream
analysis code (`analyze_routing.py`, `stats_utils.py`,
`analyze_robustness.py`) works unmodified against real results the
moment they exist.

Usage:
    python -m benchmark.run_real_dataset --manifest path/to/manifest.jsonl [--engines ours,tesseract,...]

A manifest that does not exist yet produces a clear, actionable error,
not a traceback — this script is meant to be run today, against nothing,
to confirm the scaffold itself works (see tests/test_run_real_dataset.py),
and tomorrow, against a real manifest, unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pandas as pd

from benchmark.dataset_validator import validate_manifest, write_validation_report
from benchmark.failure_taxonomy import classify_failure
from ocr_resilience.engines import AVAILABLE_ENGINES
from ocr_resilience.metrics import cer, wer
from ocr_resilience.pipeline import OCRPipeline

DEFAULT_ENGINES = ["ours", *sorted(AVAILABLE_ENGINES)]


def load_manifest(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"manifest not found: {path!r}. This script requires a JSON-Lines manifest "
            f"conforming to benchmark/dataset_schema.py's DatasetRow fields — see that "
            f"module's docstring and benchmark/dataset_validator.py for what's checked."
        )
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"manifest line {line_no} is not valid JSON: {e}") from e
    return rows


def evaluate(rows: list[dict], engine_names: list[str]) -> pd.DataFrame:
    engines_to_load = [e for e in engine_names if e != "ours"]
    if "ours" in engine_names:
        pipeline = OCRPipeline.with_engines(sorted(AVAILABLE_ENGINES))
    single_engines = {name: AVAILABLE_ENGINES[name]() for name in engines_to_load}

    result_rows = []
    for row in rows:
        image = cv2.imread(row["image_path"])
        if image is None:
            result_rows.append({
                "image_id": row["image_id"], "system": "*", "cer": None, "wer": None,
                "failure_type": "engine_error", "error": "cv2.imread returned None (unreadable image)",
            })
            continue
        truth = row["ground_truth_text"]

        for name in engine_names:
            t0 = time.perf_counter()
            error = None
            try:
                if name == "ours":
                    result = pipeline.run(image)
                    text, confidence, engine_used, reason = (
                        result.text, result.confidence, result.engine_used, result.routing.reason,
                    )
                else:
                    detections = single_engines[name].recognize(image)
                    text = " ".join(d.text for d in detections)
                    confidence = (sum(d.confidence for d in detections) / len(detections)) if detections else 0.0
                    engine_used, reason = name, "single_engine_baseline"
            except Exception as exc:  # noqa: BLE001 - any engine failure must degrade to one recorded row, not crash the run
                error, text, confidence, engine_used, reason = str(exc), "", 0.0, name, "engine_error"
            elapsed = time.perf_counter() - t0

            cer_value = None if error else cer(text, truth)
            wer_value = None if error else wer(text, truth)
            failure_type = classify_failure(text, truth, cer_value, engine_raised_exception=bool(error))

            result_rows.append({
                "image_id": row["image_id"], "system": name, "language": row.get("language"),
                "script": row.get("script"), "document_type": row.get("document_type"),
                "split": row.get("split"), "cer": cer_value, "wer": wer_value,
                "mean_confidence": confidence, "engine_used": engine_used, "routing_reason": reason,
                "catastrophic_failure": bool(cer_value is not None and cer_value >= 0.95),
                "failure_type": failure_type.value, "latency_sec": elapsed, "error": error,
            })
    return pd.DataFrame(result_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to a JSON-Lines dataset manifest")
    parser.add_argument("--engines", default=",".join(DEFAULT_ENGINES),
                         help=f"Comma-separated systems to evaluate (default: {','.join(DEFAULT_ENGINES)})")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--skip-image-check", action="store_true",
                         help="Validate the manifest's fields but don't require image files to exist yet")
    args = parser.parse_args(argv)

    rows = load_manifest(args.manifest)
    report = validate_manifest(rows, check_images_exist=not args.skip_image_check)
    os.makedirs(args.output_dir, exist_ok=True)
    write_validation_report(report, os.path.join(args.output_dir, "real_dataset_validation.txt"))
    print(report.summary())
    for issue in report.errors[:20]:
        print(f"  [ERROR] {issue.category} (image_id={issue.row_image_id}): {issue.message}")

    if not report.is_valid:
        print(f"\n{len(report.errors)} manifest error(s) found — aborting evaluation. "
              f"See {args.output_dir}/real_dataset_validation.txt for the full list.")
        return 1

    engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]
    print(f"\nEvaluating {len(rows)} rows against: {engine_names}")
    results = evaluate(rows, engine_names)
    out_path = os.path.join(args.output_dir, "real_dataset_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\nWritten to {out_path}")
    if results["cer"].notna().any():
        print(results.groupby("system")["cer"].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
