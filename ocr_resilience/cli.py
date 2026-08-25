"""Command-line entry point (`ocr-pipeline` / `python -m ocr_resilience`).

    ocr-pipeline document.jpg
    ocr-pipeline document.jpg --engine auto --output result.json
    ocr-pipeline ./scans/ --output results/          # batch: one JSON per input

A single input prints its result as JSON to stdout (or writes it to
--output if given). A directory input processes every image file inside
it; a bad file in a batch is reported and skipped rather than aborting the
whole run, since one corrupt scan shouldn't lose the other 99 results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import OCR

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_inputs(paths: list[str]) -> list[Path]:
    collected: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            collected.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS))
        else:
            collected.append(p)
    return collected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocr-pipeline", description="Adaptive OCR pipeline CLI.")
    parser.add_argument("inputs", nargs="+", help="Image file path(s), or a directory to batch-process.")
    parser.add_argument(
        "--engine", default="auto",
        help="Comma-separated engine names, or 'auto' to use whatever is installed (default: auto).",
    )
    parser.add_argument(
        "--preprocessing", choices=["adaptive", "none"], default="adaptive",
        help="Adaptive (default) or skip preprocessing entirely.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write JSON here. A directory for --output writes one file per input; omit to print to stdout.",
    )
    parser.add_argument("--no-boxes", action="store_true", help="Omit bounding boxes from the JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = _collect_inputs(args.inputs)
    if not inputs:
        print("No image files found for the given input(s).", file=sys.stderr)
        return 1

    try:
        ocr = OCR(engine=args.engine, preprocessing=args.preprocessing, return_boxes=not args.no_boxes)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir = None
    if args.output and (len(inputs) > 1 or Path(args.output).is_dir()):
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for image_path in inputs:
        try:
            result = ocr.predict_dict(str(image_path))
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            print(f"Error processing {image_path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        payload = json.dumps(result, indent=2)
        if output_dir is not None:
            (output_dir / f"{image_path.stem}.json").write_text(payload, encoding="utf-8")
        elif args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            print(payload)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
