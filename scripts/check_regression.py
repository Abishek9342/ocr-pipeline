"""CI regression gate (mission section 26): compare a fresh benchmark run
against a stored baseline and fail if accuracy/latency regressed beyond an
explicit threshold. Deliberately separate from `run_benchmark.py` itself —
this script only COMPARES an already-produced `benchmark.json` against a
baseline; it doesn't run the benchmark (the full 11-preset x 5-engine
sweep takes minutes, too slow for every push — see the CI workflow note
below for how a fast subset is meant to feed this).

Usage:
    python -m benchmark.run_benchmark --engines tesseract,ours --presets clean,heavy_blur,skewed --out /tmp/ci_bench
    python scripts/check_regression.py --current /tmp/ci_bench/benchmark.json \\
        --baseline benchmark/results/baseline_summary.json \\
        --max-cer-increase 0.02 --max-latency-increase-pct 25

Exits 1 (and prints exactly what regressed) if any system in --current
that also exists in --baseline exceeds either threshold. A system present
in --current but not in --baseline is reported, not failed (new engines
have no baseline yet). Update the baseline deliberately
(`--write-baseline`) after a real, reviewed accuracy/latency change —
never silently, since a baseline that auto-updates on every run can never
catch a regression.
"""
from __future__ import annotations

import argparse
import json
import sys


def load_summary(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["summary"] if "summary" in data else data  # benchmark.json shape, or a plain {system: {...}} baseline
    if isinstance(rows, list):
        rows = {row["system"]: row for row in rows}
    return rows


def check(current: dict[str, dict], baseline: dict[str, dict], max_cer_increase: float, max_latency_increase_pct: float) -> list[str]:
    failures = []
    for system, base in baseline.items():
        if system not in current:
            failures.append(
                f"'{system}' is in the baseline but missing from the current run entirely — "
                "treat as a regression until explained"
            )
            continue
        cur = current[system]

        cer_delta = cur["mean_cer"] - base["mean_cer"]
        if cer_delta > max_cer_increase:
            failures.append(
                f"'{system}' mean_cer regressed: {base['mean_cer']:.4f} -> {cur['mean_cer']:.4f} "
                f"(+{cer_delta:.4f}, max allowed +{max_cer_increase})"
            )

        if base["mean_latency_sec"] > 0:
            latency_pct = (cur["mean_latency_sec"] - base["mean_latency_sec"]) / base["mean_latency_sec"] * 100
            if latency_pct > max_latency_increase_pct:
                failures.append(
                    f"'{system}' mean_latency_sec regressed: {base['mean_latency_sec']:.3f}s -> "
                    f"{cur['mean_latency_sec']:.3f}s (+{latency_pct:.1f}%, max allowed +{max_latency_increase_pct}%)"
                )

    for system in current:
        if system not in baseline:
            print(f"[info] '{system}' has no baseline yet — not checked, not failed", file=sys.stderr)

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", required=True, help="Path to a fresh benchmark.json")
    parser.add_argument("--baseline", required=True, help="Path to the stored baseline (same shape, or benchmark.json)")
    parser.add_argument("--max-cer-increase", type=float, default=0.02, help="Absolute CER regression allowed (default 0.02)")
    parser.add_argument("--max-latency-increase-pct", type=float, default=25.0, help="Percent latency regression allowed (default 25%%)")
    parser.add_argument("--write-baseline", action="store_true", help="Overwrite --baseline with --current's summary instead of checking")
    args = parser.parse_args(argv)

    current = load_summary(args.current)

    if args.write_baseline:
        with open(args.baseline, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        print(f"Baseline written to {args.baseline} ({len(current)} systems). This is a deliberate action — commit it with a reason.")
        return 0

    baseline = load_summary(args.baseline)
    failures = check(current, baseline, args.max_cer_increase, args.max_latency_increase_pct)

    if failures:
        print("REGRESSION DETECTED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        f"No regression beyond thresholds (max_cer_increase={args.max_cer_increase}, "
        f"max_latency_increase_pct={args.max_latency_increase_pct}%)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
