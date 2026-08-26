"""Robustness statistics (mission section 6): the existing severity-sweep
curves in `benchmark/results/robustness_raw.csv` (from
`run_robustness.py`) show a lot of information visually but no single
number to compare systems by. This module adds four such numbers PER
(corruption_type, system) — worst-severity CER, degradation slope,
catastrophic-failure onset severity, and area-under-curve — computed
directly from the existing raw rows. It does NOT replace the curves
(deliberately, per the mission's own instruction): a single score always
hides where a cliff sits, which is the entire point `run_robustness.py`
was built to surface in the first place.

Run: python -m benchmark.analyze_robustness
(requires benchmark/results/robustness_raw.csv — run
benchmark/run_robustness.py first if missing)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

ROBUSTNESS_RAW = os.path.join(os.path.dirname(__file__), "results", "robustness_raw.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "results", "robustness_statistics.csv")
CATASTROPHIC_THRESHOLD = 0.9


def _severity_curve(df: pd.DataFrame, corruption_type: str, system: str) -> pd.Series:
    sub = df[(df["corruption_type"] == corruption_type) & (df["system"] == system)]
    if sub.empty:
        raise ValueError(f"no rows for corruption_type={corruption_type!r}, system={system!r}")
    return sub.groupby("severity_value")["cer"].mean().sort_index()


def worst_severity_cer(df: pd.DataFrame, corruption_type: str, system: str) -> float:
    """Mean CER at the single highest tested severity — "how bad does it
    get at the worst level we actually tested," not an extrapolation
    beyond that."""
    curve = _severity_curve(df, corruption_type, system)
    return curve.iloc[-1]


def degradation_slope(df: pd.DataFrame, corruption_type: str, system: str) -> float:
    """Ordinary least-squares slope of mean CER against raw severity
    value across the tested levels. A larger slope means CER rises faster
    per unit of severity — comparable within one corruption_type across
    systems, NOT across corruption_types (severity units differ: sigma,
    degrees, JPEG quality)."""
    curve = _severity_curve(df, corruption_type, system)
    xs, ys = curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 severity levels to compute a slope")
    x_mean, y_mean = xs.mean(), ys.mean()
    denom = ((xs - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((xs - x_mean) * (ys - y_mean)).sum() / denom)


def catastrophic_onset_severity(df: pd.DataFrame, corruption_type: str, system: str,
                                 threshold: float = CATASTROPHIC_THRESHOLD) -> float | None:
    """The lowest tested severity at which mean CER first reaches
    `threshold` (default 0.9 — near-total transcription failure). Returns
    None if no tested severity reaches it — meaning the cliff, if one
    exists, sits beyond the range this sweep tested, which is itself a
    meaningful (if incomplete) result and must not be silently reported
    as "no cliff.\""""
    curve = _severity_curve(df, corruption_type, system)
    onset = curve[curve >= threshold]
    return float(onset.index[0]) if not onset.empty else None


def area_under_curve(df: pd.DataFrame, corruption_type: str, system: str) -> float:
    """Trapezoidal area under the CER-vs-severity curve, with severity
    normalized to [0, 1] first so the result is comparable across
    corruption types with different severity units (this normalization
    means the AUC answers "how much does CER accumulate across the
    FRACTION of the tested range," not an absolute-unit integral)."""
    curve = _severity_curve(df, corruption_type, system)
    xs, ys = curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float)
    x_range = xs.max() - xs.min()
    if x_range == 0:
        raise ValueError("cannot normalize an AUC over a single severity level")
    x_norm = (xs - xs.min()) / x_range
    area = 0.0
    for i in range(1, len(x_norm)):
        area += (x_norm[i] - x_norm[i - 1]) * (ys[i] + ys[i - 1]) / 2
    return area


def build_statistics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for corruption_type in sorted(df["corruption_type"].unique()):
        for system in sorted(df[df["corruption_type"] == corruption_type]["system"].unique()):
            rows.append({
                "corruption_type": corruption_type,
                "system": system,
                "worst_severity_cer": worst_severity_cer(df, corruption_type, system),
                "degradation_slope": degradation_slope(df, corruption_type, system),
                "catastrophic_onset_severity": catastrophic_onset_severity(df, corruption_type, system),
                "auc_normalized": area_under_curve(df, corruption_type, system),
            })
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    df = pd.read_csv(ROBUSTNESS_RAW)
    table = build_statistics_table(df)
    table.to_csv(OUT_PATH, index=False)
    print("=== Robustness statistics (worst-severity CER, degradation slope, "
          "catastrophic onset, normalized AUC) ===")
    print(table.round(4).to_string(index=False))
    print(f"\nWritten to {OUT_PATH}")
    print("\nNote: these summary numbers do NOT replace the full curves in "
          "docs/robustness_curves.md — a single score can hide exactly where "
          "a cliff sits, which is what the curves are for.")
    return table


if __name__ == "__main__":
    main()
