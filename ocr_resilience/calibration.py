"""Confidence calibration (mission Phase 9-10): is a given engine's raw
confidence actually informative about how correct its output is, and can
that relationship be made comparable across engines?

Starts with the SIMPLEST defensible method — binned/histogram calibration
— rather than jumping straight to Platt scaling or isotonic regression,
per the mission's own explicit instruction to validate the simplest
approach first. Isotonic regression is a natural next step if binned
calibration proves too coarse (see `docs/confidence_calibration_report.md`
for whether that's actually necessary here) — not implemented
speculatively ahead of that need, and deliberately avoiding a new
dependency (e.g. scikit-learn) for a method not yet shown necessary.

Methodological honesty: there is no ground-truth "P(this detection is
correct)" label available — only per-image CER against ground truth. This
module uses `1 - min(cer, 1.0)` as a CONTINUOUS correctness proxy, not a
true binary correct/incorrect label. A calibrated output here is a
"quality proxy," not a validated probability — never present it as one.
"""
from __future__ import annotations

from dataclasses import dataclass

from .engines import Detection


def correctness_proxy(cer: float) -> float:
    """1.0 = perfect match, 0.0 = completely wrong (CER >= 1.0). A
    continuous stand-in for "was this correct," not a binary label."""
    return 1.0 - min(cer, 1.0)


@dataclass
class BinnedCalibrator:
    """Maps a raw confidence value to the empirical mean correctness
    observed in whichever training bin it falls into. `bin_edges` has
    `len(bin_correctness) + 1` entries."""
    bin_edges: list[float]
    bin_correctness: list[float]
    bin_counts: list[int]

    def calibrate(self, raw_confidence: float) -> float:
        for i in range(len(self.bin_correctness)):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            is_last_bin = i == len(self.bin_correctness) - 1
            if lo <= raw_confidence < hi or (is_last_bin and raw_confidence == hi):
                return self.bin_correctness[i]
        # raw_confidence outside [bin_edges[0], bin_edges[-1]] (e.g. an engine
        # occasionally reports something outside its usual range) — clamp to
        # the nearest edge bin rather than raise, since a router still needs
        # SOME calibrated value for every observed confidence.
        return self.bin_correctness[0] if raw_confidence < self.bin_edges[0] else self.bin_correctness[-1]


def fit_binned_calibrator(raw_confidences: list[float], cers: list[float], n_bins: int = 10) -> BinnedCalibrator:
    if len(raw_confidences) != len(cers):
        raise ValueError("raw_confidences and cers must be the same length")
    if not raw_confidences:
        raise ValueError("cannot fit a calibrator on zero samples")

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bin_correctness = []
    bin_counts = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        is_last = i == n_bins - 1
        members = [
            correctness_proxy(cer) for conf, cer in zip(raw_confidences, cers)
            if lo <= conf < hi or (is_last and conf == hi)
        ]
        bin_counts.append(len(members))
        # Empty bin: fall back to the overall mean correctness rather than an
        # arbitrary 0.0/1.0 — an empty bin means "no evidence," not "always wrong."
        bin_correctness.append(sum(members) / len(members) if members else sum(correctness_proxy(c) for c in cers) / len(cers))

    return BinnedCalibrator(bin_edges=bin_edges, bin_correctness=bin_correctness, bin_counts=bin_counts)


def reliability_curve(raw_confidences: list[float], cers: list[float], n_bins: int = 10) -> list[dict]:
    """For each bin: mean predicted confidence, mean actual correctness
    (proxy), and sample count — a perfectly calibrated engine has
    predicted ≈ actual in every non-empty bin."""
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        is_last = i == n_bins - 1
        members = [
            (conf, correctness_proxy(cer)) for conf, cer in zip(raw_confidences, cers)
            if lo <= conf < hi or (is_last and conf == hi)
        ]
        rows.append({
            "bin_lo": lo, "bin_hi": hi, "count": len(members),
            "mean_predicted_confidence": sum(c for c, _ in members) / len(members) if members else None,
            "mean_actual_correctness": sum(a for _, a in members) / len(members) if members else None,
        })
    return rows


def expected_calibration_error(raw_confidences: list[float], cers: list[float], n_bins: int = 10) -> float:
    """Standard ECE: sample-weighted mean absolute gap between predicted
    confidence and actual (proxy) correctness across bins. 0.0 = perfectly
    calibrated; higher = worse. Comparable across engines only insofar as
    the correctness proxy itself is comparable (see module docstring)."""
    curve = reliability_curve(raw_confidences, cers, n_bins)
    total = sum(row["count"] for row in curve)
    if total == 0:
        return 0.0
    return sum(
        row["count"] * abs(row["mean_predicted_confidence"] - row["mean_actual_correctness"])
        for row in curve if row["count"] > 0
    ) / total


def apply_calibration(detections: list[Detection], calibrators: dict[str, BinnedCalibrator]) -> list[Detection]:
    """Returns NEW `Detection` objects with each one's `confidence`
    replaced by its engine's calibrated value — raw detections are never
    mutated in place. A detection whose engine has no entry in
    `calibrators` keeps its raw confidence unchanged (fails open, doesn't
    raise, since a caller comparing calibrated vs. uncalibrated fusion
    might legitimately only have calibrators for some engines)."""
    out = []
    for det in detections:
        calibrator = calibrators.get(det.engine)
        confidence = calibrator.calibrate(det.confidence) if calibrator else det.confidence
        out.append(Detection(text=det.text, confidence=confidence, bbox=det.bbox, engine=det.engine))
    return out
