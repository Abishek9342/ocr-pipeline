"""An OPTIONAL composite score — NOT a replacement for the underlying
metrics, which are always published alongside it (see `benchmark/results/
summary.csv`). Reducing a multi-objective comparison (accuracy, latency,
memory) to one number is convenient but easy to make misleading (see the
project's own README: "the pipeline is not the single best performer on
every row" — averaging in a single index can quietly bury that). This
module exists so that IF a single ranking number is wanted, its formula is
explicit, its weights are named (not buried in a magic constant), and its
sensitivity to those weights is checkable via `rank_stability`, rather
than presented as an objective verdict.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreWeights:
    accuracy: float = 0.6
    latency: float = 0.25
    memory: float = 0.15

    def __post_init__(self):
        total = self.accuracy + self.latency + self.memory
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ScoreWeights must sum to 1.0, got {total}")


def composite_score(
    mean_cer: float,
    mean_latency_sec: float,
    mean_peak_memory_bytes: float,
    *,
    latency_reference_sec: float,
    memory_reference_bytes: float,
    weights: ScoreWeights = ScoreWeights(),
) -> float:
    """Higher is better, roughly on a 0-1 scale for reasonable inputs.

    accuracy component = 1 - mean_cer (clamped to >= 0; a CER above 1.0
        can happen with a badly wrong long prediction, and shouldn't make
        the accuracy component go negative and dominate the sum's sign).
    latency/memory components = latency_reference / actual (i.e. relative
        to a caller-supplied reference system — there is no
        environment-independent absolute scale for either, so this
        function refuses to invent one; pass another system's numbers
        from the same benchmark run as the reference, not a hardcoded
        constant).

    This does NOT hide the underlying numbers — callers should report
    mean_cer/mean_latency_sec/mean_peak_memory_bytes alongside this score,
    never instead of them.
    """
    accuracy_component = max(0.0, 1.0 - mean_cer)
    latency_component = latency_reference_sec / mean_latency_sec if mean_latency_sec > 0 else 0.0
    memory_component = memory_reference_bytes / mean_peak_memory_bytes if mean_peak_memory_bytes > 0 else 0.0
    return (
        weights.accuracy * accuracy_component
        + weights.latency * latency_component
        + weights.memory * memory_component
    )


def rank_stability(
    systems: dict[str, dict[str, float]],
    *,
    reference_system: str,
    weight_variants: list[ScoreWeights] | None = None,
) -> dict[str, list[str]]:
    """For each of several plausible weightings, compute the ranking of
    `systems` (each a dict with mean_cer/mean_latency_sec/mean_peak_memory_bytes)
    and return {weighting_label: [systems, best_to_worst]}. Use this to
    check whether a headline ranking claim ("X is best overall") survives
    reasonable changes to the weights, per the mission's own requirement:
    "test whether rankings are robust to reasonable weight changes." If
    the ranking flips under a mild reweighting, that's a real finding to
    report, not a reason to pick whichever weighting gives the preferred
    answer."""
    variants = weight_variants or [
        ScoreWeights(accuracy=0.6, latency=0.25, memory=0.15),
        ScoreWeights(accuracy=0.4, latency=0.4, memory=0.2),
        ScoreWeights(accuracy=0.8, latency=0.1, memory=0.1),
    ]
    ref = systems[reference_system]
    rankings = {}
    for weights in variants:
        scores = {
            name: composite_score(
                s["mean_cer"], s["mean_latency_sec"], s["mean_peak_memory_bytes"],
                latency_reference_sec=ref["mean_latency_sec"],
                memory_reference_bytes=ref["mean_peak_memory_bytes"],
                weights=weights,
            )
            for name, s in systems.items()
        }
        label = f"accuracy={weights.accuracy},latency={weights.latency},memory={weights.memory}"
        rankings[label] = sorted(scores, key=lambda name: -scores[name])
    return rankings
