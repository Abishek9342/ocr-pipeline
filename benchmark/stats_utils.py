"""Statistical rigor (mission section 21 / the routing mission's Phase 21
gap): every benchmark number published in this repo up to now has been a
single-run point estimate with no uncertainty quantification — no
confidence interval on CER, no repeated-run latency variance. This module
adds both, using the simplest defensible method for each (percentile
bootstrap for CIs — no distributional assumption needed; direct repeated
measurement for latency variance — no modeling assumption needed either),
consistent with this project's habit of starting simple and only reaching
for something more complex once simple is shown insufficient.
"""
from __future__ import annotations

import random


def bootstrap_ci(values: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int | None = None) -> dict:
    """Percentile bootstrap confidence interval on the MEAN of `values`.
    No assumption about the underlying distribution — resamples `values`
    with replacement `n_resamples` times, takes the mean of each resample,
    and reports the `ci` central interval of those resampled means."""
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")

    rng = random.Random(seed)
    n = len(values)
    resampled_means = []
    for _ in range(n_resamples):
        resample_sum = sum(values[rng.randrange(n)] for _ in range(n))
        resampled_means.append(resample_sum / n)
    resampled_means.sort()

    alpha = 1 - ci
    lo_idx = int(n_resamples * (alpha / 2))
    hi_idx = min(n_resamples - 1, int(n_resamples * (1 - alpha / 2)))

    return {
        "mean": sum(values) / n,
        "ci_lower": resampled_means[lo_idx],
        "ci_upper": resampled_means[hi_idx],
        "n": n,
        "ci_level": ci,
    }


def latency_variance_stats(latencies: list[float]) -> dict:
    """Mean/std/coefficient-of-variation across REPEATED measurements of
    the same thing (e.g. the same image run N times) — distinct from the
    variance ACROSS different images, which the main benchmark already
    reports (median/P95). A high CV here means "this latency number
    itself is noisy run-to-run," independent of how much images differ
    from each other."""
    if not latencies:
        raise ValueError("cannot compute variance stats on an empty sample")
    n = len(latencies)
    mean = sum(latencies) / n
    variance = sum((x - mean) ** 2 for x in latencies) / n if n > 1 else 0.0
    std = variance**0.5
    return {
        "mean": mean, "std": std,
        "coefficient_of_variation": (std / mean) if mean > 0 else 0.0,
        "min": min(latencies), "max": max(latencies), "n": n,
    }
