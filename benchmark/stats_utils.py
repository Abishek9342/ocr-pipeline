"""Statistical rigor (mission section 21 / the routing mission's Phase 21
gap): every benchmark number published in this repo up to now has been a
single-run point estimate with no uncertainty quantification — no
confidence interval on CER, no repeated-run latency variance. This module
adds both, using the simplest defensible method for each (percentile
bootstrap for CIs — no distributional assumption needed; direct repeated
measurement for latency variance — no modeling assumption needed either),
consistent with this project's habit of starting simple and only reaching
for something more complex once simple is shown insufficient.

Every function here validates its own inputs and raises rather than
silently dropping or coercing bad data (an empty list, mismatched paired
lengths, or an out-of-range percentile is a caller bug worth surfacing,
not something to paper over).
"""
from __future__ import annotations

import random
from collections.abc import Callable


def _bootstrap_resample_statistic(
    values: list[float], statistic_fn: Callable[[list[float]], float], n_resamples: int, seed: int | None,
) -> list[float]:
    """Shared resampling core: draw `n_resamples` bootstrap resamples (with
    replacement, same size as `values`) and apply `statistic_fn` to each.
    Returns the resampled statistics, SORTED, ready for a percentile cut."""
    rng = random.Random(seed)
    n = len(values)
    results = [statistic_fn([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples)]
    results.sort()
    return results


def _percentile_cut(sorted_values: list[float], ci: float) -> tuple[float, float]:
    n = len(sorted_values)
    alpha = 1 - ci
    lo_idx = int(n * (alpha / 2))
    hi_idx = min(n - 1, int(n * (1 - alpha / 2)))
    return sorted_values[lo_idx], sorted_values[hi_idx]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def bootstrap_ci(values: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int | None = None) -> dict:
    """Percentile bootstrap confidence interval on the MEAN of `values`."""
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")

    resampled = _bootstrap_resample_statistic(values, _mean, n_resamples, seed)
    lo, hi = _percentile_cut(resampled, ci)
    return {"mean": _mean(values), "ci_lower": lo, "ci_upper": hi, "n": len(values), "ci_level": ci}


def bootstrap_median_ci(values: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int | None = None) -> dict:
    """Percentile bootstrap confidence interval on the MEDIAN of `values`
    — more robust to outliers than `bootstrap_ci`'s mean when a
    distribution has a heavy tail (e.g. a handful of catastrophic-failure
    CER=1.0 rows among otherwise-small values)."""
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")

    resampled = _bootstrap_resample_statistic(values, _median, n_resamples, seed)
    lo, hi = _percentile_cut(resampled, ci)
    return {"median": _median(values), "ci_lower": lo, "ci_upper": hi, "n": len(values), "ci_level": ci}


def paired_bootstrap_diff_ci(
    treatment: list[float], baseline: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int | None = None,
) -> dict:
    """CI on mean(treatment) - mean(baseline), where the two lists are
    PAIRED (same length, index i in both lists is the same underlying
    case — e.g. the same image/preset for two different systems). Pairing
    matters: it removes case-to-case variance from the comparison,
    leaving only the treatment-vs-baseline effect, which is why this is
    the correct method for "ours vs. PaddleOCR on the same 220 rows," not
    treating the two lists as independent samples.

    `distinguishable_from_zero` is deliberately a plain bool, not a
    p-value — this project doesn't claim significance testing beyond what
    a percentile-bootstrap CI supports (mission section 4: "do not call a
    difference statistically significant unless the method supports that
    wording" — CI-excludes-zero is the wording this method supports)."""
    if len(treatment) != len(baseline):
        raise ValueError(f"treatment and baseline must be paired (same length): {len(treatment)} != {len(baseline)}")
    if not treatment:
        raise ValueError("cannot bootstrap an empty paired sample")

    diffs = [t - b for t, b in zip(treatment, baseline)]
    ci_result = bootstrap_ci(diffs, n_resamples=n_resamples, ci=ci, seed=seed)
    return {
        "mean_difference": ci_result["mean"],
        "ci_lower": ci_result["ci_lower"],
        "ci_upper": ci_result["ci_upper"],
        "n": ci_result["n"],
        "ci_level": ci,
        "distinguishable_from_zero": ci_result["ci_lower"] > 0 or ci_result["ci_upper"] < 0,
    }


def paired_effect_size(treatment: list[float], baseline: list[float]) -> float:
    """Cohen's d_z for paired samples: mean(difference) / std(difference).
    A magnitude, not a significance test — report alongside the CI from
    `paired_bootstrap_diff_ci`, not instead of it. Returns 0.0 if the
    differences have zero variance (e.g. treatment == baseline everywhere
    in the sample), rather than dividing by zero."""
    if len(treatment) != len(baseline):
        raise ValueError(f"treatment and baseline must be paired (same length): {len(treatment)} != {len(baseline)}")
    if not treatment:
        raise ValueError("cannot compute an effect size on an empty paired sample")

    diffs = [t - b for t, b in zip(treatment, baseline)]
    n = len(diffs)
    mean_diff = _mean(diffs)
    if n < 2:
        return 0.0
    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std = variance**0.5
    return mean_diff / std if std > 0 else 0.0


def percentile_interval(values: list[float], lower_pct: float = 2.5, upper_pct: float = 97.5) -> tuple[float, float]:
    """A direct (non-bootstrap) percentile interval of the raw OBSERVATIONS
    themselves — describes how spread out individual values are, NOT
    uncertainty in an estimate of the mean (that's what `bootstrap_ci` is
    for). Useful for e.g. "what does a typical worst-case-ish row look
    like" rather than "how confident are we in the average.\""""
    if not values:
        raise ValueError("cannot compute a percentile interval on an empty sample")
    if not (0 <= lower_pct < upper_pct <= 100):
        raise ValueError(f"require 0 <= lower_pct < upper_pct <= 100, got {lower_pct}, {upper_pct}")

    s = sorted(values)
    n = len(s)

    def _pct(p: float) -> float:
        idx = min(n - 1, max(0, round(p / 100 * (n - 1))))
        return s[idx]

    return _pct(lower_pct), _pct(upper_pct)


def pairwise_comparison_summary(
    baseline_name: str, baseline_values: list[float], others: dict[str, list[float]],
    n_resamples: int = 2000, seed: int | None = None,
) -> list[dict]:
    """Paired-compares every system in `others` against `baseline_values`
    (all must be the same length and index-aligned — same underlying
    cases). Returns one summary row per comparison: the system name, the
    paired-difference CI, and the effect size. Deliberately does NOT
    apply a multiple-comparison correction by default — see
    `bonferroni_correction` below and this project's own multiple-
    comparison audit (`docs/OVERNIGHT_RESEARCH_REPORT.md`) for why: the
    headline claim this repo makes is narrow (one pipeline vs. baselines
    on one benchmark), not an exploratory sweep over many hypotheses."""
    if not others:
        raise ValueError("others must contain at least one system to compare")
    rows = []
    for name, values in others.items():
        diff_ci = paired_bootstrap_diff_ci(values, baseline_values, n_resamples=n_resamples, seed=seed)
        effect = paired_effect_size(values, baseline_values)
        rows.append({"system": name, "baseline": baseline_name, **diff_ci, "cohens_d": effect})
    return rows


def bonferroni_correction(alpha: float, n_comparisons: int) -> float:
    """The simplest, most conservative multiple-comparison correction:
    divide the significance level by the number of comparisons. Provided
    as an OPT-IN utility, not applied anywhere by default — see the
    module-level note on why this project keeps its central claim narrow
    enough to mostly avoid needing it."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be at least 1")
    return alpha / n_comparisons


def latency_variance_stats(latencies: list[float]) -> dict:
    """Mean/std/coefficient-of-variation/median/P95/min/max across
    REPEATED measurements of the same thing (e.g. the same image run N
    times) — distinct from the variance ACROSS different images, which
    the main benchmark already reports (median/P95 over the whole
    corpus). A high CV here means "this latency number itself is noisy
    run-to-run," independent of how much images differ from each other."""
    if not latencies:
        raise ValueError("cannot compute variance stats on an empty sample")
    n = len(latencies)
    mean = _mean(latencies)
    variance = sum((x - mean) ** 2 for x in latencies) / n if n > 1 else 0.0
    std = variance**0.5
    sorted_latencies = sorted(latencies)
    p95_idx = min(n - 1, round(0.95 * (n - 1)))
    return {
        "mean": mean, "std": std,
        "coefficient_of_variation": (std / mean) if mean > 0 else 0.0,
        "median": _median(latencies), "p95": sorted_latencies[p95_idx],
        "min": min(latencies), "max": max(latencies), "n": n,
    }
