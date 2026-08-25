import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))

from stats_utils import bootstrap_ci, latency_variance_stats


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_ci([])


def test_bootstrap_ci_rejects_invalid_confidence_level():
    with pytest.raises(ValueError, match="between 0 and 1"):
        bootstrap_ci([1.0, 2.0], ci=1.5)


def test_bootstrap_ci_mean_matches_sample_mean():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = bootstrap_ci(values, seed=42)
    assert result["mean"] == pytest.approx(0.3)


def test_bootstrap_ci_contains_the_true_mean_for_a_reasonable_sample():
    # 100 samples tightly clustered around 0.1 - the CI should be narrow
    # and should contain 0.1.
    values = [0.1] * 90 + [0.15] * 5 + [0.05] * 5
    result = bootstrap_ci(values, n_resamples=2000, seed=1)
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]


def test_bootstrap_ci_is_narrower_for_larger_samples():
    """More data -> more confident (narrower) interval, holding the
    underlying variability constant — the basic sanity check for any CI
    method."""
    small_sample = [0.1, 0.5] * 3       # n=6, high variance, few points
    large_sample = [0.1, 0.5] * 100     # n=200, same variance, many points
    small_ci = bootstrap_ci(small_sample, n_resamples=2000, seed=7)
    large_ci = bootstrap_ci(large_sample, n_resamples=2000, seed=7)
    small_width = small_ci["ci_upper"] - small_ci["ci_lower"]
    large_width = large_ci["ci_upper"] - large_ci["ci_lower"]
    assert large_width < small_width


def test_bootstrap_ci_is_deterministic_given_a_seed():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    a = bootstrap_ci(values, seed=42)
    b = bootstrap_ci(values, seed=42)
    assert a == b


def test_latency_variance_stats_rejects_empty_input():
    with pytest.raises(ValueError, match="empty sample"):
        latency_variance_stats([])


def test_latency_variance_stats_zero_variance_for_identical_repeats():
    stats = latency_variance_stats([0.5, 0.5, 0.5])
    assert stats["std"] == 0.0
    assert stats["coefficient_of_variation"] == 0.0


def test_latency_variance_stats_reports_min_max_and_cv():
    stats = latency_variance_stats([0.1, 0.2, 0.3])
    assert stats["min"] == 0.1
    assert stats["max"] == 0.3
    assert stats["mean"] == pytest.approx(0.2)
    assert stats["coefficient_of_variation"] > 0
