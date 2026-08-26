import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))

from stats_utils import (
    bonferroni_correction,
    bootstrap_ci,
    bootstrap_median_ci,
    latency_variance_stats,
    paired_bootstrap_diff_ci,
    paired_effect_size,
    pairwise_comparison_summary,
    percentile_interval,
)


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


def test_latency_variance_stats_reports_median_and_p95():
    stats = latency_variance_stats([0.1, 0.2, 0.3, 0.4, 0.5])
    assert stats["median"] == pytest.approx(0.3)
    assert stats["p95"] == pytest.approx(0.5)


def test_bootstrap_median_ci_rejects_empty_input():
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_median_ci([])


def test_bootstrap_median_ci_matches_sample_median():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = bootstrap_median_ci(values, seed=42)
    assert result["median"] == pytest.approx(3.0)
    assert result["ci_lower"] <= result["median"] <= result["ci_upper"]


def test_bootstrap_median_ci_is_deterministic_given_a_seed():
    values = [0.1, 0.2, 0.3, 0.9, 1.0]
    a = bootstrap_median_ci(values, seed=3)
    b = bootstrap_median_ci(values, seed=3)
    assert a == b


def test_paired_bootstrap_diff_ci_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap_diff_ci([1.0, 2.0], [1.0])


def test_paired_bootstrap_diff_ci_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_diff_ci([], [])


def test_paired_bootstrap_diff_ci_zero_when_identical():
    treatment = baseline = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = paired_bootstrap_diff_ci(treatment, baseline, seed=1)
    assert result["mean_difference"] == pytest.approx(0.0)
    assert not result["distinguishable_from_zero"]


def test_paired_bootstrap_diff_ci_detects_a_clear_effect():
    baseline = [0.5] * 50
    treatment = [0.1] * 50  # consistently 0.4 lower, no noise
    result = paired_bootstrap_diff_ci(treatment, baseline, seed=1)
    assert result["mean_difference"] == pytest.approx(-0.4)
    assert result["distinguishable_from_zero"]
    assert result["ci_upper"] < 0


def test_paired_effect_size_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="paired"):
        paired_effect_size([1.0, 2.0], [1.0])


def test_paired_effect_size_zero_for_identical_samples():
    values = [0.1, 0.2, 0.3, 0.4]
    assert paired_effect_size(values, values) == 0.0


def test_paired_effect_size_is_signed():
    baseline = [0.5, 0.5, 0.5, 0.5]
    lower = [0.1, 0.1, 0.2, 0.2]   # treatment consistently lower -> negative d
    higher = [0.9, 0.9, 0.8, 0.8]  # treatment consistently higher -> positive d
    assert paired_effect_size(lower, baseline) < 0
    assert paired_effect_size(higher, baseline) > 0


def test_percentile_interval_rejects_empty_input():
    with pytest.raises(ValueError, match="empty sample"):
        percentile_interval([])


def test_percentile_interval_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        percentile_interval([1.0, 2.0], lower_pct=60, upper_pct=40)


def test_percentile_interval_spans_the_full_range_at_0_100():
    values = [3.0, 1.0, 5.0, 2.0, 4.0]
    lo, hi = percentile_interval(values, lower_pct=0, upper_pct=100)
    assert lo == 1.0
    assert hi == 5.0


def test_pairwise_comparison_summary_rejects_empty_others():
    with pytest.raises(ValueError, match="at least one"):
        pairwise_comparison_summary("ours", [0.1, 0.2], {})


def test_pairwise_comparison_summary_reports_one_row_per_system():
    baseline = [0.1, 0.2, 0.3, 0.4]
    others = {"tesseract": [0.5, 0.6, 0.7, 0.8], "paddleocr": [0.1, 0.2, 0.3, 0.4]}
    rows = pairwise_comparison_summary("ours", baseline, others, seed=0)
    by_system = {r["system"]: r for r in rows}
    assert set(by_system) == {"tesseract", "paddleocr"}
    assert by_system["tesseract"]["distinguishable_from_zero"]
    assert by_system["paddleocr"]["mean_difference"] == pytest.approx(0.0)


def test_bonferroni_correction_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        bonferroni_correction(0.05, 0)
    with pytest.raises(ValueError):
        bonferroni_correction(1.5, 4)


def test_bonferroni_correction_divides_alpha_by_comparison_count():
    assert bonferroni_correction(0.05, 5) == pytest.approx(0.01)
