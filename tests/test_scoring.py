import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.scoring import ScoreWeights, composite_score, rank_stability


def test_score_weights_must_sum_to_one():
    ScoreWeights(accuracy=0.5, latency=0.3, memory=0.2)  # ok
    with pytest.raises(ValueError, match="must sum to 1.0"):
        ScoreWeights(accuracy=0.5, latency=0.5, memory=0.5)


def test_composite_score_rewards_lower_cer():
    good = composite_score(0.05, 0.2, 1_000_000, latency_reference_sec=0.2, memory_reference_bytes=1_000_000)
    bad = composite_score(0.5, 0.2, 1_000_000, latency_reference_sec=0.2, memory_reference_bytes=1_000_000)
    assert good > bad


def test_composite_score_rewards_lower_latency_relative_to_reference():
    fast = composite_score(0.1, 0.1, 1_000_000, latency_reference_sec=0.1, memory_reference_bytes=1_000_000)
    slow = composite_score(0.1, 1.0, 1_000_000, latency_reference_sec=0.1, memory_reference_bytes=1_000_000)
    assert fast > slow


def test_composite_score_clamps_accuracy_component_at_zero_for_cer_above_one():
    score = composite_score(5.0, 0.1, 1_000_000, latency_reference_sec=0.1, memory_reference_bytes=1_000_000)
    assert score >= 0.0


def test_rank_stability_returns_one_ranking_per_weighting():
    systems = {
        "a": {"mean_cer": 0.05, "mean_latency_sec": 0.5, "mean_peak_memory_bytes": 2_000_000},
        "b": {"mean_cer": 0.15, "mean_latency_sec": 0.1, "mean_peak_memory_bytes": 100_000},
    }
    rankings = rank_stability(systems, reference_system="b")
    assert len(rankings) == 3
    for ranking in rankings.values():
        assert set(ranking) == {"a", "b"}


def test_rank_stability_can_flip_under_reasonable_reweighting():
    """The whole point of this function: a system that's much more
    accurate but much slower can rank #1 under an accuracy-heavy weighting
    and #2 under a latency-heavy one. If it never flips for ANY of these
    three variants, the two systems aren't different enough for this test
    to be meaningful — this fixture is deliberately extreme in both
    directions so a flip is guaranteed and demonstrable."""
    systems = {
        "accurate_but_slow": {"mean_cer": 0.01, "mean_latency_sec": 5.0, "mean_peak_memory_bytes": 1_000_000},
        "fast_but_inaccurate": {"mean_cer": 0.5, "mean_latency_sec": 0.05, "mean_peak_memory_bytes": 1_000_000},
    }
    rankings = rank_stability(systems, reference_system="fast_but_inaccurate")
    winners = {ranking[0] for ranking in rankings.values()}
    assert len(winners) == 2  # both systems win under at least one weighting
