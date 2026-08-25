import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.calibration import (
    apply_calibration,
    correctness_proxy,
    expected_calibration_error,
    fit_binned_calibrator,
    reliability_curve,
)
from ocr_resilience.engines import Detection


def test_correctness_proxy_perfect_match():
    assert correctness_proxy(0.0) == 1.0


def test_correctness_proxy_total_failure():
    assert correctness_proxy(1.0) == 0.0


def test_correctness_proxy_clamps_cer_above_one():
    """CER can exceed 1.0 for a badly-wrong long prediction — correctness
    proxy should still floor at 0.0, not go negative."""
    assert correctness_proxy(2.5) == 0.0


def test_fit_binned_calibrator_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        fit_binned_calibrator([0.5], [0.1, 0.2])


def test_fit_binned_calibrator_rejects_empty_input():
    with pytest.raises(ValueError, match="zero samples"):
        fit_binned_calibrator([], [])


def test_perfectly_calibrated_engine_has_near_zero_ece():
    """An engine whose confidence exactly equals its correctness proxy
    every time should have ECE ~= 0."""
    confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
    cers = [1 - c for c in confidences]  # correctness_proxy(cer) == confidence exactly
    ece = expected_calibration_error(confidences, cers, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_overconfident_engine_has_high_ece():
    """An engine that always reports 0.95 confidence but is actually wrong
    (CER=1.0, correctness=0.0) every time should show a large gap."""
    confidences = [0.95] * 10
    cers = [1.0] * 10
    ece = expected_calibration_error(confidences, cers, n_bins=10)
    assert ece == pytest.approx(0.95, abs=0.05)


def test_binned_calibrator_maps_confidence_to_empirical_correctness():
    # bin [0.9, 1.0]: all samples here have correctness proxy 1.0
    confidences = [0.95, 0.92, 0.98]
    cers = [0.0, 0.0, 0.0]
    calibrator = fit_binned_calibrator(confidences, cers, n_bins=10)
    assert calibrator.calibrate(0.95) == pytest.approx(1.0)


def test_binned_calibrator_falls_back_to_overall_mean_for_empty_bin():
    # All samples in the top bin; the bottom bin [0.0, 0.1) is empty.
    confidences = [0.95] * 5
    cers = [0.2] * 5  # correctness proxy 0.8
    calibrator = fit_binned_calibrator(confidences, cers, n_bins=10)
    assert calibrator.calibrate(0.05) == pytest.approx(0.8)  # empty bin -> overall mean, not 0 or 1


def test_binned_calibrator_clamps_out_of_range_confidence():
    calibrator = fit_binned_calibrator([0.5], [0.5], n_bins=10)
    assert calibrator.calibrate(-1.0) == calibrator.calibrate(0.0)
    assert calibrator.calibrate(5.0) == calibrator.calibrate(1.0)


def test_reliability_curve_reports_count_per_bin():
    confidences = [0.05, 0.15, 0.95]
    cers = [0.0, 0.0, 0.0]
    curve = reliability_curve(confidences, cers, n_bins=10)
    assert sum(row["count"] for row in curve) == 3
    assert curve[0]["count"] == 1  # 0.05 -> bin [0.0, 0.1)
    assert curve[1]["count"] == 1  # 0.15 -> bin [0.1, 0.2)
    assert curve[-1]["count"] == 1  # 0.95 -> bin [0.9, 1.0]


def test_reliability_curve_none_for_empty_bins():
    curve = reliability_curve([0.95], [0.0], n_bins=10)
    assert curve[0]["mean_predicted_confidence"] is None  # bin [0.0, 0.1) has no samples
    assert curve[-1]["mean_predicted_confidence"] == pytest.approx(0.95)


def test_apply_calibration_replaces_confidence_for_known_engines():
    calibrator = fit_binned_calibrator([0.9], [0.0], n_bins=10)  # bin [0.9,1.0] -> correctness 1.0
    detections = [Detection("x", 0.9, (0, 0, 1, 1), "tesseract")]
    out = apply_calibration(detections, {"tesseract": calibrator})
    assert out[0].confidence == pytest.approx(1.0)
    assert out[0].text == "x" and out[0].bbox == (0, 0, 1, 1) and out[0].engine == "tesseract"


def test_apply_calibration_leaves_unknown_engines_unchanged():
    detections = [Detection("x", 0.42, (0, 0, 1, 1), "easyocr")]
    out = apply_calibration(detections, {})  # no calibrator for "easyocr"
    assert out[0].confidence == 0.42


def test_apply_calibration_does_not_mutate_input_detections():
    calibrator = fit_binned_calibrator([0.9], [0.0], n_bins=10)
    original = Detection("x", 0.9, (0, 0, 1, 1), "tesseract")
    apply_calibration([original], {"tesseract": calibrator})
    assert original.confidence == 0.9
