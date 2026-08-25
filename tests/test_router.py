import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.quality import QualityReport
from ocr_resilience.router import decide

CLEAN_REPORT = QualityReport(
    blur_score=500.0, noise_score=1.0, impulse_noise_score=0.0, contrast_score=80.0, brightness=200.0,
    skew_angle_deg=0.0, is_blurry=False, is_noisy=False, is_impulse_noisy=False, is_low_contrast=False,
    is_skewed=False, likely_handwritten=False,
)


def test_decide_raises_on_no_available_engines():
    with pytest.raises(ValueError, match="No OCR engines available"):
        decide(CLEAN_REPORT, [], degradation_threshold=2)


def test_decide_routes_clean_image_to_single_first_engine():
    decision = decide(CLEAN_REPORT, ["tesseract", "easyocr"], degradation_threshold=2)
    assert decision.engines_to_run == ["tesseract"]


def test_decide_routes_handwritten_to_ensemble_regardless_of_degradation_count():
    report = replace(CLEAN_REPORT, likely_handwritten=True)
    decision = decide(report, ["tesseract", "easyocr"], degradation_threshold=2)
    assert decision.engines_to_run == ["tesseract", "easyocr"]


def test_decide_ensembles_at_the_degradation_threshold_boundary():
    report = replace(CLEAN_REPORT, is_blurry=True, is_noisy=True)  # exactly 2 flags
    decision = decide(report, ["tesseract", "easyocr"], degradation_threshold=2)
    assert decision.engines_to_run == ["tesseract", "easyocr"]


def test_decide_stays_single_engine_just_below_threshold():
    report = replace(CLEAN_REPORT, is_blurry=True)  # only 1 flag
    decision = decide(report, ["tesseract", "easyocr"], degradation_threshold=2)
    assert decision.engines_to_run == ["tesseract"]
