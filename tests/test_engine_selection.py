"""Tests for quality-aware single-engine selection — the replacement for
picking `available_engines[0]` (pure registration order). Each rule is
tested in isolation via a synthetic QualityReport, independent of any
particular benchmark run's exact numbers, so these stay stable even as
thresholds get recalibrated against fresh data."""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.engine_selection import select_primary_engine
from ocr_resilience.quality import QualityReport

CLEAN = QualityReport(
    blur_score=500.0, noise_score=1.0, impulse_noise_score=0.0, contrast_score=80.0,
    brightness=200.0, skew_angle_deg=0.0, is_blurry=False, is_noisy=False,
    is_impulse_noisy=False, is_low_contrast=False, is_skewed=False, likely_handwritten=False,
)


def test_falls_back_to_first_available_engine_when_no_rule_matches():
    engine, reason = select_primary_engine(CLEAN, ["tesseract", "easyocr"])
    assert engine == "tesseract"
    assert "no condition-specific rule matched" in reason


def test_prefers_paddleocr_for_low_contrast_when_available():
    report = replace(CLEAN, is_low_contrast=True)
    engine, reason = select_primary_engine(report, ["tesseract", "paddleocr", "easyocr"])
    assert engine == "paddleocr"
    assert "low contrast" in reason


def test_falls_through_when_preferred_engine_not_registered():
    report = replace(CLEAN, is_low_contrast=True)
    engine, _ = select_primary_engine(report, ["tesseract", "easyocr"])  # no paddleocr available
    assert engine == "tesseract"  # falls back, doesn't crash or return an unregistered engine


def test_prefers_paddleocr_or_tesseract_over_rapidocr_for_severe_blur():
    report = replace(CLEAN, is_blurry=True, blur_score=10.0)  # well below the is_blurry threshold of 100
    engine, reason = select_primary_engine(report, ["rapidocr", "tesseract", "paddleocr"])
    assert engine in ("paddleocr", "tesseract")
    assert engine != "rapidocr"
    assert "blur" in reason


def test_mild_blur_does_not_trigger_the_severe_blur_rule():
    report = replace(CLEAN, is_blurry=True, blur_score=90.0)  # is_blurry, but not severely so
    engine, reason = select_primary_engine(report, ["rapidocr", "tesseract"])
    assert engine == "rapidocr"  # falls through to first-available, the severe-blur rule doesn't fire
    assert "no condition-specific rule matched" in reason


def test_prefers_paddleocr_or_easyocr_over_tesseract_for_high_general_noise():
    report = replace(CLEAN, is_noisy=True, noise_score=30.0)  # above the sigma-cliff proxy threshold
    engine, reason = select_primary_engine(report, ["tesseract", "paddleocr"])
    assert engine == "paddleocr"
    assert "noise" in reason


def test_mild_noise_does_not_trigger_the_high_noise_rule():
    report = replace(CLEAN, is_noisy=True, noise_score=9.0)  # is_noisy, but below the severity threshold
    engine, reason = select_primary_engine(report, ["tesseract", "paddleocr"])
    assert engine == "tesseract"  # first-available; rule doesn't fire at this severity
    assert "no condition-specific rule matched" in reason


def test_impulse_noise_alone_asserts_no_engine_preference():
    """Deliberate: impulse noise is handled by preprocessing
    (median_denoise), not by an engine-preference rule not yet backed by
    post-preprocessing evidence — see the module docstring."""
    report = replace(CLEAN, is_impulse_noisy=True)
    engine, reason = select_primary_engine(report, ["tesseract", "paddleocr"])
    assert engine == "tesseract"
    assert "no condition-specific rule matched" in reason


def test_prefers_paddleocr_for_skew():
    report = replace(CLEAN, is_skewed=True)
    engine, reason = select_primary_engine(report, ["tesseract", "paddleocr"])
    assert engine == "paddleocr"
    assert "skew" in reason


def test_never_raises_and_always_returns_an_available_engine():
    reports = [
        replace(CLEAN, is_noisy=True, noise_score=100.0, is_blurry=True, blur_score=1.0,
                is_skewed=True, is_low_contrast=True, is_impulse_noisy=True),
        CLEAN,
    ]
    for report in reports:
        engine, reason = select_primary_engine(report, ["tesseract"])
        assert engine == "tesseract"
        assert isinstance(reason, str) and reason
