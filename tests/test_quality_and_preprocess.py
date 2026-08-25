"""Regression test for the skew-angle sign bug the benchmark caught (see
quality.py's skew_angle docstring/comment for the full story), plus basic
sanity checks on the other quality metrics and preprocessing operators."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.metrics import cer
from ocr_resilience.preprocess import deskew, median_denoise
from ocr_resilience.quality import assess, blur_score, contrast_score, impulse_noise_score, noise_score, skew_angle


def _render_line(text: str = "Hello World 12345", width: int = 400, height: int = 100) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (20, height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))


def test_deskew_actually_reduces_measured_skew_not_increases_it():
    """The regression this guards: skew_angle() used to return a
    double-negated angle, so deskew() rotated the WRONG way and made
    skew worse, not better — only caught because the benchmark showed the
    "skewed" preset's accuracy getting worse after "correction," not by
    inspecting the angle math directly."""
    clean = _render_line()
    tilted = _rotate(clean, angle_deg=8.0)

    angle_before = abs(skew_angle(cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY)))
    corrected = deskew(cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY), skew_angle(cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY)))
    angle_after = abs(skew_angle(corrected))

    assert angle_after < angle_before, (
        f"deskew() should reduce measured skew ({angle_before:.2f} -> should be less), "
        f"got {angle_after:.2f} instead — rotation direction may be inverted again"
    )


def test_blur_score_lower_for_blurrier_image():
    sharp = _render_line()
    blurry = cv2.GaussianBlur(sharp, (0, 0), sigmaX=4)
    assert blur_score(cv2.cvtColor(blurry, cv2.COLOR_BGR2GRAY)) < blur_score(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY))


def test_contrast_score_lower_for_washed_out_image():
    normal = _render_line()
    washed = np.clip((normal.astype(np.float64) - 128) * 0.2 + 128, 0, 255).astype(np.uint8)
    assert contrast_score(cv2.cvtColor(washed, cv2.COLOR_BGR2GRAY)) < contrast_score(cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY))


def test_assess_flags_are_internally_consistent():
    report = assess(_render_line())
    assert isinstance(report.is_blurry, bool)
    assert isinstance(report.skew_angle_deg, float)


def test_cer_zero_for_exact_match():
    assert cer("Hello World", "Hello World") == 0.0


def test_cer_nonzero_for_mismatch():
    assert cer("Hallo World", "Hello World") > 0.0


def _add_salt_and_pepper(image: np.ndarray, amount: float = 0.03, rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    out = image.copy()
    mask = rng.random(image.shape[:2]) < amount
    out[mask] = rng.choice([0, 255], size=mask.sum())
    return out


def test_impulse_noise_score_is_near_zero_on_clean_text():
    clean = cv2.cvtColor(_render_line(), cv2.COLOR_BGR2GRAY)
    assert impulse_noise_score(clean) < 0.01


def test_impulse_noise_score_is_high_on_salt_and_pepper_noise():
    """Regression for a real finding from the full benchmark run: Tesseract
    returns completely empty output (1.0 CER) on salt-and-pepper-degraded
    images, and `noise_score` (a MAD-based ROBUST estimator, deliberately
    insensitive to sparse outliers) never flags it as noisy — so no
    denoising step ever ran. This is a distinct signal that must actually
    fire where `noise_score` does not."""
    clean_gray = cv2.cvtColor(_render_line(), cv2.COLOR_BGR2GRAY)
    noisy_gray = _add_salt_and_pepper(clean_gray)
    assert impulse_noise_score(noisy_gray) > 0.02
    # the whole point: noise_score's robust-statistic design stays low here
    assert noise_score(noisy_gray) < 8.0


def test_median_denoise_recovers_text_edges_after_salt_and_pepper():
    clean_gray = cv2.cvtColor(_render_line(), cv2.COLOR_BGR2GRAY)
    noisy_gray = _add_salt_and_pepper(clean_gray)
    denoised = median_denoise(noisy_gray)
    assert impulse_noise_score(denoised) < impulse_noise_score(noisy_gray)
