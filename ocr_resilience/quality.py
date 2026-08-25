"""Classical computer-vision image-quality assessment — no ML model, no
learned weights. Every metric here is a well-known, decades-old CV
technique (Laplacian-variance blur detection, Hough-line skew estimation,
CLAHE-adjacent contrast statistics). The point isn't novelty; it's giving
the router (see router.py) enough signal to pick the RIGHT preprocessing
and the RIGHT engine for a given image, instead of running everything on
everything.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityReport:
    blur_score: float          # variance of Laplacian — LOWER means blurrier
    noise_score: float         # estimated noise std-dev
    impulse_noise_score: float  # fraction of pixels sharply off their local median — see impulse_noise_score()
    contrast_score: float      # std-dev of pixel intensities — LOWER means flatter/washed-out
    brightness: float          # mean pixel intensity, 0-255
    skew_angle_deg: float      # estimated rotation needed to deskew, degrees
    is_blurry: bool
    is_noisy: bool
    is_impulse_noisy: bool     # salt-and-pepper-like — a DIFFERENT signal from is_noisy, see impulse_noise_score()
    is_low_contrast: bool
    is_skewed: bool
    likely_handwritten: bool   # stroke-width-variance heuristic, see _stroke_width_variance


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard, widely-cited no-reference
    blur metric (Pech-Pacheco et al., 2000). Sharp edges produce high
    variance; blur smooths them out, lowering it."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def noise_score(gray: np.ndarray) -> float:
    """Fast noise estimate via the median-absolute-deviation of the
    high-frequency (Laplacian) response — a standard robust noise-sigma
    estimator, insensitive to the occasional genuine edge."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    mad = np.median(np.abs(lap - np.median(lap)))
    return float(mad * 1.4826)  # 1.4826 converts MAD to a Gaussian-equivalent sigma


def contrast_score(gray: np.ndarray) -> float:
    return float(gray.std())


def impulse_noise_score(gray: np.ndarray) -> float:
    """Fraction of pixels that differ sharply from their local 3x3 median —
    the standard impulse/"salt-and-pepper" noise detector (the basis of an
    adaptive median filter): an isolated corrupted pixel sticks out hard
    against its immediate neighborhood, while Gaussian noise or blur
    changes pixels more gradually and doesn't produce that.

    This exists as a SEPARATE signal from `noise_score` on purpose:
    `noise_score` is a median-absolute-deviation (MAD) estimator, which is
    a *robust* statistic specifically BECAUSE it's insensitive to sparse
    outliers — real salt-and-pepper noise (found via the benchmark to make
    Tesseract's segmentation return completely empty output, 1.0 CER, with
    zero preprocessing triggered) never registered as "noisy" under
    `noise_score`/`is_noisy`, precisely because that heuristic's robustness
    property is what made it blind to exactly this noise type. The two
    detectors are complementary, not redundant."""
    median = cv2.medianBlur(gray, 3)
    diff = np.abs(gray.astype(np.int16) - median.astype(np.int16))
    return float(np.mean(diff > 40))


def skew_angle(gray: np.ndarray) -> float:
    """Estimate rotation via the minimum-area bounding rectangle of all
    foreground (text) pixels after Otsu binarization — the standard
    projection-free deskew estimator for document images. Returns degrees;
    positive = rotate counter-clockwise to correct."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 20:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect returns an angle in [-90, 0); normalize to a small
    # correction (documents are rarely rotated more than ~45 degrees, and
    # a raw angle near -90 usually means the rect just latched onto the
    # image's own long axis, not real skew).
    if angle < -45:
        angle = 90 + angle
    # NOT `-angle` — verified empirically via the benchmark harness against
    # a known-correct reference rotation (benchmark/degrade.py's own
    # `rotate()`, used in reverse to build the test case): the sign
    # in the docstring above ("positive = counter-clockwise") already
    # matches cv2.getRotationMatrix2D's own convention, so re-negating
    # here was flipping deskew() to always rotate the WRONG way — it was
    # actively increasing skew instead of correcting it, and had been
    # silently doing so since this function was first written (only
    # caught because the benchmark showed the "skewed" preset getting
    # WORSE after "correction," not because of a code review).
    return float(angle)


def stroke_width_variance(gray: np.ndarray) -> float:
    """Cheap handwriting-vs-print heuristic: printed fonts have a narrow,
    consistent stroke width across all characters; handwriting's stroke
    width varies a lot more (pen pressure, cursive joins, inconsistent
    pen lifts). Approximated via the distance transform's coefficient of
    variation on foreground pixels — a lightweight stand-in for the full
    Stroke Width Transform (Epshtein et al., 2010) without needing its
    edge-ray-casting machinery."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    fg = dist[dist > 0]
    if len(fg) < 20 or fg.mean() == 0:
        return 0.0
    return float(fg.std() / fg.mean())


def assess(image: np.ndarray) -> QualityReport:
    gray = _to_gray(image)
    blur = blur_score(gray)
    noise = noise_score(gray)
    impulse = impulse_noise_score(gray)
    contrast = contrast_score(gray)
    brightness = float(gray.mean())
    skew = skew_angle(gray)
    swv = stroke_width_variance(gray)

    return QualityReport(
        blur_score=blur, noise_score=noise, impulse_noise_score=impulse, contrast_score=contrast,
        brightness=brightness, skew_angle_deg=skew,
        is_blurry=blur < 100.0,
        is_noisy=noise > 8.0,
        is_impulse_noisy=impulse > 0.015,
        is_low_contrast=contrast < 40.0,
        is_skewed=abs(skew) > 1.0,
        likely_handwritten=swv > 0.55,
    )
