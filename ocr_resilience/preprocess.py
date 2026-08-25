"""Classical (non-learned) preprocessing operators for degraded document
images. Each function is a single, well-established computer-vision
technique — no trained model anywhere in this file. `build_pipeline()`
composes only the operators the QualityReport says are actually needed,
so a clean image passes through mostly untouched instead of being
needlessly blurred/sharpened/binarized into a worse state than it started.
"""
import cv2
import numpy as np

from .quality import QualityReport, assess


def deskew(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.3:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Non-local means — the standard denoiser for scan/photo noise,
    chosen over a simple Gaussian blur because it smooths flat regions
    while preserving character edges (a Gaussian blur would soften both
    equally, actively hurting downstream OCR)."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def median_denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filtering — the standard, specific fix for salt-and-pepper
    (impulse) noise, gated on `QualityReport.is_impulse_noisy` rather than
    `is_noisy`/`denoise()` above: the benchmark found Tesseract returns
    completely EMPTY output (1.0 CER) on salt-and-pepper-degraded images,
    and that `noise_score`'s MAD-based estimator never flags that case as
    noisy at all (a robust statistic is, by design, insensitive to the
    sparse outlier pixels salt-and-pepper noise actually is — see
    `quality.impulse_noise_score`'s docstring). Non-local means (`denoise`)
    is tuned for the gradual, spatially-correlated noise it was built for,
    not isolated single-pixel corruption; median filtering directly
    targets that instead."""
    return cv2.medianBlur(gray, ksize)


def deblur_unsharp(gray: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Unsharp masking: subtract a blurred copy from the original to boost
    high frequencies. A real blur kernel is usually unknown (motion vs.
    defocus vs. compression), so a full deconvolution isn't well-posed
    without that estimate — unsharp masking is the standard blur-agnostic
    fallback that recovers usable edge contrast without needing one."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) — unlike
    global histogram equalization, this adapts per-tile, so a document
    with uneven lighting (a phone photo with a shadow across half the
    page) gets corrected locally instead of the shadowed half staying
    dark while the lit half blows out."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def remove_smudges(gray: np.ndarray) -> np.ndarray:
    """Detects large, low-gradient dark blobs (smudges/stains — visually
    distinct from thin, high-gradient text strokes) via morphological
    opening with a kernel too large for individual characters to survive,
    then inpaints just those regions from their surroundings. Leaves
    actual text untouched because no single character is big enough to
    pass the size filter."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for c in contours:
        if cv2.contourArea(c) > 600:  # bigger than any plausible single character
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    if not mask.any():
        return gray
    return cv2.inpaint(gray, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def binarize_sauvola(gray: np.ndarray, window: int = 25, k: float = 0.2) -> np.ndarray:
    """Sauvola adaptive thresholding — the standard document-image
    binarizer in the DIBCO benchmark literature specifically because it
    handles uneven illumination and background staining far better than
    a single global (Otsu) threshold: the threshold at each pixel adapts
    to the LOCAL mean and std-dev of its neighborhood."""
    gray_f = gray.astype(np.float64)
    mean = cv2.boxFilter(gray_f, cv2.CV_64F, (window, window))
    sq_mean = cv2.boxFilter(gray_f * gray_f, cv2.CV_64F, (window, window))
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
    threshold = mean * (1 + k * (std / 128.0 - 1))
    binary = np.where(gray_f > threshold, 255, 0).astype(np.uint8)
    return binary


def build_pipeline(image: np.ndarray, report: QualityReport | None = None) -> tuple[np.ndarray, list[str]]:
    """Apply only the operators the quality report says are needed, in a
    principled order: geometry first (deskew), then denoise/deblur BEFORE
    contrast enhancement (that amplifies whatever noise is still present).
    Returns (processed_image, steps_applied) — the log matters for the
    benchmark harness to explain *why* a given result differs from
    running the raw image through an engine directly.

    Sauvola binarization is deliberately NOT part of this default chain.
    An ablation in benchmark/run_benchmark.py found it actively hurts both
    Tesseract's and EasyOCR's modern LSTM/CRNN recognizers — e.g. on a
    heavy-blur test case, deblurring alone got both engines to 0.000 CER,
    but binarizing on top of that pushed EasyOCR's CER to 0.235 (worse
    than doing NO preprocessing at all, 0.118). Binarization is a
    template-matching-era technique; these engines are trained on natural
    grayscale/color images and their own internal binarization (if any)
    is already tuned to their own feature extractor — a second, external,
    untuned threshold just destroys information they could have used. Call
    `binarize_sauvola()` directly if you've verified it helps YOUR engine/
    data — don't assume it does."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if report is None:
        report = assess(image)
    steps = []

    if report.is_skewed:
        gray = deskew(gray, report.skew_angle_deg)
        steps.append(f"deskew({report.skew_angle_deg:.1f}deg)")

    if report.is_impulse_noisy:
        gray = median_denoise(gray)
        steps.append("median_denoise")

    if report.is_noisy:
        gray = denoise(gray)
        steps.append("denoise")

    if report.is_blurry:
        gray = deblur_unsharp(gray)
        steps.append("deblur_unsharp")

    gray = remove_smudges(gray)
    steps.append("remove_smudges")

    if report.is_low_contrast:
        gray = enhance_contrast(gray)
        steps.append("enhance_contrast")

    return gray, steps


_SINGLE_STEP_FUNCS = {
    "deskew": lambda gray, report: deskew(gray, report.skew_angle_deg),
    "denoise": lambda gray, report: denoise(gray),
    "median_denoise": lambda gray, report: median_denoise(gray),
    "deblur_unsharp": lambda gray, report: deblur_unsharp(gray),
    "enhance_contrast": lambda gray, report: enhance_contrast(gray),
    "remove_smudges": lambda gray, report: remove_smudges(gray),
    "binarize_sauvola": lambda gray, report: binarize_sauvola(gray),
}


def apply_single_step(image: np.ndarray, step: str, report: QualityReport | None = None) -> tuple[np.ndarray, list[str]]:
    """Force exactly one preprocessing operator, unconditionally — bypasses
    `build_pipeline`'s quality-gated selection entirely. This exists for
    ablation experiments (`benchmark/run_ablation.py`) that need to isolate
    one component's effect (e.g. "does deskew alone help?") in controlled
    comparison; it is not part of the adaptive default chain and skipping
    the quality gate here is deliberate, not an oversight."""
    if step not in _SINGLE_STEP_FUNCS:
        raise ValueError(f"Unknown preprocessing step '{step}'. Available: {list(_SINGLE_STEP_FUNCS)}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if report is None:
        report = assess(image)
    gray = _SINGLE_STEP_FUNCS[step](gray, report)
    return gray, [step]
