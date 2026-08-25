"""Synthetic degradation functions for building a labeled OCR benchmark.
No real scanned/handwritten samples are available in this environment, so
the benchmark instead renders clean text with known ground truth (via
PIL), then applies a controlled degradation — this is the same "build the
ground truth by construction" strategy used throughout this portfolio's
other datasets, and it's the only way to get exact per-image ground truth
without a labeled real-world corpus (e.g. IAM handwriting, which requires
registration this environment can't complete).

Honesty note: font-rendered "handwriting" (via a cursive/script TrueType
font) is a proxy for real handwriting, not a substitute for it — real
handwritten samples have far more irregular stroke geometry than any font
can produce. Treat handwriting results here as a lower bound on how hard
real handwriting would be, not an equivalent test.
"""
import random

import cv2
import numpy as np


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def motion_blur(image: np.ndarray, kernel_size: int = 9, angle_deg: float = 0.0) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    rot = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (kernel_size, kernel_size))
    kernel = kernel / kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def _numpy_rng(rng: random.Random | None) -> np.random.Generator:
    """Derive a numpy Generator from the stdlib `random.Random` every other
    degradation function here already takes, so noise generation is
    ALSO seed-reproducible via `apply_degradation(..., seed=...)` — a
    real bug found via `benchmark.run_robustness`: this function used to
    call `np.random.normal`/`np.random.randint` directly, which reads
    numpy's own GLOBAL random state and ignores the `rng` parameter
    entirely. `apply_degradation` passes a fresh `random.Random(seed)`
    per call, so every OTHER degradation (blur sigma is fixed; motion
    blur/skew/smudge/combo_hard all consume `rng.uniform`/`rng.randint`)
    was correctly seed-reproducible — but `noisy`, `salt_pepper`, and
    `combo_hard` (which chains gaussian_noise) were NOT, despite
    `stable_seed()`'s entire purpose being reproducibility. Confirmed
    directly: `apply_degradation(img, "noisy", seed=42)` called twice
    produced two DIFFERENT images."""
    seed = rng.getrandbits(32) if rng is not None else None
    return np.random.default_rng(seed)


def gaussian_noise(image: np.ndarray, sigma: float, rng: random.Random | None = None) -> np.ndarray:
    noise = _numpy_rng(rng).normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def salt_and_pepper(image: np.ndarray, amount: float = 0.02, rng: random.Random | None = None) -> np.ndarray:
    generator = _numpy_rng(rng)
    out = image.copy()
    n_pixels = int(amount * image.size)
    coords = [generator.integers(0, dim, n_pixels) for dim in image.shape[:2]]
    out[tuple(coords)] = 255
    coords = [generator.integers(0, dim, n_pixels) for dim in image.shape[:2]]
    out[tuple(coords)] = 0
    return out


def rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255), flags=cv2.INTER_CUBIC)


def low_contrast(image: np.ndarray, factor: float = 0.4) -> np.ndarray:
    mean = image.mean()
    return np.clip((image.astype(np.float64) - mean) * factor + mean, 0, 255).astype(np.uint8)


def smudge(image: np.ndarray, n_blobs: int = 3, rng: random.Random | None = None) -> np.ndarray:
    rng = rng or random.Random()
    out = image.copy()
    h, w = image.shape[:2]
    for _ in range(n_blobs):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        radius = rng.randint(15, 40)
        color = rng.randint(120, 180)
        overlay = out.copy()
        cv2.circle(overlay, (cx, cy), radius, (color, color, color), thickness=-1)
        out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
    return out


def jpeg_artifact(image: np.ndarray, quality: int = 15) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image


DEGRADATION_PRESETS = {
    "clean": lambda img, rng: img,
    "light_blur": lambda img, rng: gaussian_blur(img, sigma=1.2),
    "heavy_blur": lambda img, rng: gaussian_blur(img, sigma=3.0),
    "motion_blur": lambda img, rng: motion_blur(img, kernel_size=11, angle_deg=rng.uniform(0, 45)),
    "noisy": lambda img, rng: gaussian_noise(img, sigma=25, rng=rng),
    "salt_pepper": lambda img, rng: salt_and_pepper(img, amount=0.03, rng=rng),
    "skewed": lambda img, rng: rotate(img, angle_deg=rng.uniform(-8, 8)),
    "low_contrast": lambda img, rng: low_contrast(img, factor=0.35),
    "smudged": lambda img, rng: smudge(img, n_blobs=3, rng=rng),
    "jpeg_compressed": lambda img, rng: jpeg_artifact(img, quality=12),
    "combo_hard": lambda img, rng: smudge(
        gaussian_noise(gaussian_blur(rotate(img, angle_deg=rng.uniform(-5, 5)), sigma=1.5), sigma=15, rng=rng),
        n_blobs=2, rng=rng,
    ),
}


def apply_degradation(image: np.ndarray, preset: str, seed: int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    return DEGRADATION_PRESETS[preset](image, rng)
