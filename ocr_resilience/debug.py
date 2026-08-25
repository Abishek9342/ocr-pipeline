"""Visual debugging export (mission section 34): dump the original image,
the preprocessed image, and an annotated (detected-region) image side by
side, so a failure can be inspected visually instead of only as text/CER
numbers. Every bug found in this project so far was diagnosed via text
comparison alone — this exists for the NEXT one, where a picture might
actually be faster than reading a diff.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .engines import Detection


def annotate(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Original image with each detection's bounding box and text drawn
    on top — a copy, the input is never mutated."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
        label = f"{det.text[:20]} ({det.confidence:.2f})"
        cv2.putText(out, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def export_debug_bundle(
    original: np.ndarray,
    preprocessed: np.ndarray,
    detections: list[Detection],
    out_dir: str,
) -> dict[str, str]:
    """Writes original.png, preprocessed.png, annotated.png to `out_dir`
    (created if needed) and returns their paths. `annotated.png` draws
    every detection's box + text + confidence onto a copy of the
    ORIGINAL image (not the preprocessed one), so it's directly
    comparable to what a human looking at the source document would see."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "original": os.path.join(out_dir, "original.png"),
        "preprocessed": os.path.join(out_dir, "preprocessed.png"),
        "annotated": os.path.join(out_dir, "annotated.png"),
    }
    cv2.imwrite(paths["original"], original)
    cv2.imwrite(paths["preprocessed"], preprocessed)
    cv2.imwrite(paths["annotated"], annotate(original, detections))
    return paths
