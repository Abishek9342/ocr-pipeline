"""Pluggable OCR-engine adapters. Every adapter implements the same
`OCREngine` interface (`recognize(image) -> list[Detection]`), so the
router/fusion layers never know or care which underlying library actually
produced a result — this is a THIN wrapper layer, not a reimplementation of
any of these engines' own text-detection/recognition models.

Availability on this development machine (Windows): all three verified
working end-to-end against real rendered-text images — EasyOCRAdapter,
TesseractAdapter (binary installed via the UB-Mannheim Windows installer),
and PaddleOCRAdapter (initially blocked by a corporate Application Control
policy on one of its dependencies; resolved on the IT/policy side, not
worked around here). PaddleOCR's API changed substantially between the 2.x
line (`use_angle_cls`/`.ocr(cls=True)`) and 3.x (`use_textline_orientation`/
`.predict()` returning dict-like results) — this adapter targets 3.x,
verified against the actually-installed version rather than assumed from
tutorials, several of which still document the old 2.x calling convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    text: str
    confidence: float           # 0.0-1.0
    bbox: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max
    engine: str


class OCREngine(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> list[Detection]: ...


class EasyOCRAdapter:
    """Verified on this machine — see module docstring."""

    name = "easyocr"

    def __init__(self, languages: list[str] | None = None, gpu: bool = False):
        import easyocr  # deferred — avoids paying easyocr's model-load cost for adapters that aren't used
        self._reader = easyocr.Reader(languages or ["en"], gpu=gpu, verbose=False)

    def recognize(self, image: np.ndarray) -> list[Detection]:
        results = self._reader.readtext(image)
        out = []
        for bbox_points, text, conf in results:
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            out.append(Detection(
                text=text, confidence=float(conf),
                bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                engine=self.name,
            ))
        return out


_WINDOWS_DEFAULT_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


class TesseractAdapter:
    """`pip install pytesseract` only installs the Python wrapper — the
    actual `tesseract` binary is a separate native install (on Windows,
    the UB-Mannheim installer). If it's not on PATH yet (common right
    after installing, before a shell restart), fall back to the standard
    Windows install locations rather than failing outright."""

    name = "tesseract"

    def __init__(self, tesseract_cmd: str | None = None):
        import os
        import shutil

        import pytesseract
        self._pytesseract = pytesseract
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif not shutil.which("tesseract"):
            for path in _WINDOWS_DEFAULT_TESSERACT_PATHS:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

    def recognize(self, image: np.ndarray) -> list[Detection]:
        data = self._pytesseract.image_to_data(image, output_type=self._pytesseract.Output.DICT)
        out = []
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            conf = max(0.0, float(data["conf"][i])) / 100.0
            out.append(Detection(text=text, confidence=conf, bbox=(x, y, x + w, y + h), engine=self.name))
        return out


class PaddleOCRAdapter:
    """PaddleOCR 3.x replaced the old `use_angle_cls`/`.ocr(cls=True)` API
    (still documented in plenty of tutorials, including an earlier draft
    of this adapter) with `use_textline_orientation`/`.predict()`, and
    `.predict()` returns one dict-like result object per image (with
    `rec_texts`/`rec_scores`/`rec_polys` keys) instead of the old nested
    `[[bbox, (text, conf)], ...]` list structure. Verified against the
    actually-installed version, not assumed from documentation."""

    name = "paddleocr"

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image: np.ndarray) -> list[Detection]:
        results = self._ocr.predict(image)
        out = []
        for result in results:
            texts = result["rec_texts"]
            scores = result["rec_scores"]
            polys = result["rec_polys"]
            for text, score, poly in zip(texts, scores, polys):
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                out.append(Detection(
                    text=text, confidence=float(score),
                    bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                    engine=self.name,
                ))
        return out


AVAILABLE_ENGINES = {
    "easyocr": EasyOCRAdapter,
    "tesseract": TesseractAdapter,
    "paddleocr": PaddleOCRAdapter,
}
