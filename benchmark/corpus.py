"""Renders a small labeled text-image corpus via PIL — since no real
scanned/handwritten sample images are available in this environment, every
image here is generated from known ground-truth text, so CER/WER against
that ground truth is exact, not estimated. Two font styles: a standard
printed font, and a cursive script font (Windows' bundled Lucida
Handwriting) as an HONEST PROXY for handwriting — real handwritten strokes
are far more irregular than any font can produce; see benchmark/degrade.py's
module docstring for the same caveat.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SENTENCES = [
    "Hello World 12345",
    "Invoice Number INV-8452",
    "The quick brown fox jumps",
    "Payment due on 15th June",
    "Account Balance Rs 45,231.90",
    "Please sign below the line",
    "Reference Code AB7729XZ",
    "Total Amount Payable 9876",
    "Customer Name John Smith",
    "Branch Code CHR-GEN-042",
]

PRINTED_FONT_CANDIDATES = ["arial.ttf", "calibri.ttf", "tahoma.ttf"]
HANDWRITTEN_FONT_CANDIDATES = [r"C:\Windows\Fonts\LHANDW.TTF", "LHANDW.TTF"]


def _load_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_text_image(text: str, handwritten: bool = False, width: int = 700, height: int = 120) -> np.ndarray:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _load_font(HANDWRITTEN_FONT_CANDIDATES if handwritten else PRINTED_FONT_CANDIDATES, size=40 if handwritten else 34)
    draw.text((20, height // 2 - 25), text, fill=(10, 10, 10), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def build_corpus(out_dir: str) -> list[dict]:
    """Returns a list of {path, ground_truth, style} and writes each clean
    (pre-degradation) rendered image to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, text in enumerate(SENTENCES):
        for style, handwritten in [("printed", False), ("handwritten_proxy", True)]:
            img = render_text_image(text, handwritten=handwritten)
            filename = f"{i:02d}_{style}.png"
            path = os.path.join(out_dir, filename)
            cv2.imwrite(path, img)
            manifest.append({"path": path, "ground_truth": text, "style": style})
    return manifest
