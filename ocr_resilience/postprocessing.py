"""Post-processing over fused OCR output. Every function here is text-level
only — none of it touches pixels or re-runs an engine. The raw fused text
(exactly what the engines/fusion layer produced) is always preserved
alongside the processed text; nothing here overwrites it, per the "don't
aggressively modify text without preserving the raw OCR output" requirement.
"""
from __future__ import annotations

import re
import unicodedata

from .engines import Detection


def normalize_unicode(text: str) -> str:
    """NFC-normalize so visually-identical characters (e.g. a composed vs.
    combining-accent form of the same letter) compare equal downstream."""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace to a single space and strip
    trailing whitespace per line, without touching intentional line breaks
    (those encode real line structure from `_reading_order`, not noise)."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def postprocess_text(text: str) -> str:
    """The default text post-processing chain: unicode normalization, then
    whitespace normalization. Order matters — normalizing unicode first
    means whitespace-adjacent combining characters don't confuse the
    whitespace regex."""
    return normalize_whitespace(normalize_unicode(text))


def filter_low_confidence(detections: list[Detection], min_confidence: float = 0.0) -> list[Detection]:
    """Drop detections below a confidence floor. Default threshold of 0.0
    is a no-op — this is opt-in, not applied automatically by the pipeline,
    since what counts as "too low to trust" is task-dependent."""
    if min_confidence <= 0.0:
        return detections
    return [d for d in detections if d.confidence >= min_confidence]


def dedupe_detections(detections: list[Detection], iou_threshold: float = 0.9) -> list[Detection]:
    """Drop exact-duplicate (same text, near-identical box) detections.
    This is a safety net for a caller feeding pre-fused detections from
    multiple sources — `fusion.fuse()` already prevents duplicates for the
    pipeline's own multi-engine path, so this rarely fires on pipeline
    output, but a public function accepting arbitrary `list[Detection]`
    (e.g. hand-assembled by a downstream user) shouldn't assume that."""
    from .fusion import overlap_ratio

    kept: list[Detection] = []
    for det in detections:
        is_duplicate = any(
            det.text == other.text and overlap_ratio(det.bbox, other.bbox) >= iou_threshold
            for other in kept
        )
        if not is_duplicate:
            kept.append(det)
    return kept
