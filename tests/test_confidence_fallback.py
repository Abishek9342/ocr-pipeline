"""Tests for OCRPipeline.run()'s min_confidence_for_fallback: a genuine
second OCR attempt (escalating to every available engine) triggered by
low first-pass confidence, not a preprocessing retry. Priority 10 from the
mission doc ("implement confidence-based fallback") — kept off by default
(None) so it costs nothing for callers who don't ask for it."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.engines import Detection
from ocr_resilience.pipeline import OCRPipeline

IMAGE = np.full((50, 200, 3), 255, dtype=np.uint8)


class _FixedEngine:
    def __init__(self, name: str, text: str, confidence: float):
        self.name = name
        self._text = text
        self._confidence = confidence

    def recognize(self, image):
        return [Detection(self._text, self._confidence, (0, 0, 10, 10), self.name)]


def test_fallback_disabled_by_default_stays_on_single_engine_result():
    pipeline = OCRPipeline(engines={
        "low": _FixedEngine("low", "weak guess", 0.1),
        "high": _FixedEngine("high", "correct text", 0.95),
    })
    result = pipeline.run(IMAGE, force_ensemble=False)  # single engine, no fallback param passed
    assert result.raw_text == "weak guess"
    assert "fallback" not in result.routing.reason


def test_fallback_triggers_and_improves_result():
    pipeline = OCRPipeline(engines={
        "low": _FixedEngine("low", "weak guess", 0.1),
        "high": _FixedEngine("high", "correct text", 0.95),
    })
    result = pipeline.run(IMAGE, force_ensemble=False, min_confidence_for_fallback=0.5)

    assert "confidence-based fallback" in result.routing.reason
    assert result.engine_used == "high+low"  # both engines ran in the escalated pass
    assert result.confidence > 0.1


def test_fallback_triggers_but_does_not_improve_keeps_first_pass():
    # Both engines low-confidence and disagreeing: escalating can't raise
    # the mean confidence above the first (single-engine) pass's.
    pipeline = OCRPipeline(engines={
        "low": _FixedEngine("low", "weak guess", 0.2),
        "also_low": _FixedEngine("also_low", "different weak guess", 0.15),
    })
    result = pipeline.run(IMAGE, force_ensemble=False, min_confidence_for_fallback=0.9)

    assert "none improved" in result.routing.reason
    assert result.raw_text == "weak guess"  # kept the cheaper first-pass result


def test_fallback_does_not_trigger_above_threshold():
    pipeline = OCRPipeline(engines={
        "low": _FixedEngine("low", "confident text", 0.99),
        "high": _FixedEngine("high", "other text", 0.5),
    })
    result = pipeline.run(IMAGE, force_ensemble=False, min_confidence_for_fallback=0.5)
    assert "fallback" not in result.routing.reason


def test_fallback_does_not_trigger_with_only_one_engine_available():
    pipeline = OCRPipeline(engines={"only": _FixedEngine("only", "text", 0.01)})
    result = pipeline.run(IMAGE, force_ensemble=False, min_confidence_for_fallback=0.9)
    assert "fallback" not in result.routing.reason


def test_fallback_with_three_engines_tries_ranked_tier_before_full_ensemble():
    """With 3+ engines, tier 1 should add only the single best-ranked
    fallback engine (per engine_selection.OVERALL_ENGINE_RANKING) rather
    than jumping straight to all three — the actual new capability this
    ranked-fallback-chain feature adds over the old "primary -> everything"
    behavior."""
    pipeline = OCRPipeline(engines={
        "tesseract": _FixedEngine("tesseract", "weak guess", 0.1),  # primary (first-registered)
        "paddleocr": _FixedEngine("paddleocr", "correct text", 0.97),  # top of OVERALL_ENGINE_RANKING
        "rapidocr": _FixedEngine("rapidocr", "also correct", 0.5),
    })
    result = pipeline.run(IMAGE, force_ensemble=False, min_confidence_for_fallback=0.5)

    assert "tier 1" in result.routing.reason
    assert "rapidocr" not in result.engine_used  # tier 1 resolved it; tier 2 (all three) never had to run
    assert set(result.engine_used.split("+")) == {"tesseract", "paddleocr"}


def test_fallback_with_three_engines_escalates_to_tier_two_when_tier_one_insufficient():
    pipeline = OCRPipeline(engines={
        "tesseract": _FixedEngine("tesseract", "weak guess", 0.1),
        "paddleocr": _FixedEngine("paddleocr", "also weak", 0.2),
        "rapidocr": _FixedEngine("rapidocr", "the real answer", 0.9),
    })
    result = pipeline.run(IMAGE, force_ensemble=False, min_confidence_for_fallback=0.85)

    assert "tier 2" in result.routing.reason
    assert set(result.engine_used.split("+")) == {"tesseract", "paddleocr", "rapidocr"}
