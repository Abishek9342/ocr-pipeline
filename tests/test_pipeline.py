"""Regression tests for pipeline.py, plus the edge-case audit run before
publishing (blank images, missing engines, missing files) — none of these
found bugs when tested, which is worth confirming stays true over time."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.engines import AVAILABLE_ENGINES, Detection
from ocr_resilience.pipeline import OCRPipeline, OCRResult, _reading_order, _resolve_engine_names
from ocr_resilience.quality import QualityReport
from ocr_resilience.router import RoutingDecision


def test_auto_engine_resolution_priority_list_covers_every_registered_engine():
    """Regression: `_ENGINE_PRIORITY` (used by `engine="auto"`, the
    documented default `OCR()` usage) previously omitted RapidOCR
    entirely — added as a fourth engine in `AVAILABLE_ENGINES` without
    updating this separate, easy-to-forget list, so `OCR()` would never
    use it even if installed, silently. Checked structurally (every
    registered engine must appear in the priority list) so a future
    fifth engine can't repeat this by omission."""
    from ocr_resilience.pipeline import _ENGINE_PRIORITY
    assert set(_ENGINE_PRIORITY) == set(AVAILABLE_ENGINES)


def test_resolve_engine_names_auto_includes_rapidocr_when_importable(monkeypatch):
    import importlib.util as importlib_util

    monkeypatch.setattr(importlib_util, "find_spec", lambda name: object())  # pretend every binding is installed
    assert "rapidocr" in _resolve_engine_names("auto")


def test_reading_order_sorts_scrambled_lines_top_to_bottom():
    """Nothing upstream (engine output order, fusion's union-find cluster
    discovery order) actually guarantees reading order — this was only
    passing incidentally in manual testing because a single engine's own
    output happened to already be raster-ordered. Verified directly with
    deliberately out-of-order input."""
    scrambled = [
        Detection("Line Four", 0.9, (20, 180, 200, 210), "x"),
        Detection("Line One", 0.9, (20, 20, 200, 50), "x"),
        Detection("Line Three", 0.9, (20, 140, 200, 170), "x"),
        Detection("Line Two", 0.9, (20, 80, 200, 110), "x"),
    ]
    ordered = _reading_order(scrambled)
    assert [d.text for d in ordered] == ["Line One", "Line Two", "Line Three", "Line Four"]


def test_reading_order_left_to_right_within_same_line():
    same_row = [
        Detection("World", 0.9, (100, 0, 180, 40), "x"),
        Detection("Hello", 0.9, (0, 0, 80, 40), "x"),
    ]
    ordered = _reading_order(same_row)
    assert [d.text for d in ordered] == ["Hello", "World"]


def test_reading_order_tolerates_per_word_baseline_variance_on_one_line():
    """The exact bug found live via the benchmark: a naive (y_min, x_min)
    global sort was tried first and shipped briefly — cursive/italic fonts
    give words on the SAME visual line noticeably different y_min values
    (ascenders/descenders), which split one line into two under a naive
    sort and scrambled "Payment due on 15th June" into "due on Payment
    15th June" (word order preserved within each fragment, but the
    fragments interleaved wrong). Caught because an UNDEGRADED test image
    regressed from ~0.04 CER to 0.83 CER the moment this function was
    added — a stark enough signal to investigate immediately rather than
    attribute it to font-rendering noise."""
    line = [
        Detection("Payment", 0.9, (24, 47, 228, 87), "tesseract"),   # y_min=47
        Detection("due", 0.9, (240, 38, 317, 92), "tesseract"),      # y_min=38 (taller ascender/descender)
        Detection("15th", 0.9, (411, 48, 515, 77), "tesseract"),     # y_min=48
        Detection("June", 0.9, (533, 38, 641, 92), "tesseract"),     # y_min=38
    ]
    ordered = _reading_order(line)
    assert [d.text for d in ordered] == ["Payment", "due", "15th", "June"]


def test_pipeline_with_no_engines_raises_clear_error():
    pipeline = OCRPipeline(engines={})
    with pytest.raises(ValueError, match="No OCR engines available"):
        pipeline.run(np.full((50, 50, 3), 255, dtype=np.uint8))


def test_pipeline_with_engines_rejects_unknown_engine_name():
    with pytest.raises(ValueError, match="Unknown engine"):
        OCRPipeline.with_engines(["not_a_real_engine"])


def test_pipeline_raises_on_missing_file():
    pipeline = OCRPipeline(engines={})
    with pytest.raises(ValueError, match="Could not load image"):
        pipeline.run("this_file_does_not_exist_anywhere.png")


def _dummy_result(detections: list[Detection]) -> OCRResult:
    quality = QualityReport(
        blur_score=500.0, noise_score=1.0, impulse_noise_score=0.0, contrast_score=80.0, brightness=200.0,
        skew_angle_deg=0.0, is_blurry=False, is_noisy=False, is_impulse_noisy=False, is_low_contrast=False,
        is_skewed=False, likely_handwritten=False,
    )
    routing = RoutingDecision(engines_to_run=["fake"], reason="test")
    return OCRResult(detections=detections, quality=quality, routing=routing, preprocessing_steps=[])


def test_result_text_joins_same_line_word_detections_with_space_not_newline():
    """Regression for a real bug: word-level detections (Tesseract's
    default, one box per word) on the SAME line used to be joined with
    "\\n" — turning "Hello World 12345" into three separate output lines
    instead of one space-joined line. Invisible in the benchmark because
    it bypasses `.text` entirely and space-joins `result.detections`
    itself; caught only by exercising the CLI end-to-end against a real
    engine, which uses `.text`/`raw_text` as its headline output."""
    words_on_one_line = [
        Detection("Hello", 0.96, (23, 41, 98, 66), "tesseract"),
        Detection("World", 0.96, (108, 41, 194, 66), "tesseract"),
        Detection("12345", 0.96, (210, 41, 300, 66), "tesseract"),
    ]
    result = _dummy_result(words_on_one_line)
    assert result.text == "Hello World 12345"
    assert result.raw_text == "Hello World 12345"


def test_result_text_still_separates_distinct_lines_with_newline():
    two_lines = [
        Detection("Second", 0.9, (0, 80, 100, 110), "tesseract"),
        Detection("Line", 0.9, (110, 80, 200, 110), "tesseract"),
        Detection("First", 0.9, (0, 0, 100, 30), "tesseract"),
        Detection("Line", 0.9, (110, 0, 200, 30), "tesseract"),
    ]
    result = _dummy_result(two_lines)
    assert result.text == "First Line\nSecond Line"
