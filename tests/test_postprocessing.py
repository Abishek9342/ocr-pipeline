import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.engines import Detection
from ocr_resilience.postprocessing import (
    dedupe_detections,
    filter_low_confidence,
    normalize_unicode,
    normalize_whitespace,
    postprocess_text,
)


def test_normalize_whitespace_collapses_runs_and_trims_lines():
    assert normalize_whitespace("Hello    World  \n  Second   Line ") == "Hello World\nSecond Line"


def test_normalize_whitespace_preserves_intentional_line_breaks():
    assert normalize_whitespace("Line One\nLine Two\nLine Three") == "Line One\nLine Two\nLine Three"


def test_normalize_unicode_composes_combining_accents():
    decomposed = "é"  # 'e' + combining acute accent
    assert normalize_unicode(decomposed) == "é"  # precomposed 'é'


def test_postprocess_text_chains_unicode_then_whitespace():
    assert postprocess_text("é    World  ") == "é World"


def test_filter_low_confidence_is_noop_at_default_threshold():
    dets = [Detection("a", 0.1, (0, 0, 1, 1), "x")]
    assert filter_low_confidence(dets) == dets


def test_filter_low_confidence_drops_below_threshold():
    dets = [Detection("keep", 0.9, (0, 0, 1, 1), "x"), Detection("drop", 0.1, (0, 0, 1, 1), "x")]
    result = filter_low_confidence(dets, min_confidence=0.5)
    assert [d.text for d in result] == ["keep"]


def test_dedupe_detections_removes_same_text_overlapping_box():
    dets = [
        Detection("Hello", 0.9, (0, 0, 100, 40), "a"),
        Detection("Hello", 0.8, (2, 1, 98, 39), "b"),  # near-identical box, same text
    ]
    result = dedupe_detections(dets)
    assert len(result) == 1


def test_dedupe_detections_keeps_distinct_text_or_disjoint_boxes():
    dets = [
        Detection("Hello", 0.9, (0, 0, 100, 40), "a"),
        Detection("World", 0.9, (0, 0, 100, 40), "b"),
        Detection("Hello", 0.9, (500, 500, 600, 540), "c"),
    ]
    result = dedupe_detections(dets)
    assert len(result) == 3
