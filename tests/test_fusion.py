"""Regression tests for fusion.py — specifically the two clustering bugs
the benchmark harness caught (see fusion.py's docstrings for the full
story): plain-IoU failing on word-vs-line granularity mismatches, and
greedy first-match clustering failing to transitively merge a bridging
detection with multiple pre-existing groups.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.engines import Detection
from ocr_resilience.fusion import fuse, group_by_region, overlap_ratio


def test_overlap_ratio_full_containment_scores_high_despite_size_mismatch():
    # A small box fully inside a much larger one — standard IoU would
    # score this low (union dominated by the big box); overlap_ratio
    # (intersection / smaller area) should score it 1.0.
    small = (20, 20, 50, 50)
    large = (0, 0, 500, 100)
    assert overlap_ratio(small, large) == 1.0


def test_overlap_ratio_disjoint_boxes_score_zero():
    assert overlap_ratio((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_grouping_transitively_merges_bridging_detection():
    """The exact bug found via the benchmark: one line-level box (easyocr)
    overlaps three word-level boxes (tesseract) that DON'T overlap each
    other. All four must end up in ONE group, not three separate ones
    with the bridge attached to only the first."""
    word1 = Detection("Hello", 0.9, (0, 0, 80, 40), "tesseract")
    word2 = Detection("World", 0.9, (90, 0, 170, 40), "tesseract")
    word3 = Detection("12345", 0.9, (180, 0, 260, 40), "tesseract")
    line = Detection("Hello World 12345", 0.9, (0, 0, 260, 40), "easyocr")

    groups = group_by_region([word1, word2, word3, line])
    assert len(groups) == 1
    assert len(groups[0]) == 4


def test_fuse_reconstructs_full_line_from_fragments_plus_whole_hypothesis():
    word1 = Detection("Hello", 0.9, (0, 0, 80, 40), "tesseract")
    word2 = Detection("World", 0.9, (90, 0, 170, 40), "tesseract")
    word3 = Detection("12345", 0.9, (180, 0, 260, 40), "tesseract")
    line = Detection("Hello World 12345", 0.95, (0, 0, 260, 40), "easyocr")

    fused = fuse([word1, word2, word3, line])
    assert len(fused) == 1
    assert fused[0].text == "Hello World 12345"
    assert fused[0].engine == "easyocr+tesseract"


def test_fuse_single_engine_is_pass_through():
    dets = [Detection("Hello", 0.9, (0, 0, 80, 40), "tesseract")]
    fused = fuse(dets)
    assert fused == dets


def test_fuse_disagreement_prefers_higher_confidence_engine():
    correct = Detection("World", 0.95, (0, 0, 80, 40), "engine_a")
    wrong = Detection("Word", 0.2, (5, 0, 75, 40), "engine_b")
    fused = fuse([correct, wrong])
    assert len(fused) == 1
    assert fused[0].text == "World"
