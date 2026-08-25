import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.debug import annotate, export_debug_bundle
from ocr_resilience.engines import AVAILABLE_ENGINES, Detection
from ocr_resilience.pipeline import OCRPipeline


def test_annotate_does_not_mutate_input():
    image = np.full((50, 50, 3), 255, dtype=np.uint8)
    original = image.copy()
    annotate(image, [Detection("x", 0.9, (5, 5, 20, 20), "fake")])
    assert (image == original).all()


def test_annotate_draws_something_where_the_box_is():
    image = np.full((50, 50, 3), 255, dtype=np.uint8)
    annotated = annotate(image, [Detection("x", 0.9, (5, 5, 20, 20), "fake")])
    assert not (annotated == 255).all()  # something was drawn


def test_export_debug_bundle_writes_three_files(tmp_path):
    original = np.full((50, 50, 3), 255, dtype=np.uint8)
    preprocessed = np.full((50, 50), 200, dtype=np.uint8)
    detections = [Detection("x", 0.9, (5, 5, 20, 20), "fake")]

    paths = export_debug_bundle(original, preprocessed, detections, str(tmp_path))

    assert set(paths) == {"original", "preprocessed", "annotated"}
    for path in paths.values():
        assert Path(path).exists()
        assert cv2.imread(path) is not None


class _FakeEngine:
    name = "fake"

    def recognize(self, image):
        return [Detection("Fake Text", 0.9, (0, 0, 10, 10), "fake")]


def test_pipeline_run_writes_debug_bundle_when_requested(tmp_path, monkeypatch):
    monkeypatch.setitem(AVAILABLE_ENGINES, "fake", _FakeEngine)
    pipeline = OCRPipeline.with_engines(["fake"])
    image = np.full((50, 200, 3), 255, dtype=np.uint8)

    debug_dir = tmp_path / "debug"
    pipeline.run(image, debug_dir=str(debug_dir))

    assert (debug_dir / "original.png").exists()
    assert (debug_dir / "preprocessed.png").exists()
    assert (debug_dir / "annotated.png").exists()


def test_pipeline_run_skips_debug_export_by_default(tmp_path, monkeypatch):
    monkeypatch.setitem(AVAILABLE_ENGINES, "fake", _FakeEngine)
    pipeline = OCRPipeline.with_engines(["fake"])
    image = np.full((50, 200, 3), 255, dtype=np.uint8)

    pipeline.run(image)  # no debug_dir

    assert list(tmp_path.iterdir()) == []  # nothing written anywhere
