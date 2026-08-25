"""Invalid/corrupted-input handling: a truncated or non-image file on disk,
and a degenerate (tiny/blank) but validly-loaded array. Uses a fake
single-engine registration (see test_cli.py's approach) so these don't
depend on a real OCR install being present."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.engines import AVAILABLE_ENGINES, Detection
from ocr_resilience.pipeline import OCRPipeline


class _FakeEngine:
    name = "fake"

    def recognize(self, image):
        return [Detection("x", 0.9, (0, 0, 1, 1), "fake")] if image.size else []


@pytest.fixture(autouse=True)
def register_fake_engine(monkeypatch):
    monkeypatch.setitem(AVAILABLE_ENGINES, "fake", _FakeEngine)


@pytest.fixture
def pipeline() -> OCRPipeline:
    return OCRPipeline.with_engines(["fake"])


def test_truncated_file_with_image_extension_raises_clear_error(tmp_path, pipeline):
    bogus = tmp_path / "not_really_an_image.png"
    bogus.write_bytes(b"this is not a valid PNG file")
    with pytest.raises(ValueError, match="Could not load image"):
        pipeline.run(str(bogus))


def test_zero_byte_file_raises_clear_error(tmp_path, pipeline):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="Could not load image"):
        pipeline.run(str(empty))


def test_tiny_one_pixel_image_does_not_crash(pipeline):
    tiny = np.full((1, 1, 3), 128, dtype=np.uint8)
    result = pipeline.run(tiny)
    assert result.text == "" or isinstance(result.text, str)


def test_grayscale_single_channel_array_does_not_crash(pipeline):
    gray = np.full((50, 50), 200, dtype=np.uint8)
    result = pipeline.run(gray)
    assert isinstance(result.text, str)
