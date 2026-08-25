"""Adapter-level contract tests. Each engine's *own* recognizer is mocked
at the module-import boundary (sys.modules) rather than actually invoked —
these tests aren't verifying EasyOCR/Tesseract/PaddleOCR themselves (that's
what benchmark/run_benchmark.py is for, against a real install); they're
verifying THIS package's adapter correctly reshapes each library's own
result format into `Detection`, without requiring the real binaries/models
to be installed wherever the test suite runs (e.g. CI)."""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience.engines import Detection


def test_tesseract_adapter_converts_image_to_data_rows_and_normalizes_confidence(monkeypatch):
    fake_pytesseract = types.ModuleType("pytesseract")
    fake_pytesseract.pytesseract = types.SimpleNamespace(tesseract_cmd="tesseract")
    fake_pytesseract.Output = types.SimpleNamespace(DICT="dict")

    def fake_image_to_data(image, output_type):
        return {
            "text": ["", "Hello", "World"],
            "left": [0, 10, 60],
            "top": [0, 5, 5],
            "width": [0, 40, 45],
            "height": [0, 20, 20],
            "conf": [-1, 95, 80],
        }

    fake_pytesseract.image_to_data = fake_image_to_data
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

    from ocr_resilience.engines import TesseractAdapter

    adapter = TesseractAdapter()
    detections = adapter.recognize(image=None)

    assert [d.text for d in detections] == ["Hello", "World"]  # blank entry skipped
    assert detections[0].confidence == 0.95
    assert detections[0].bbox == (10, 5, 50, 25)
    assert all(d.engine == "tesseract" for d in detections)


def test_tesseract_adapter_guards_negative_confidence_sentinel(monkeypatch):
    fake_pytesseract = types.ModuleType("pytesseract")
    fake_pytesseract.pytesseract = types.SimpleNamespace(tesseract_cmd="tesseract")
    fake_pytesseract.Output = types.SimpleNamespace(DICT="dict")
    fake_pytesseract.image_to_data = lambda image, output_type: {
        "text": ["X"], "left": [0], "top": [0], "width": [10], "height": [10], "conf": [-1],
    }
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

    from ocr_resilience.engines import TesseractAdapter

    detections = TesseractAdapter().recognize(image=None)
    assert detections[0].confidence == 0.0  # Tesseract's -1 "no confidence" sentinel must not go negative


def test_easyocr_adapter_converts_polygon_points_to_bbox(monkeypatch):
    fake_easyocr = types.ModuleType("easyocr")

    class FakeReader:
        def __init__(self, languages, gpu, verbose):
            pass

        def readtext(self, image):
            return [([[10, 20], [50, 20], [50, 40], [10, 40]], "Hi", 0.87)]

    fake_easyocr.Reader = FakeReader
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)

    from ocr_resilience.engines import EasyOCRAdapter

    detections = EasyOCRAdapter().recognize(image=None)
    assert detections == [Detection("Hi", 0.87, (10, 20, 50, 40), "easyocr")]


def test_paddleocr_adapter_converts_predict_result_dicts(monkeypatch):
    fake_paddleocr = types.ModuleType("paddleocr")
    captured_kwargs = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def predict(self, image):
            return [{
                "rec_texts": ["Total"],
                "rec_scores": [0.93],
                "rec_polys": [[[5, 5], [55, 5], [55, 25], [5, 25]]],
            }]

    fake_paddleocr.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    from ocr_resilience.engines import PaddleOCRAdapter

    detections = PaddleOCRAdapter().recognize(image=np.zeros((30, 60, 3), dtype=np.uint8))
    assert detections == [Detection("Total", 0.93, (5, 5, 55, 25), "paddleocr")]
    # PP-OCRv4 is the default, NOT the library's own default (v5/v6) — see
    # the adapter's docstring for the specific PIR bug this avoids.
    assert captured_kwargs["ocr_version"] == "PP-OCRv4"


def test_paddleocr_adapter_converts_grayscale_input_to_bgr(monkeypatch):
    """Regression: PaddleX's internal resize step does `h, w, _ = img.shape`,
    which crashes on a 2D (grayscale) array with 'not enough values to
    unpack (expected 3, got 2)'. Tesseract/EasyOCR both accept the shared
    pipeline's grayscale preprocessing output directly, so this only
    surfaced the first time all three engines actually ran together in one
    ensemble — caught via the full benchmark run, not code review."""
    fake_paddleocr = types.ModuleType("paddleocr")
    received_images = []

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            pass

        def predict(self, image):
            received_images.append(image)
            return []

    fake_paddleocr.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    from ocr_resilience.engines import PaddleOCRAdapter

    grayscale = np.full((40, 60), 200, dtype=np.uint8)
    PaddleOCRAdapter().recognize(grayscale)

    assert received_images[0].ndim == 3
    assert received_images[0].shape[2] == 3


def test_paddleocr_adapter_ocr_version_is_overridable(monkeypatch):
    fake_paddleocr = types.ModuleType("paddleocr")
    captured_kwargs = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def predict(self, image):
            return []

    fake_paddleocr.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    from ocr_resilience.engines import PaddleOCRAdapter

    PaddleOCRAdapter(ocr_version="PP-OCRv6")
    assert captured_kwargs["ocr_version"] == "PP-OCRv6"


def _fake_rapidocr_output(boxes, txts, scores):
    return types.SimpleNamespace(boxes=boxes, txts=txts, scores=scores)


def test_rapidocr_adapter_converts_output_dataclass(monkeypatch):
    fake_rapidocr = types.ModuleType("rapidocr")

    class FakeRapidOCR:
        def __init__(self, **kwargs):
            pass

        def __call__(self, image):
            return _fake_rapidocr_output(
                boxes=[[[5, 5], [55, 5], [55, 25], [5, 25]]],
                txts=("Total",),
                scores=(0.93,),
            )

    fake_rapidocr.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake_rapidocr)

    from ocr_resilience.engines import RapidOCRAdapter

    detections = RapidOCRAdapter().recognize(image=np.zeros((30, 60), dtype=np.uint8))
    assert detections == [Detection("Total", 0.93, (5, 5, 55, 25), "rapidocr")]


def test_rapidocr_adapter_returns_empty_list_when_no_text_detected(monkeypatch):
    fake_rapidocr = types.ModuleType("rapidocr")

    class FakeRapidOCR:
        def __init__(self, **kwargs):
            pass

        def __call__(self, image):
            return _fake_rapidocr_output(boxes=None, txts=(), scores=())

    fake_rapidocr.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake_rapidocr)

    from ocr_resilience.engines import RapidOCRAdapter

    assert RapidOCRAdapter().recognize(image=np.zeros((30, 60), dtype=np.uint8)) == []
