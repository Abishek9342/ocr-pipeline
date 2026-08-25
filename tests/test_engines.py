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

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            pass

        def predict(self, image):
            return [{
                "rec_texts": ["Total"],
                "rec_scores": [0.93],
                "rec_polys": [[[5, 5], [55, 5], [55, 25], [5, 25]]],
            }]

    fake_paddleocr.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    from ocr_resilience.engines import PaddleOCRAdapter

    detections = PaddleOCRAdapter().recognize(image=None)
    assert detections == [Detection("Total", 0.93, (5, 5, 55, 25), "paddleocr")]
