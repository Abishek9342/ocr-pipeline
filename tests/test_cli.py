"""CLI tests against a fake registered engine — exercising argument
parsing, single-file/batch/directory handling, JSON output shape, and
partial-failure behavior, without requiring a real OCR engine install."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_resilience import cli
from ocr_resilience.engines import AVAILABLE_ENGINES, Detection


class _FakeEngine:
    name = "fake"

    def recognize(self, image):
        return [Detection("Fake Text", 0.9, (0, 0, 10, 10), "fake")]


@pytest.fixture(autouse=True)
def register_fake_engine(monkeypatch):
    monkeypatch.setitem(AVAILABLE_ENGINES, "fake", _FakeEngine)


@pytest.fixture
def sample_image(tmp_path) -> Path:
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), np.full((50, 200, 3), 255, dtype=np.uint8))
    return path


def test_cli_single_file_prints_json_to_stdout(sample_image, capsys):
    exit_code = cli.main([str(sample_image), "--engine", "fake"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["raw_text"] == "Fake Text"
    assert payload["engine_used"] == "fake"
    assert "bounding_boxes" in payload


def test_cli_no_boxes_flag_omits_bounding_boxes(sample_image, capsys):
    cli.main([str(sample_image), "--engine", "fake", "--no-boxes"])
    payload = json.loads(capsys.readouterr().out)
    assert "bounding_boxes" not in payload
    assert "bbox" not in payload["detections"][0]


def test_cli_writes_single_output_file(sample_image, tmp_path):
    out_file = tmp_path / "result.json"
    cli.main([str(sample_image), "--engine", "fake", "--output", str(out_file)])
    payload = json.loads(out_file.read_text())
    assert payload["raw_text"] == "Fake Text"


def test_cli_batch_directory_writes_one_json_per_image(tmp_path):
    in_dir = tmp_path / "scans"
    in_dir.mkdir()
    for name in ["a.png", "b.png"]:
        cv2.imwrite(str(in_dir / name), np.full((50, 200, 3), 255, dtype=np.uint8))
    out_dir = tmp_path / "out"

    exit_code = cli.main([str(in_dir), "--engine", "fake", "--output", str(out_dir)])

    assert exit_code == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["a.json", "b.json"]


def test_cli_reports_missing_file_but_does_not_crash(tmp_path, capsys):
    good = tmp_path / "good.png"
    cv2.imwrite(str(good), np.full((50, 200, 3), 255, dtype=np.uint8))

    exit_code = cli.main([str(good), str(tmp_path / "missing.png"), "--engine", "fake"])

    assert exit_code == 1  # one file failed
    err = capsys.readouterr().err
    assert "missing.png" in err


def test_cli_rejects_unknown_engine_name(sample_image, capsys):
    exit_code = cli.main([str(sample_image), "--engine", "not_a_real_engine"])
    assert exit_code == 1
    assert "Unknown engine" in capsys.readouterr().err


def test_cli_debug_dir_writes_debug_images_for_single_file(sample_image, tmp_path):
    debug_dir = tmp_path / "debug"
    cli.main([str(sample_image), "--engine", "fake", "--debug-dir", str(debug_dir)])
    assert {"original.png", "preprocessed.png", "annotated.png"} == {p.name for p in debug_dir.iterdir()}


def test_cli_debug_dir_uses_a_subdirectory_per_input_for_batch(tmp_path):
    in_dir = tmp_path / "scans"
    in_dir.mkdir()
    for name in ["a.png", "b.png"]:
        cv2.imwrite(str(in_dir / name), np.full((50, 200, 3), 255, dtype=np.uint8))
    debug_dir = tmp_path / "debug"

    cli.main([str(in_dir), "--engine", "fake", "--debug-dir", str(debug_dir)])

    assert (debug_dir / "a" / "original.png").exists()
    assert (debug_dir / "b" / "original.png").exists()
