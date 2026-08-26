"""Tests for the real-dataset infrastructure (schema, validator, failure
taxonomy, run_real_dataset scaffold) built ahead of any real dataset
existing (mission sections 7-11). These fixtures use this project's own
synthetic corpus images purely to exercise the SCAFFOLD's mechanics
(file I/O, validation logic, CLI plumbing) — nothing here is, or is
represented as, a real-world or multilingual accuracy result.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.corpus import build_corpus
from benchmark.dataset_schema import DatasetRow
from benchmark.dataset_validator import validate_manifest, write_validation_report
from benchmark.failure_taxonomy import FailureType, classify_failure
from benchmark.run_real_dataset import evaluate, load_manifest, main


def _valid_row(**overrides) -> dict:
    row = {
        "image_id": "img_001", "image_path": "some/path.png", "ground_truth_text": "hello world",
        "language": "en", "script": "Latin", "document_type": "printed",
        "source_dataset": "unit_test_fixture", "license": "N/A", "split": "test",
    }
    row.update(overrides)
    return row


# --- dataset_schema ---------------------------------------------------

def test_dataset_row_from_dict_round_trips():
    d = _valid_row()
    row = DatasetRow.from_dict(d)
    assert row.image_id == "img_001"
    assert row.to_dict()["ground_truth_text"] == "hello world"


def test_dataset_row_from_dict_rejects_missing_required_field():
    d = _valid_row()
    del d["license"]
    with pytest.raises(ValueError, match="missing required fields"):
        DatasetRow.from_dict(d)


# --- dataset_validator --------------------------------------------------

def test_validate_manifest_accepts_a_clean_manifest(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"not a real png but non-empty")
    rows = [_valid_row(image_path=str(img))]
    report = validate_manifest(rows, check_images_exist=True)
    assert report.is_valid
    assert report.n_rows == 1


def test_validate_manifest_flags_missing_field():
    rows = [_valid_row(license="")]
    report = validate_manifest(rows)
    assert not report.is_valid
    assert any(i.category == "missing_field" for i in report.errors)


def test_validate_manifest_flags_duplicate_image_id():
    rows = [_valid_row(image_id="dup"), _valid_row(image_id="dup")]
    report = validate_manifest(rows, check_images_exist=False)
    assert any(i.category == "duplicate_id" for i in report.errors)


def test_validate_manifest_flags_split_overlap():
    rows = [_valid_row(image_id="x", split="train"), _valid_row(image_id="x", split="test")]
    report = validate_manifest(rows, check_images_exist=False)
    assert any(i.category == "split_overlap" for i in report.errors)


def test_validate_manifest_flags_invalid_split():
    rows = [_valid_row(split="bogus")]
    report = validate_manifest(rows, check_images_exist=False)
    assert any(i.category == "invalid_split" for i in report.errors)


def test_validate_manifest_flags_broken_path():
    rows = [_valid_row(image_path="/does/not/exist.png")]
    report = validate_manifest(rows, check_images_exist=True)
    assert any(i.category == "broken_path" for i in report.errors)


def test_validate_manifest_skips_image_check_when_disabled():
    rows = [_valid_row(image_path="/does/not/exist.png")]
    report = validate_manifest(rows, check_images_exist=False)
    assert report.is_valid


def test_validate_manifest_flags_invalid_metadata_type():
    rows = [_valid_row(metadata="not a dict")]
    report = validate_manifest(rows, check_images_exist=False)
    assert any(i.category == "invalid_metadata" for i in report.errors)


def test_validate_manifest_flags_malformed_bounding_boxes():
    rows = [_valid_row(bounding_boxes=[{"missing": "required keys"}])]
    report = validate_manifest(rows, check_images_exist=False)
    assert any(i.category == "invalid_bounding_boxes" for i in report.errors)


def test_validate_manifest_empty_list_is_valid_with_zero_rows():
    report = validate_manifest([], check_images_exist=False)
    assert report.is_valid
    assert report.n_rows == 0


def test_write_validation_report_produces_readable_file(tmp_path):
    report = validate_manifest([_valid_row(license="")], check_images_exist=False)
    out = tmp_path / "report.txt"
    write_validation_report(report, str(out))
    content = out.read_text(encoding="utf-8")
    assert "ERROR" in content
    assert "missing_field" in content


# --- failure_taxonomy ---------------------------------------------------

def test_classify_failure_engine_error_on_exception():
    assert classify_failure("", "hello", None, engine_raised_exception=True) == FailureType.ENGINE_ERROR


def test_classify_failure_engine_error_when_cer_is_none():
    assert classify_failure("hello", "hello", None) == FailureType.ENGINE_ERROR


def test_classify_failure_none_for_exact_match():
    assert classify_failure("hello world", "hello world", 0.0) == FailureType.NONE


def test_classify_failure_blank_output():
    assert classify_failure("", "hello world", 1.0) == FailureType.BLANK_OUTPUT


def test_classify_failure_blank_prediction_with_blank_truth_is_not_a_failure():
    assert classify_failure("", "", 0.0) == FailureType.NONE


def test_classify_failure_catastrophic():
    assert classify_failure("xyzzy plugh", "hello world today", 0.97) == FailureType.CATASTROPHIC_FAILURE


def test_classify_failure_wrong_order():
    result = classify_failure("world hello", "hello world", 0.4)
    assert result == FailureType.WRONG_ORDER


def test_classify_failure_partial_text():
    result = classify_failure("hello", "hello world today", 0.5)
    assert result == FailureType.PARTIAL_TEXT


def test_classify_failure_missing_text():
    result = classify_failure("today", "hello world foo bar today", 0.7)
    assert result == FailureType.MISSING_TEXT


def test_classify_failure_recognition_noise():
    result = classify_failure("hallo world", "hello world", 0.09)
    assert result == FailureType.RECOGNITION_NOISE


def test_classify_failure_wrong_text():
    result = classify_failure("completely different words here", "hello world today please", 0.6)
    assert result == FailureType.WRONG_TEXT


# --- run_real_dataset scaffold ------------------------------------------

def test_load_manifest_raises_clear_error_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_manifest(str(tmp_path / "does_not_exist.jsonl"))


def test_load_manifest_rejects_malformed_json_line(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text('{"valid": "json"}\nnot json at all\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_manifest(str(p))


def test_load_manifest_skips_blank_lines(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    rows = load_manifest(str(p))
    assert rows == [{"a": 1}, {"b": 2}]


def test_evaluate_runs_tesseract_against_synthetic_corpus_fixture(tmp_path):
    """Exercises evaluate()'s mechanics end-to-end using this project's
    own synthetic corpus as a stand-in image — the result is a plumbing
    check, not a claim about real-dataset accuracy."""
    manifest = build_corpus(os.path.join(os.path.dirname(__file__), "..", "benchmark", "_corpus_cache"))
    item = manifest[0]
    rows = [_valid_row(image_id="fixture_0", image_path=item["path"], ground_truth_text=item["ground_truth"])]
    results = evaluate(rows, ["tesseract"])
    assert len(results) == 1
    assert results.iloc[0]["system"] == "tesseract"
    assert results.iloc[0]["cer"] is not None
    assert results.iloc[0]["error"] is None


def test_evaluate_records_error_row_for_unreadable_image():
    rows = [_valid_row(image_path="/definitely/not/a/real/image.png")]
    results = evaluate(rows, ["tesseract"])
    assert results.iloc[0]["failure_type"] == "engine_error"
    assert results.iloc[0]["cer"] is None


def test_main_aborts_cleanly_on_invalid_manifest(tmp_path, capsys):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(_valid_row(license="")) + "\n", encoding="utf-8")
    exit_code = main(["--manifest", str(manifest), "--output-dir", str(tmp_path / "out")])
    assert exit_code == 1
    assert (tmp_path / "out" / "real_dataset_validation.txt").exists()


def test_main_end_to_end_with_synthetic_fixture(tmp_path):
    manifest_corpus = build_corpus(os.path.join(os.path.dirname(__file__), "..", "benchmark", "_corpus_cache"))
    item = manifest_corpus[0]
    manifest_path = tmp_path / "manifest.jsonl"
    row = _valid_row(image_id="fixture_0", image_path=item["path"], ground_truth_text=item["ground_truth"])
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    exit_code = main(["--manifest", str(manifest_path), "--engines", "tesseract",
                       "--output-dir", str(tmp_path / "out")])
    assert exit_code == 0
    assert (tmp_path / "out" / "real_dataset_results.csv").exists()
