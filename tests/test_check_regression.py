import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from check_regression import check, load_summary, main


def _system(mean_cer, mean_latency_sec):
    return {"mean_cer": mean_cer, "mean_latency_sec": mean_latency_sec}


def test_check_passes_when_nothing_regressed():
    baseline = {"tesseract": _system(0.1, 0.2)}
    current = {"tesseract": _system(0.1, 0.2)}
    assert check(current, baseline, max_cer_increase=0.02, max_latency_increase_pct=25) == []


def test_check_passes_when_metrics_improve():
    baseline = {"tesseract": _system(0.1, 0.2)}
    current = {"tesseract": _system(0.05, 0.1)}
    assert check(current, baseline, max_cer_increase=0.02, max_latency_increase_pct=25) == []


def test_check_flags_cer_regression_beyond_threshold():
    baseline = {"tesseract": _system(0.1, 0.2)}
    current = {"tesseract": _system(0.15, 0.2)}  # +0.05, threshold 0.02
    failures = check(current, baseline, max_cer_increase=0.02, max_latency_increase_pct=25)
    assert len(failures) == 1
    assert "mean_cer regressed" in failures[0]


def test_check_tolerates_cer_regression_within_threshold():
    baseline = {"tesseract": _system(0.1, 0.2)}
    current = {"tesseract": _system(0.11, 0.2)}  # +0.01, threshold 0.02
    assert check(current, baseline, max_cer_increase=0.02, max_latency_increase_pct=25) == []


def test_check_flags_latency_regression_beyond_percent_threshold():
    baseline = {"tesseract": _system(0.1, 0.2)}
    current = {"tesseract": _system(0.1, 0.3)}  # +50%, threshold 25%
    failures = check(current, baseline, max_cer_increase=0.02, max_latency_increase_pct=25)
    assert len(failures) == 1
    assert "mean_latency_sec regressed" in failures[0]


def test_check_flags_system_missing_from_current_run():
    baseline = {"tesseract": _system(0.1, 0.2), "easyocr": _system(0.1, 0.2)}
    current = {"tesseract": _system(0.1, 0.2)}
    failures = check(current, baseline, max_cer_increase=0.02, max_latency_increase_pct=25)
    assert len(failures) == 1
    assert "easyocr" in failures[0]


def test_load_summary_accepts_benchmark_json_shape(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({
        "config": {"systems": ["tesseract"]},
        "summary": [{"system": "tesseract", "mean_cer": 0.1, "mean_latency_sec": 0.2}],
    }))
    summary = load_summary(str(path))
    assert summary == {"tesseract": {"system": "tesseract", "mean_cer": 0.1, "mean_latency_sec": 0.2}}


def test_load_summary_accepts_plain_baseline_shape(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"tesseract": {"mean_cer": 0.1, "mean_latency_sec": 0.2}}))
    summary = load_summary(str(path))
    assert summary == {"tesseract": {"mean_cer": 0.1, "mean_latency_sec": 0.2}}


def test_main_exits_nonzero_on_regression(tmp_path, capsys):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"tesseract": {"mean_cer": 0.1, "mean_latency_sec": 0.2}}))
    current_path = tmp_path / "current.json"
    current_path.write_text(json.dumps({"tesseract": {"mean_cer": 0.5, "mean_latency_sec": 0.2}}))

    exit_code = main(["--current", str(current_path), "--baseline", str(baseline_path)])

    assert exit_code == 1
    assert "REGRESSION DETECTED" in capsys.readouterr().err


def test_main_write_baseline_overwrites_baseline_file(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"tesseract": {"mean_cer": 0.1, "mean_latency_sec": 0.2}}))
    current_path = tmp_path / "current.json"
    current_path.write_text(json.dumps({"tesseract": {"mean_cer": 0.05, "mean_latency_sec": 0.15}}))

    exit_code = main(["--current", str(current_path), "--baseline", str(baseline_path), "--write-baseline"])

    assert exit_code == 0
    assert json.loads(baseline_path.read_text()) == {"tesseract": {"mean_cer": 0.05, "mean_latency_sec": 0.15}}
