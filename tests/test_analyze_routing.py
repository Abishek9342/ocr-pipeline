"""Tests for benchmark/analyze_routing.py — in particular a real bug
found while reviewing its own first real-data run: `top1` was computed by
comparing ENGINE NAMES (`selected == best_row["system"]`), where
`best_row` came from `df["cer"].idxmin()`. Pandas' `idxmin()` resolves
ties by returning the FIRST matching row in the DataFrame's existing
order — which is always Tesseract, since `run_benchmark.py` appends rows
in a fixed engine order. On easy images, many engines tie at CER=0.0, so
this silently inflated Tesseract's apparent "always the best" rate
without it actually being uniquely best. Fixed by comparing CER VALUES
(`selected_cer <= best_cer`), not names."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))

import pandas as pd
from analyze_routing import condition_engine_table, routing_regret


def test_condition_engine_table_finds_the_true_best_engine_per_preset():
    df = pd.DataFrame([
        {"preset": "clean", "system": "tesseract", "cer": 0.05, "latency_sec": 0.1},
        {"preset": "clean", "system": "paddleocr", "cer": 0.01, "latency_sec": 0.2},
        {"preset": "clean", "system": "tesseract", "cer": 0.03, "latency_sec": 0.1},
        {"preset": "clean", "system": "paddleocr", "cer": 0.02, "latency_sec": 0.2},
    ])
    table = condition_engine_table(df, ["tesseract", "paddleocr"])
    assert table.loc["clean", "best_engine"] == "paddleocr"
    assert table.loc["clean", "fastest_engine"] == "tesseract"


def test_routing_regret_top1_is_tie_fair_not_name_biased(tmp_path, monkeypatch):
    """The actual regression: two engines tied at CER=0.0 on the same
    image, with Tesseract appearing FIRST in row order (as
    run_benchmark.py always writes it). A name-based `idxmin()` comparison
    would call this a Tesseract-only win; a tie-fair comparison must
    credit BOTH engines with a top-1 hit here."""
    df = pd.DataFrame([
        {"image_id": "a.png", "preset": "clean", "system": "tesseract", "cer": 0.0},
        {"image_id": "a.png", "preset": "clean", "system": "paddleocr", "cer": 0.0},
    ])

    # Stub out the corpus/degradation/quality dependencies routing_regret()
    # normally uses — this test is only about the tie-breaking arithmetic,
    # not about re-deriving a real QualityReport.
    import analyze_routing

    monkeypatch.setattr(analyze_routing, "build_corpus", lambda corpus_dir: [{"path": "a.png"}])
    monkeypatch.setattr(analyze_routing, "apply_degradation", lambda img, preset, seed: img)
    monkeypatch.setattr(analyze_routing, "assess", lambda img: object())
    monkeypatch.setattr(analyze_routing, "select_primary_engine", lambda report, engines: ("paddleocr", "test"))
    monkeypatch.setattr(analyze_routing.cv2, "imread", lambda path: "fake_image")

    regret_df = routing_regret(df, ["tesseract", "paddleocr"], corpus_dir="unused")

    assert len(regret_df) == 1
    row = regret_df.iloc[0]
    # old_router always picks baseline_systems[0] == "tesseract"
    assert row["old_router_selected"] == "tesseract"
    assert row["old_router_top1"] == True  # noqa: E712 - tesseract DID tie for the best CER (0.0)
    assert row["new_router_selected"] == "paddleocr"
    assert row["new_router_top1"] == True  # noqa: E712 - paddleocr ALSO tied for the best CER
    assert row["old_router_regret"] == 0.0
    assert row["new_router_regret"] == 0.0
