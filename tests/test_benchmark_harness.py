"""Structural tests for the benchmark harness itself (corpus manifest
shape, degradation preset registry, seed determinism) — none of this was
tested before; the seed-determinism test in particular guards against the
`hash()`-based reproducibility bug fixed in run_benchmark.py (Python's
string hash is salted per-process, so the old seed wasn't actually
reproducible across runs despite looking deterministic within one)."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.corpus import SENTENCES, build_corpus
from benchmark.degrade import DEGRADATION_PRESETS, apply_degradation
from benchmark.run_benchmark import stable_seed


def test_degradation_presets_cover_the_documented_set():
    expected = {
        "clean", "light_blur", "heavy_blur", "motion_blur", "noisy", "salt_pepper",
        "skewed", "low_contrast", "smudged", "jpeg_compressed", "combo_hard",
    }
    assert set(DEGRADATION_PRESETS) == expected


def test_build_corpus_manifest_has_one_entry_per_sentence_per_style(tmp_path):
    manifest = build_corpus(str(tmp_path))
    assert len(manifest) == len(SENTENCES) * 2
    styles = {item["style"] for item in manifest}
    assert styles == {"printed", "handwritten_proxy"}
    for item in manifest:
        assert os.path.exists(item["path"])
        assert item["ground_truth"] in SENTENCES


def test_stable_seed_is_deterministic_across_calls():
    assert stable_seed("a.png", "skewed") == stable_seed("a.png", "skewed")


def test_stable_seed_differs_for_different_inputs():
    assert stable_seed("a.png", "skewed") != stable_seed("b.png", "skewed")
    assert stable_seed("a.png", "skewed") != stable_seed("a.png", "noisy")


@pytest.mark.parametrize("preset", sorted(DEGRADATION_PRESETS))
def test_apply_degradation_with_stable_seed_is_reproducible_for_every_preset(tmp_path, preset):
    """Regression for a real bug: `gaussian_noise`/`salt_and_pepper` used
    to call `np.random.normal`/`np.random.randint` directly, reading
    numpy's GLOBAL random state and ignoring the `rng` parameter every
    other degradation function respects — so `noisy`, `salt_pepper`, and
    `combo_hard` (which chains gaussian_noise) were NOT actually
    seed-reproducible despite `stable_seed()`'s whole purpose being
    reproducibility. The pre-existing version of this test only checked
    `skewed`, which never used numpy's global state and so never caught
    it — parametrized over every preset specifically so a future
    non-reproducible degradation can't hide the same way."""
    manifest = build_corpus(str(tmp_path))
    import cv2
    img = cv2.imread(manifest[0]["path"])

    seed = stable_seed(manifest[0]["path"], preset)
    out_a = apply_degradation(img, preset, seed=seed)
    out_b = apply_degradation(img, preset, seed=seed)

    assert (out_a == out_b).all(), f"preset '{preset}' is not seed-reproducible"
