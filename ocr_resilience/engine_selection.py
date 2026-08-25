"""Quality-aware single-engine selection — replaces picking
`available_engines[0]` (pure registration order, see `router.decide`)
with an interpretable, rule-based choice informed by this project's own
benchmark evidence (`docs/engine_selection_report.md`).

Deliberately NOT a learned/trained model (see `docs/engineering_backlog.md`'s
"requires compute/scale this environment doesn't have" section — the
20-image corpus is far too small to train anything without memorizing
it). Deliberately NOT keyed on degradation preset NAMES either — a real
image never arrives labeled "heavy_blur" — every rule here reasons from
`QualityReport`'s own continuous measurements (`blur_score`, `noise_score`,
`contrast_score`, etc.), which exist for exactly this reason and were
already being computed, just never consulted for engine choice before.

`EngineProfile` separates STATIC capability (what an engine's vendor/docs
claim it supports — see `docs/engine_landscape.md`) from EMPIRICAL
capability (what this project's own benchmark measured). The selector
below only uses empirical evidence; static capability is recorded for
context and for candidates without benchmark data yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .quality import QualityReport


@dataclass
class EngineProfile:
    name: str
    # Static (vendor/docs-level) capability — informational, not used for selection.
    static_notes: str = ""
    # Empirical: conditions (informal labels, for humans reading this file /
    # docs/engine_selection_report.md) where THIS benchmark found the engine
    # notably strong or fragile. Not consulted programmatically by name —
    # the actual selection rules below key off QualityReport fields, derived
    # FROM these findings, not off these strings.
    empirical_strengths: list[str] = field(default_factory=list)
    empirical_weaknesses: list[str] = field(default_factory=list)


# Populated from docs/engine_selection_report.md's condition-x-engine table —
# kept here as a human-readable cross-reference, not as selection logic itself.
KNOWN_PROFILES: dict[str, EngineProfile] = {
    "paddleocr": EngineProfile(
        name="paddleocr",
        static_notes="PP-OCR family, 80+ languages claimed, CPU-friendly, fastest of the neural engines here.",
        empirical_strengths=["clean", "light_blur", "noisy", "skewed", "low_contrast", "smudged"],
        empirical_weaknesses=["salt_pepper (can fail badly without preprocessing)"],
    ),
    "tesseract": EngineProfile(
        name="tesseract",
        static_notes="Classical LSTM engine, no neural detector, smallest memory footprint by far.",
        empirical_strengths=["general noise at LOW severity", "low_contrast (mild)"],
        empirical_weaknesses=[
            "salt_pepper without preprocessing (complete failure, CER 1.0)",
            "Gaussian noise above a sharp cliff around sigma 50 (complete failure, CER 1.0) — see docs/robustness_curves.md",
        ],
    ),
    "easyocr": EngineProfile(
        name="easyocr",
        static_notes="PyTorch CRNN-based, GPU-optional, slower than PaddleOCR on CPU.",
        empirical_strengths=["combo_hard (stacked degradations)"],
        empirical_weaknesses=["motion_blur (catastrophic, CER > 0.5)"],
    ),
    "rapidocr": EngineProfile(
        name="rapidocr",
        static_notes="ONNX-runtime port of PP-OCR, no PyTorch/Paddle dependency, lightest install.",
        empirical_strengths=["motion_blur (surprisingly, best of the four single engines)"],
        empirical_weaknesses=["heavy_blur (catastrophic, CER 1.0)"],
    ),
}


def select_primary_engine(report: QualityReport, available_engines: list[str]) -> tuple[str, str]:
    """Pick ONE engine for the "easy path" (single-engine, non-ensembled)
    route, using the observed QualityReport rather than registration
    order. Falls back to the first available engine (previous behavior)
    when no rule applies or the preferred engine isn't registered — this
    function must always return something usable, never raise, since the
    router still needs a primary engine even for conditions this hasn't
    been tuned for yet.

    Every threshold here is a DIRECT translation of a specific finding in
    docs/engine_selection_report.md / docs/robustness_curves.md — not
    invented. Update this docstring's threshold values together with
    those docs if the evidence changes; don't let them drift apart.
    """
    def pick(preferred: str, reason: str) -> tuple[str, str] | None:
        if preferred in available_engines:
            return preferred, reason
        return None

    # Tesseract's Gaussian-noise cliff sits between sigma 25 and 50 (robustness
    # curves); noise_score is not on the same numeric scale as `sigma`, but a
    # noticeably elevated noise_score alongside the is_noisy flag is the
    # closest available proxy for "past that cliff" without re-deriving a
    # sigma estimate from a scanned/photographed image, which isn't possible.
    if report.is_noisy and not report.is_impulse_noisy and report.noise_score > 20:
        for engine, reason in [
            ("paddleocr", "high general-noise severity: PaddleOCR strongest here, and Tesseract has a measured failure cliff here"),
            ("easyocr", "high general-noise severity: avoiding Tesseract's measured failure cliff here"),
        ]:
            result = pick(engine, reason)
            if result:
                return result

    # Impulse (salt-and-pepper) noise is deliberately NOT given an engine
    # preference rule here: both Tesseract (without preprocessing) and
    # PaddleOCR can fail badly on it per this benchmark, and it's primarily
    # handled by `preprocess.median_denoise` (gated on this same flag)
    # rather than by single-engine choice — asserting a rule here without
    # evidence of which engine is actually best POST-preprocessing would be
    # exactly the "guess, don't measure" anti-pattern this file exists to avoid.

    if report.is_low_contrast:
        result = pick("paddleocr", "low contrast: PaddleOCR was the strongest single engine on this condition in benchmarking")
        if result:
            return result

    if report.is_blurry and report.blur_score < 30:
        # Severely blurred (well below the is_blurry threshold of 100) — RapidOCR's
        # measured heavy-blur cliff (CER 1.0) makes it a bad primary-engine choice here.
        for engine, reason in [
            ("paddleocr", "severe blur: PaddleOCR handled this well in benchmarking; avoiding RapidOCR's measured failure"),
            ("tesseract", "severe blur: avoiding RapidOCR's measured heavy-blur failure"),
        ]:
            result = pick(engine, reason)
            if result:
                return result

    if report.is_skewed:
        result = pick("paddleocr", "skew: PaddleOCR was the strongest single engine on this condition in benchmarking")
        if result:
            return result

    return available_engines[0], f"no condition-specific rule matched -> first available engine ({available_engines[0]})"


# Overall engine strength ranking from this benchmark's aggregate mean CER
# (see docs/engine_selection_report.md for the exact numbers) — used ONLY to
# order a ranked fallback chain, never for the initial engine choice (that's
# `select_primary_engine`'s job, using condition-specific evidence instead of
# one aggregate number).
OVERALL_ENGINE_RANKING = ["paddleocr", "tesseract", "easyocr", "rapidocr"]


def rank_fallback_chain(primary: str, available_engines: list[str]) -> list[str]:
    """Ordered list of engines to escalate through AFTER `primary`, most-
    promising-first, before resorting to the full ensemble — replaces
    jumping straight from one engine to "every engine at once." Engines
    not in `OVERALL_ENGINE_RANKING` (a future fifth engine, say) are
    appended at the end in their given order, not dropped."""
    ranked = [e for e in OVERALL_ENGINE_RANKING if e in available_engines and e != primary]
    unranked = [e for e in available_engines if e not in OVERALL_ENGINE_RANKING and e != primary]
    return ranked + unranked
