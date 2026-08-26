"""Failure-type taxonomy (mission section 10). A fixed vocabulary of WHY
an OCR result was wrong, not just HOW wrong (CER/WER already capture
that). This is schema + a deliberately coarse, fully rule-based
classifier over CER and the two text strings — explicitly NOT a
learned/statistical classifier (the mission's hard limit on that applies
here too), and explicitly NOT claimed to be accurate on real-world text:
it was designed and only sanity-checked against this project's synthetic
corpus, where ground truth is exact. A future real dataset (see
`dataset_schema.py`) is what would let this be validated properly.

Two of the nine failure types in the mission's list —
`preprocessing_failure` and `engine_error` beyond a raised exception —
are NOT reliably determinable from (predicted_text, ground_truth_text,
cer) alone, and `classify` says so rather than guessing: forcing every
case into one of the nine labels would manufacture false precision this
method cannot support.
"""
from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    NONE = "none"                                # not a failure: exact or near-exact match
    BLANK_OUTPUT = "blank_output"                 # engine returned nothing (or whitespace only)
    MISSING_TEXT = "missing_text"                 # some text recognized, but most of it absent
    WRONG_TEXT = "wrong_text"                      # substantial text present but substantially incorrect
    PARTIAL_TEXT = "partial_text"                 # a correct SUBSET of the ground truth, nothing extra wrong
    WRONG_ORDER = "wrong_order"                   # same words present, different order (e.g. line-reconstruction bug)
    RECOGNITION_NOISE = "recognition_noise"       # mostly correct, minor character-level errors
    CATASTROPHIC_FAILURE = "catastrophic_failure" # CER at or near 1.0 — near-total failure
    ENGINE_ERROR = "engine_error"                 # the engine raised an exception, or produced no CER at all
    PREPROCESSING_FAILURE = "preprocessing_failure"  # NOT determinable from text/CER alone — see module docstring
    UNDETERMINED = "undetermined"                 # doesn't cleanly fit any of the above from this evidence alone


CATASTROPHIC_THRESHOLD = 0.95
RECOGNITION_NOISE_THRESHOLD = 0.15
MISSING_TEXT_WORD_RATIO = 0.5


def classify_failure(predicted_text: str, ground_truth_text: str, cer_value: float | None,
                      engine_raised_exception: bool = False) -> FailureType:
    """Deterministic, order-of-checks classification. Each branch is a
    plain, auditable rule over CER/word overlap — no fitting, no
    thresholds tuned against this project's own benchmark numbers beyond
    the CATASTROPHIC_THRESHOLD this project already uses elsewhere
    (`benchmark/run_benchmark.py`'s own catastrophic_failure flag) for
    consistency."""
    if engine_raised_exception:
        return FailureType.ENGINE_ERROR
    if cer_value is None:
        return FailureType.ENGINE_ERROR

    pred_stripped = predicted_text.strip()
    truth_stripped = ground_truth_text.strip()

    if not pred_stripped:
        return FailureType.BLANK_OUTPUT if truth_stripped else FailureType.NONE
    if cer_value <= 0.0:
        return FailureType.NONE
    if cer_value >= CATASTROPHIC_THRESHOLD:
        return FailureType.CATASTROPHIC_FAILURE

    pred_words, truth_words = pred_stripped.split(), truth_stripped.split()
    if pred_words and sorted(pred_words) == sorted(truth_words) and pred_words != truth_words:
        return FailureType.WRONG_ORDER
    if truth_stripped.startswith(pred_stripped) and pred_stripped != truth_stripped:
        return FailureType.PARTIAL_TEXT
    if truth_words and len(pred_words) < MISSING_TEXT_WORD_RATIO * len(truth_words):
        return FailureType.MISSING_TEXT
    if cer_value <= RECOGNITION_NOISE_THRESHOLD:
        return FailureType.RECOGNITION_NOISE
    if pred_words:
        return FailureType.WRONG_TEXT
    return FailureType.UNDETERMINED
