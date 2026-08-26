"""Generic real-dataset schema (mission section 7). This project's only
corpus so far is the synthetic, rendered-from-known-text one in
`corpus.py` — useful for controlled ablations, but it cannot support any
real-world or multilingual accuracy claim (see
`docs/engineering_backlog.md`'s "Blocked externally" section for exactly
why: no real dataset is downloaded or fabricated here, network access is
sandboxed, and inventing one would violate this project's own data-
integrity rule).

What CAN be built without a dataset in hand is the schema, validator,
and harness that would consume one the moment it becomes available —
that is what this module and `dataset_validator.py` / `run_real_dataset.py`
provide. Every field below is deliberately generic (not tied to any one
public dataset's column names) so this schema can absorb whatever
real dataset eventually gets used, including multilingual/non-Latin-
script ones (mission section 12) — hence `language`/`script` as separate
fields and `ground_truth_text` as a plain Python str (Unicode-safe by
construction; the validator below explicitly checks UTF-8 encodability
rather than assuming it).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetRow:
    """One labeled example in a real-dataset manifest. Required fields
    have no default; everything after them is optional because not every
    real dataset will have bounding-box ground truth or rich metadata."""

    image_id: str
    image_path: str
    ground_truth_text: str
    language: str          # e.g. "en", "hi", "zh" — ISO 639-1/639-3 preferred, not enforced here
    script: str            # e.g. "Latin", "Devanagari", "Han" — free text, not enforced against a fixed list
    document_type: str     # e.g. "printed", "handwritten", "scene_text", "form"
    source_dataset: str    # name/citation of the real dataset this row came from
    license: str           # the source dataset's license — required so this project never
                            # ships derived results without knowing it's allowed to
    split: str             # "train" | "validation" | "test" — see dataset_validator's split-overlap check
    metadata: dict = field(default_factory=dict)
    bounding_boxes: list | None = None   # optional [{"text": str, "box": [x, y, w, h]}, ...]

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id, "image_path": self.image_path,
            "ground_truth_text": self.ground_truth_text, "language": self.language,
            "script": self.script, "document_type": self.document_type,
            "source_dataset": self.source_dataset, "license": self.license,
            "split": self.split, "metadata": self.metadata,
            "bounding_boxes": self.bounding_boxes,
        }

    @staticmethod
    def from_dict(d: dict) -> DatasetRow:
        known = {"image_id", "image_path", "ground_truth_text", "language", "script",
                 "document_type", "source_dataset", "license", "split", "metadata", "bounding_boxes"}
        missing = {"image_id", "image_path", "ground_truth_text", "language", "script",
                   "document_type", "source_dataset", "license", "split"} - d.keys()
        if missing:
            raise ValueError(f"dataset row missing required fields: {sorted(missing)}")
        return DatasetRow(**{k: v for k, v in d.items() if k in known})


REQUIRED_FIELDS = ("image_id", "image_path", "ground_truth_text", "language",
                   "script", "document_type", "source_dataset", "license", "split")
VALID_SPLITS = ("train", "validation", "test")
