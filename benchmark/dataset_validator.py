"""Validator for the generic real-dataset schema in `dataset_schema.py`
(mission section 8). Runs entirely on a manifest (a list of row dicts,
typically loaded from a JSONL/CSV file) — no dataset needs to exist yet
for this module to be tested, only a manifest structure to validate
against, which is exactly the point: this is infrastructure to be ready
the moment a real dataset shows up, not a claim that one has.

Every check below produces a structured `ValidationIssue`, not a bare
string — `severity` lets a caller decide whether to abort ("error") or
just take note ("warning") programmatically, and `row_image_id`
(nullable) lets an issue be either row-scoped or manifest-scoped.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from benchmark.dataset_schema import REQUIRED_FIELDS, VALID_SPLITS


@dataclass(frozen=True)
class ValidationIssue:
    severity: str          # "error" | "warning"
    category: str          # e.g. "missing_field", "broken_path", "duplicate_id", "encoding", "split_overlap"
    message: str
    row_image_id: str | None = None


@dataclass
class ValidationReport:
    n_rows: int
    issues: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        return (f"{self.n_rows} rows checked: {len(self.errors)} error(s), "
                f"{len(self.warnings)} warning(s)")


def validate_manifest(rows: list[dict], check_images_exist: bool = True) -> ValidationReport:
    """Validates a list of dataset-row dicts. `check_images_exist=False`
    lets this run against a manifest whose image files aren't actually
    present yet (e.g. a manifest describing a dataset not yet downloaded)
    while still catching every other class of problem."""
    issues: list[ValidationIssue] = []
    seen_ids: dict[str, int] = {}
    seen_id_splits: dict[str, set] = {}

    for idx, row in enumerate(rows):
        image_id = row.get("image_id")
        row_label = image_id if image_id else f"<row {idx}, no image_id>"

        missing = [f for f in REQUIRED_FIELDS if not row.get(f)]
        if missing:
            issues.append(ValidationIssue("error", "missing_field",
                          f"missing/empty required field(s) {missing}", row_label))
            continue  # remaining checks assume required fields are present

        if image_id in seen_ids:
            issues.append(ValidationIssue("error", "duplicate_id",
                          f"image_id {image_id!r} also used at row {seen_ids[image_id]}", image_id))
        else:
            seen_ids[image_id] = idx
        seen_id_splits.setdefault(image_id, set()).add(row.get("split"))

        if row.get("split") not in VALID_SPLITS:
            issues.append(ValidationIssue("error", "invalid_split",
                          f"split {row.get('split')!r} not one of {VALID_SPLITS}", image_id))

        ground_truth = row.get("ground_truth_text", "")
        try:
            ground_truth.encode("utf-8")
        except (UnicodeEncodeError, AttributeError):
            issues.append(ValidationIssue("error", "encoding",
                          "ground_truth_text is not valid UTF-8-encodable text", image_id))

        metadata = row.get("metadata", {})
        if metadata is not None and not isinstance(metadata, dict):
            issues.append(ValidationIssue("error", "invalid_metadata",
                          f"metadata must be a dict or None, got {type(metadata).__name__}", image_id))

        boxes = row.get("bounding_boxes")
        if boxes is not None:
            if not isinstance(boxes, list):
                issues.append(ValidationIssue("error", "invalid_bounding_boxes",
                              "bounding_boxes must be a list or None", image_id))
            else:
                for b in boxes:
                    if not isinstance(b, dict) or "text" not in b or "box" not in b:
                        issues.append(ValidationIssue("error", "invalid_bounding_boxes",
                                      f"each bounding box needs 'text' and 'box' keys, got {b!r}", image_id))
                        break

        image_path = row.get("image_path", "")
        if check_images_exist:
            if not image_path or not os.path.isfile(image_path):
                issues.append(ValidationIssue("error", "broken_path",
                              f"image_path {image_path!r} does not exist or is not a file", image_id))
            elif os.path.getsize(image_path) == 0:
                issues.append(ValidationIssue("error", "broken_path",
                              f"image_path {image_path!r} exists but is empty", image_id))

    for image_id, splits in seen_id_splits.items():
        if len(splits) > 1:
            issues.append(ValidationIssue("error", "split_overlap",
                          f"image_id {image_id!r} appears in more than one split: {sorted(splits)}", image_id))

    return ValidationReport(n_rows=len(rows), issues=issues)


def write_validation_report(report: ValidationReport, out_path: str) -> None:
    lines = [report.summary(), ""]
    for issue in report.issues:
        lines.append(f"[{issue.severity.upper()}] {issue.category} (image_id={issue.row_image_id}): {issue.message}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
