"""Multi-engine consensus fusion — the classical technique this borrows
from is ROVER (Recognizer Output Voting Error Reduction, Fiscus 1997),
originally built for combining multiple speech-recognizer transcripts.
The idea transfers directly to OCR: run N engines, align their outputs,
vote per position. No ML model, no LLM — just dynamic-programming
sequence alignment and weighted majority voting.

Two levels of alignment happen here:
  1. SPATIAL — which detections from different engines are even talking
     about the same region of the image (via bounding-box IoU).
  2. TEXTUAL — for detections that ARE the same region, which characters
     the engines agree on vs. disagree on (via Needleman-Wunsch character
     alignment), then a confidence-weighted vote per aligned position.
"""
from __future__ import annotations

from collections import Counter

from .engines import Detection

GAP = "\0"  # sentinel for "no character here" in an alignment column


def overlap_ratio(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Intersection over the SMALLER box's area (not standard IoU).

    Different engines segment text at different granularities — Tesseract
    tends to return one box per WORD, EasyOCR often returns one box per
    LINE. A word box fully contained inside a line box has a tiny
    standard IoU (union is dominated by the much larger line box) even
    though it's 100% the same text region. Intersection-over-smaller
    fixes this: a fully-contained small box always scores 1.0, regardless
    of how much larger the other box is. An earlier version of this
    function used plain IoU with a 0.3 threshold — verified via the
    benchmark to silently fail to merge exactly this word-vs-line case,
    causing text to be duplicated instead of fused (both detections kept
    as separate "regions"), which was the single dominant cause of the
    pipeline scoring WORSE than either engine alone on every case where
    multi-engine ensembling actually triggered."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def group_by_region(detections: list[Detection], overlap_threshold: float = 0.5) -> list[list[Detection]]:
    """Union-find clustering on bbox overlap — NOT simple greedy "join the
    first matching group and stop." A single line-level box from one
    engine routinely bridges several word-level boxes from another engine
    that don't overlap each other AT ALL (adjacent words on the same
    line). Greedy first-match clustering only attaches the bridging
    detection to whichever word-box group it happens to check first,
    leaving the other word boxes stranded as separate singleton groups
    instead of transitively merged into one — verified via the benchmark
    to be the dominant remaining cause of text duplication after the
    IoU-vs-overlap-ratio fix above."""
    n = len(detections)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if overlap_ratio(detections[i].bbox, detections[j].bbox) >= overlap_threshold:
                union(i, j)

    clusters: dict[int, list[Detection]] = {}
    for idx, det in enumerate(detections):
        clusters.setdefault(find(idx), []).append(det)
    return list(clusters.values())


def _needleman_wunsch(a: str, b: str, match: int = 2, mismatch: int = -1, gap: int = -1) -> tuple[str, str]:
    """Standard global alignment DP — returns (a_aligned, b_aligned), same
    length, with GAP inserted wherever one string has no counterpart."""
    n, m = len(a), len(b)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * gap
    for j in range(1, m + 1):
        score[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap
            score[i][j] = max(diag, up, left)

    aligned_a, aligned_b = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i][j] == score[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            aligned_a.append(a[i - 1]); aligned_b.append(b[j - 1])
            i -= 1; j -= 1
        elif i > 0 and score[i][j] == score[i - 1][j] + gap:
            aligned_a.append(a[i - 1]); aligned_b.append(GAP)
            i -= 1
        else:
            aligned_a.append(GAP); aligned_b.append(b[j - 1])
            j -= 1
    return "".join(reversed(aligned_a)), "".join(reversed(aligned_b))


def _vote_text(candidates: list[tuple[str, float]], weighted: bool = True) -> str:
    """ROVER-style consensus: iteratively align each new candidate against
    the running consensus backbone, then vote per column.

    `weighted=True` (default) is confidence-weighted, with the highest-
    confidence candidate as the starting backbone. This assumes different
    engines' confidence SCALES are comparable — investigated directly (see
    docs/failure_analysis.md) and found NOT reliably true: on the
    `combo_hard` degradation, Tesseract's confidence ran systematically
    higher than EasyOCR's even in cases where Tesseract was the WRONG
    answer (e.g. 0.802 vs 0.679 confidence, but Tesseract's text had
    higher CER), which structurally biases the weighted vote toward
    whichever engine happens to over-report confidence, independent of
    actual correctness. `weighted=False` gives every candidate equal
    weight (uniform majority vote) and uses the CALLER's ordering for the
    backbone/tie-break (see `fuse()`, which sorts by engine name for
    determinism) instead of a confidence-based one. Neither is
    unconditionally better — see the ablation comparing them before
    changing the default."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0][0]
    if len(set(text for text, _ in candidates)) == 1:
        return candidates[0][0]  # unanimous — skip alignment entirely

    ordered = sorted(candidates, key=lambda c: -c[1]) if weighted else list(candidates)
    backbone_text, backbone_conf = ordered[0]
    backbone_weight = backbone_conf if weighted else 1.0
    columns: list[list[tuple[str, float]]] = [[(ch, backbone_weight)] for ch in backbone_text]

    for text, conf in ordered[1:]:
        weight = conf if weighted else 1.0
        aligned_backbone, aligned_new = _needleman_wunsch(
            "".join(col[0][0] if len(col) == 1 else _majority_char(col) for col in columns), text,
        )
        new_columns: list[list[tuple[str, float]]] = []
        col_idx = 0
        for bch, nch in zip(aligned_backbone, aligned_new):
            votes = list(columns[col_idx]) if bch != GAP else []
            if nch != GAP:
                votes.append((nch, weight))
            if votes:
                new_columns.append(votes)
            if bch != GAP:
                col_idx += 1
        columns = new_columns

    return "".join(_majority_char(col) for col in columns)


def _majority_char(votes: list[tuple[str, float]]) -> str:
    tally: Counter = Counter()
    for ch, conf in votes:
        if ch != GAP:
            tally[ch] += conf
    if not tally:
        return ""
    return tally.most_common(1)[0][0]


def _reconstruct_engine_hypothesis(dets: list[Detection]) -> tuple[str, float]:
    """Different engines segment text at different granularities (see
    group_by_region's docstring) — a single "region" group can contain
    several of ONE engine's word-level fragments alongside another
    engine's single line-level detection. Concatenating same-engine
    fragments left-to-right first (reading order) turns them back into
    one whole-region hypothesis, so cross-engine voting always compares
    whole hypotheses to whole hypotheses, never a fragment to a whole
    line — voting a lone word against a full line is meaningless."""
    ordered = sorted(dets, key=lambda d: d.bbox[0])
    text = " ".join(d.text for d in ordered)
    confidence = sum(d.confidence for d in ordered) / len(ordered)
    return text, confidence


def fuse(detections: list[Detection], weighted: bool = True) -> list[Detection]:
    """Combine detections from however many engines actually ran. With one
    engine, this is a pass-through (nothing to vote on); with 2+, spatial
    grouping + per-engine reconstruction + textual voting produces one
    consensus Detection per region. `weighted` is forwarded to
    `_vote_text` — see its docstring for the cross-engine confidence-
    calibration caveat that motivated adding this as a real option rather
    than an assumption."""
    fused = []
    for group in group_by_region(detections):
        by_engine: dict[str, list[Detection]] = {}
        for det in group:
            by_engine.setdefault(det.engine, []).append(det)

        if len(by_engine) == 1:
            (engine, dets), = by_engine.items()
            text, confidence = _reconstruct_engine_hypothesis(dets)
        else:
            # sorted by engine name: deterministic backbone/tie-break for
            # weighted=False, and irrelevant to weighted=True's own re-sort.
            hypotheses = {engine: _reconstruct_engine_hypothesis(dets) for engine, dets in sorted(by_engine.items())}
            text = _vote_text(list(hypotheses.values()), weighted=weighted)
            confidence = sum(conf for _, conf in hypotheses.values()) / len(hypotheses)

        xs1 = min(d.bbox[0] for d in group); ys1 = min(d.bbox[1] for d in group)
        xs2 = max(d.bbox[2] for d in group); ys2 = max(d.bbox[3] for d in group)
        engines = "+".join(sorted(by_engine.keys()))
        fused.append(Detection(text=text, confidence=confidence, bbox=(xs1, ys1, xs2, ys2), engine=engines))
    return fused
