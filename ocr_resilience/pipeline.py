"""Top-level orchestrator: quality assessment -> targeted classical-CV
preprocessing -> quality-aware engine routing -> run engine(s) -> fuse.
This is the "layer built on top of existing OCR engines" — it never
reimplements text detection/recognition itself.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .engines import AVAILABLE_ENGINES, Detection, OCREngine
from .fusion import fuse
from .postprocessing import postprocess_text
from .preprocess import apply_single_step, build_pipeline
from .quality import QualityReport, assess
from .router import RoutingDecision, decide


def _reading_order(detections: list[Detection]) -> list[Detection]:
    """Group into text lines by vertical-center proximity, THEN sort each
    line left-to-right — NOT a naive (y_min, x_min) global sort.

    A naive global sort was tried first and is a real, confirmed-live bug:
    cursive/italic fonts (and mixed-case text generally, via ascenders/
    descenders) give words on the SAME visual line noticeably different
    y_min values — e.g. "due"/"ow"/"June" at y_min=38 but "Payment" at
    y_min=47 and "15th" at y_min=48, all on one line. Sorting by y_min
    directly split that single line into two groups and produced "due ow
    June Payment 15th" instead of "Payment due on 15th June" — caught by
    the benchmark showing an UNDEGRADED ("clean") image regress to 0.83
    CER after this function was added, i.e. this exact fix introduced a
    new bug while fixing the original scrambled-multi-line-order one.
    Clustering by vertical CENTER with a height-relative tolerance (half
    the median box height) is robust to that per-word baseline variance.
    Known limitation: still a single-column heuristic, not full layout
    analysis — genuine multi-column documents need column detection
    first, out of scope here."""
    return [det for line in _group_into_lines(detections) for det in line]


def _group_into_lines(detections: list[Detection]) -> list[list[Detection]]:
    """The clustering half of `_reading_order` — split out so `OCRResult`
    can also use it to reconstruct TEXT correctly (words on the same line
    joined by spaces, lines joined by newlines). `_reading_order` alone
    only fixes ORDER; `OCRResult.text` used to flatten every detection
    with `"\\n"` regardless of which line it came from, so a single line
    made of several word-level detections (e.g. Tesseract's default
    per-word boxes) rendered as one word per output line instead of one
    space-joined line — invisible until now because the benchmark harness
    always bypassed `.text` and space-joined `result.detections` directly
    itself, so this never showed up in a CER/WER number."""
    if not detections:
        return []

    heights = [d.bbox[3] - d.bbox[1] for d in detections]
    tolerance = (sorted(heights)[len(heights) // 2] / 2) or 1.0

    def y_center(d: Detection) -> float:
        return (d.bbox[1] + d.bbox[3]) / 2

    by_y = sorted(detections, key=y_center)
    lines: list[list[Detection]] = []
    line_running_mean_y: list[float] = []
    for det in by_y:
        yc = y_center(det)
        if lines and abs(yc - line_running_mean_y[-1]) <= tolerance:
            lines[-1].append(det)
            n = len(lines[-1])
            line_running_mean_y[-1] += (yc - line_running_mean_y[-1]) / n
        else:
            lines.append([det])
            line_running_mean_y.append(yc)

    return [sorted(line, key=lambda d: d.bbox[0]) for line in lines]


@dataclass
class OCRResult:
    detections: list[Detection]
    quality: QualityReport
    routing: RoutingDecision
    preprocessing_steps: list[str]
    timing_sec: dict[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Words on the same visual line joined by a space, lines joined
        by a newline — reconstructed via `_group_into_lines` from the
        detections' own bounding boxes, independent of whatever order
        `self.detections` happens to already be in."""
        lines = _group_into_lines(self.detections)
        return "\n".join(" ".join(d.text for d in line) for line in lines)

    @property
    def raw_text(self) -> str:
        """Exactly what the engine(s)/fusion layer produced, before any
        text-level post-processing — always available alongside
        `processed_text` so post-processing never discards the original."""
        return self.text

    @property
    def processed_text(self) -> str:
        return postprocess_text(self.text)

    @property
    def confidence(self) -> float:
        """Mean per-detection confidence. 0.0 for a page with no detections
        (rather than raising) — an empty page is a valid, if uninteresting,
        result."""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

    @property
    def bounding_boxes(self) -> list[tuple[int, int, int, int]]:
        return [d.bbox for d in self.detections]

    @property
    def engine_used(self) -> str:
        names = sorted({name for d in self.detections for name in d.engine.split("+")})
        return "+".join(names)

    @property
    def preprocessing_pipeline(self) -> list[str]:
        return self.preprocessing_steps

    @property
    def processing_time(self) -> float:
        return sum(self.timing_sec.values())

    def to_dict(self, include_boxes: bool = True) -> dict:
        detections = [
            {"text": d.text, "confidence": d.confidence, "engine": d.engine, **({"bbox": list(d.bbox)} if include_boxes else {})}
            for d in self.detections
        ]
        result = {
            "raw_text": self.raw_text,
            "processed_text": self.processed_text,
            "confidence": self.confidence,
            "engine_used": self.engine_used,
            "preprocessing_pipeline": self.preprocessing_pipeline,
            "processing_time": self.processing_time,
            "detections": detections,
            "routing_reason": self.routing.reason,
            "timing_sec": self.timing_sec,
        }
        if include_boxes:
            result["bounding_boxes"] = [list(b) for b in self.bounding_boxes]
        return result


class OCRPipeline:
    def __init__(self, engines: dict[str, OCREngine] | None = None, degradation_threshold: int = 2):
        self._engines = engines or {}
        self._degradation_threshold = degradation_threshold

    @classmethod
    def with_engines(cls, engine_names: list[str], **engine_kwargs) -> OCRPipeline:
        engines = {}
        for name in engine_names:
            if name not in AVAILABLE_ENGINES:
                raise ValueError(f"Unknown engine '{name}'. Available: {list(AVAILABLE_ENGINES)}")
            engines[name] = AVAILABLE_ENGINES[name](**engine_kwargs.get(name, {}))
        return cls(engines=engines)

    def run(
        self,
        image: np.ndarray | str,
        skip_preprocessing: bool = False,
        force_ensemble: bool | None = None,
        force_step: str | None = None,
        fusion_weighted: bool = True,
        min_confidence_for_fallback: float | None = None,
        debug_dir: str | None = None,
    ) -> OCRResult:
        """Run the full pipeline on one image.

        `skip_preprocessing`, `force_ensemble`, and `force_step` exist to
        make ablation experiments (does adaptive preprocessing/multi-engine
        routing/one specific operator actually help?) a first-class,
        reusable capability instead of a benchmark-only hack:
        `skip_preprocessing=True` reproduces the "Baseline OCR" cell,
        `force_step="deskew"` reproduces "Baseline + deskew" (that one
        operator applied unconditionally, see `preprocess.apply_single_step`),
        `force_ensemble=False` reproduces "Baseline + adaptive preprocessing"
        without routing, `force_ensemble=True` reproduces "Baseline +
        multi-engine selection" unconditionally. `fusion_weighted=False`
        reproduces "Baseline + unweighted fusion" (see `fusion.fuse`'s
        docstring for why confidence-weighted voting isn't unconditionally
        correct — cross-engine confidence scales aren't verified
        comparable). Default (all None/False/True) is the normal adaptive
        behavior. `force_step` and `skip_preprocessing=True` are mutually
        exclusive by construction — the former is checked first if both
        are somehow passed.

        `min_confidence_for_fallback`: if set, and the first attempt's
        mean confidence falls below this threshold AND not every
        available engine already ran, escalate to a second pass using
        every available engine, and keep whichever attempt has higher
        confidence. This is a genuine second OCR attempt with a different
        configuration (more engines), not a preprocessing retry — costs
        roughly double the latency only when it actually triggers. `None`
        (default) disables it entirely, at zero overhead.

        `debug_dir`: if set, writes original.png/preprocessed.png/
        annotated.png (detected boxes + text + confidence drawn on the
        original) to this directory — see `ocr_resilience.debug`. `None`
        (default) skips this entirely.
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        if image is None:
            raise ValueError("Could not load image.")
        if not self._engines:
            raise ValueError("No OCR engines available. Construct with OCRPipeline.with_engines([...]).")

        timing = {}

        t0 = time.perf_counter()
        report = assess(image)
        timing["quality_assessment"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if force_step is not None:
            processed, steps = apply_single_step(image, force_step, report)
        elif skip_preprocessing:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            processed, steps = gray, []
        else:
            processed, steps = build_pipeline(image, report)
        timing["preprocessing"] = time.perf_counter() - t0

        if force_ensemble is None:
            routing = decide(report, list(self._engines.keys()), self._degradation_threshold)
        elif force_ensemble:
            routing = RoutingDecision(engines_to_run=list(self._engines.keys()), reason="forced ensemble (ablation override)")
        else:
            routing = RoutingDecision(engines_to_run=[next(iter(self._engines))], reason="forced single engine (ablation override)")

        ordered, engine_timing = self._run_engines_and_fuse(processed, routing.engines_to_run, fusion_weighted)
        timing.update(engine_timing)
        result = OCRResult(detections=ordered, quality=report, routing=routing, preprocessing_steps=steps, timing_sec=timing)

        if (
            min_confidence_for_fallback is not None
            and result.confidence < min_confidence_for_fallback
            and len(routing.engines_to_run) < len(self._engines)
        ):
            result = self._confidence_fallback(
                processed, report, routing, steps, timing, result, fusion_weighted, min_confidence_for_fallback,
            )

        if debug_dir is not None:
            from .debug import export_debug_bundle
            export_debug_bundle(image, processed, result.detections, debug_dir)

        return result

    def _confidence_fallback(
        self,
        processed: np.ndarray,
        report: QualityReport,
        first_routing: RoutingDecision,
        steps: list[str],
        timing: dict[str, float],
        first_result: OCRResult,
        fusion_weighted: bool,
        threshold: float,
    ) -> OCRResult:
        """Ranked escalation, not a straight jump to the full ensemble:
        tier 1 adds just the single best-ranked fallback engine (see
        `engine_selection.rank_fallback_chain`) to the first pass; only if
        THAT still doesn't clear the confidence threshold does tier 2 (the
        full ensemble) run. With exactly two engines total, tier 1 and the
        full ensemble are the same set — the two-tier structure only does
        real extra work (and costs real extra latency) with three or more
        engines available."""
        from .engine_selection import rank_fallback_chain

        primary = first_routing.engines_to_run[0]
        all_engines = list(self._engines.keys())
        fallback_chain = rank_fallback_chain(primary, all_engines)

        candidates = [first_result]

        if fallback_chain and len(first_routing.engines_to_run) + 1 < len(all_engines):
            tier1_engines = [primary, fallback_chain[0]]
            tier1_ordered, tier1_timing = self._run_engines_and_fuse(processed, tier1_engines, fusion_weighted)
            timing.update({f"fallback_tier1_{k}": v for k, v in tier1_timing.items()})
            tier1_routing = RoutingDecision(
                engines_to_run=tier1_engines,
                reason=(
                    f"confidence-based fallback tier 1: first pass confidence {first_result.confidence:.2f} "
                    f"< threshold {threshold} -> escalate to ranked fallback engine '{fallback_chain[0]}' "
                    f"(first pass: {first_routing.reason})"
                ),
            )
            candidates.append(OCRResult(
                detections=tier1_ordered, quality=report, routing=tier1_routing,
                preprocessing_steps=steps, timing_sec=timing,
            ))

        if max(c.confidence for c in candidates) < threshold and len(all_engines) > len(candidates[-1].routing.engines_to_run):
            tier2_ordered, tier2_timing = self._run_engines_and_fuse(processed, all_engines, fusion_weighted)
            timing.update({f"fallback_tier2_{k}": v for k, v in tier2_timing.items()})
            tier2_routing = RoutingDecision(
                engines_to_run=all_engines,
                reason=(
                    f"confidence-based fallback tier 2 (final escalation): "
                    f"still below threshold {threshold} after tier 1 -> full ensemble"
                ),
            )
            candidates.append(OCRResult(
                detections=tier2_ordered, quality=report, routing=tier2_routing,
                preprocessing_steps=steps, timing_sec=timing,
            ))

        best = max(candidates, key=lambda r: r.confidence)
        if best is first_result:
            best.routing.reason += (
                f" (confidence-based fallback attempted through {len(candidates) - 1} escalation tier(s), "
                f"none improved on {first_result.confidence:.2f} — kept first pass)"
            )
        return best

    def _run_engines_and_fuse(
        self, processed: np.ndarray, engine_names: list[str], fusion_weighted: bool,
    ) -> tuple[list[Detection], dict[str, float]]:
        timing: dict[str, float] = {}
        all_detections: list[Detection] = []
        for name in engine_names:
            t0 = time.perf_counter()
            all_detections.extend(self._engines[name].recognize(processed))
            timing[f"engine:{name}"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        fused = fuse(all_detections, weighted=fusion_weighted) if len(engine_names) > 1 else all_detections
        timing["fusion"] = time.perf_counter() - t0

        return _reading_order(fused), timing

    def run_batch(self, images: list[np.ndarray | str], **run_kwargs) -> list[OCRResult]:
        """Run each image independently through `run()`. Fails fast on the
        first bad image (standard Python semantics) — a CLI or script
        driving many files should wrap each call individually if it wants
        one bad file to not abort the rest; see `ocr_resilience.cli` for
        that behavior."""
        return [self.run(image, **run_kwargs) for image in images]


_ENGINE_PRIORITY = ["tesseract", "easyocr", "paddleocr", "rapidocr"]


def _resolve_engine_names(engine: str) -> list[str]:
    if engine != "auto":
        return [name.strip() for name in engine.split(",") if name.strip()]

    import importlib.util

    _IMPORT_MODULE = {"tesseract": "pytesseract", "easyocr": "easyocr", "paddleocr": "paddleocr", "rapidocr": "rapidocr"}
    available = [name for name in _ENGINE_PRIORITY if importlib.util.find_spec(_IMPORT_MODULE[name]) is not None]
    if not available:
        raise ValueError(
            "engine='auto' found no installed OCR engine bindings. "
            "Install one of: pip install ocr-resilience[tesseract|easyocr|paddleocr]"
        )
    return available


class OCR:
    """Thin convenience wrapper around `OCRPipeline` matching the public
    API surface documented in the README (`OCR(engine=..., preprocessing=...)`
    / `ocr.predict(...)`). `OCRPipeline` remains the lower-level class for
    callers who want to pass pre-built engine instances or run ablations
    directly via `run(skip_preprocessing=..., force_ensemble=...)`.
    """

    def __init__(
        self,
        engine: str = "auto",
        preprocessing: str = "adaptive",
        return_boxes: bool = True,
        degradation_threshold: int = 2,
    ):
        if preprocessing not in ("adaptive", "none"):
            raise ValueError("preprocessing must be 'adaptive' or 'none'")
        self._skip_preprocessing = preprocessing == "none"
        self._return_boxes = return_boxes
        self._pipeline = OCRPipeline.with_engines(_resolve_engine_names(engine))
        self._pipeline._degradation_threshold = degradation_threshold

    def predict(self, image: np.ndarray | str, debug_dir: str | None = None) -> OCRResult:
        return self._pipeline.run(image, skip_preprocessing=self._skip_preprocessing, debug_dir=debug_dir)

    def predict_batch(self, images: list[np.ndarray | str]) -> list[OCRResult]:
        return self._pipeline.run_batch(images, skip_preprocessing=self._skip_preprocessing)

    def predict_dict(self, image: np.ndarray | str, debug_dir: str | None = None) -> dict:
        """`predict()` plus serialization, honoring `return_boxes` from the
        constructor — the CLI uses this for its JSON output."""
        return self.predict(image, debug_dir=debug_dir).to_dict(include_boxes=self._return_boxes)
