"""Quality-aware engine routing. The whole point of assessing image
quality up front (quality.py) is to NOT pay ensemble cost on easy images —
a clean, well-lit, printed document should hit one fast engine and return;
only genuinely hard cases (multiple degradations stacked, or handwriting)
are worth the latency of running everything and fusing (fusion.py).
"""
from dataclasses import dataclass

from .engine_selection import select_primary_engine
from .quality import QualityReport


@dataclass
class RoutingDecision:
    engines_to_run: list[str]
    reason: str


def decide(report: QualityReport, available_engines: list[str], degradation_threshold: int = 2) -> RoutingDecision:
    if not available_engines:
        raise ValueError("No OCR engines available to route to.")

    degradation_flags = [report.is_blurry, report.is_noisy, report.is_impulse_noisy, report.is_low_contrast, report.is_skewed]
    degraded_count = sum(degradation_flags)

    if report.likely_handwritten or degraded_count >= degradation_threshold:
        reasons = []
        if report.likely_handwritten:
            reasons.append("likely handwritten")
        if degraded_count >= degradation_threshold:
            reasons.append(f"{degraded_count} degradation flags set (blur/noise/impulse-noise/contrast/skew)")
        return RoutingDecision(
            engines_to_run=list(available_engines),
            reason=f"hard case ({', '.join(reasons)}) -> ensemble all {len(available_engines)} available engine(s)",
        )

    engine, selection_reason = select_primary_engine(report, available_engines)
    return RoutingDecision(
        engines_to_run=[engine],
        reason=f"clean/mildly-degraded image ({degraded_count} flags) -> single engine, {selection_reason}",
    )
