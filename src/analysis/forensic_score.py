"""Weighted forensic confidence from Visual, Audio, and Metadata evidence."""

from __future__ import annotations

from dataclasses import dataclass

from config import (
    FORENSIC_EVIDENCE_W_AUDIO,
    FORENSIC_EVIDENCE_W_METADATA,
    FORENSIC_EVIDENCE_W_VISUAL,
)
from src.analysis.forensic_report import AnalysisContext
from src.processors.metadata_extractor import MetadataReport


@dataclass
class EvidenceBreakdown:
    visual_pct: float | None
    audio_pct: float | None
    metadata_pct: float
    visual_weight: float
    audio_weight: float
    metadata_weight: float
    weighted_base_pct: float
    final_pct: float
    vouching_delta_pct: float


def metadata_evidence_pct(
    report: MetadataReport | None,
    vouching: dict | None,
) -> float:
    """Metadata-channel authenticity in [0, 100]."""
    if vouching is not None and "b_metadata" in vouching:
        return float(vouching["b_metadata"]) * 100.0
    if report is None:
        return 50.0
    red_count = len(report.red_flag_summary)
    if red_count == 0:
        return 65.0
    return max(15.0, 100.0 - red_count * 12.0)


def compute_evidence_breakdown(ctx: AnalysisContext) -> EvidenceBreakdown:
    """
    Weighted average of active evidence channels.

    Visual / Audio use the pre-vouching AI score. Metadata uses hardware trust
    (B_metadata) or red-flag heuristics. ``vouching_delta_pct`` is the change
    from the weighted base to the post-vouching final score.
    """
    visual_pct: float | None = None
    audio_pct: float | None = None

    if ctx.media_type in {"image", "video"}:
        visual_pct = ctx.ai_authenticity_pct
    elif ctx.media_type == "audio":
        audio_pct = ctx.ai_authenticity_pct

    metadata_pct = metadata_evidence_pct(ctx.metadata_report, ctx.metadata_vouching)

    channel_weights: dict[str, tuple[float, float]] = {
        "metadata": (metadata_pct, FORENSIC_EVIDENCE_W_METADATA),
    }
    if visual_pct is not None:
        channel_weights["visual"] = (visual_pct, FORENSIC_EVIDENCE_W_VISUAL)
    if audio_pct is not None:
        channel_weights["audio"] = (audio_pct, FORENSIC_EVIDENCE_W_AUDIO)

    total_w = sum(w for _, w in channel_weights.values())
    weighted_base = sum(
        score * (weight / total_w) for score, weight in channel_weights.values()
    )

    final_pct = float(ctx.authenticity_pct)
    vouching_delta = round(final_pct - weighted_base, 2)

    def _effective(key: str, default_w: float) -> float:
        if key not in channel_weights:
            return 0.0
        return channel_weights[key][1] / total_w

    return EvidenceBreakdown(
        visual_pct=visual_pct,
        audio_pct=audio_pct,
        metadata_pct=round(metadata_pct, 2),
        visual_weight=round(_effective("visual", FORENSIC_EVIDENCE_W_VISUAL), 3),
        audio_weight=round(_effective("audio", FORENSIC_EVIDENCE_W_AUDIO), 3),
        metadata_weight=round(_effective("metadata", FORENSIC_EVIDENCE_W_METADATA), 3),
        weighted_base_pct=round(weighted_base, 2),
        final_pct=round(final_pct, 2),
        vouching_delta_pct=vouching_delta,
    )
