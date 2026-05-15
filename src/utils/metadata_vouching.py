"""Metadata vouching — trust bonus from verified camera EXIF hardware."""

from __future__ import annotations

from config import (
    METADATA_VOUCH_NOTE,
    METADATA_VOUCH_TRUST_BOOST,
    METADATA_VOUCH_W_AI,
    METADATA_VOUCH_W_META,
    TRUSTED_CAMERA_MANUFACTURERS,
)
from src.processors.metadata_extractor import MetadataReport


def extract_camera_make_model(report: MetadataReport | None) -> tuple[str | None, str | None]:
    """Read Make and Model from a Metadata Explorer report."""
    if report is None:
        return None, None
    return report.camera_make, report.camera_model


def is_trusted_camera_hardware(make: str | None, model: str | None) -> bool:
    """True when EXIF Make/Model matches a known camera manufacturer."""
    haystack = f"{make or ''} {model or ''}".lower()
    if not haystack.strip():
        return False
    return any(brand in haystack for brand in TRUSTED_CAMERA_MANUFACTURERS)


def compute_metadata_vouching(
    s_model: float,
    report: MetadataReport | None,
) -> tuple[float, dict]:
    """
    Apply metadata vouching to AI authenticity score S_model in [0, 1].

    A_weighted = (w_ai * S_model) + (w_meta * B_metadata)
    If trusted hardware: A_final = min(1, A_weighted + 20% trust boost)
    """
    s_model = min(1.0, max(0.0, float(s_model)))
    make, model = extract_camera_make_model(report)
    trusted = is_trusted_camera_hardware(make, model)
    b_metadata = 1.0 if trusted else 0.0

    a_weighted = (METADATA_VOUCH_W_AI * s_model) + (METADATA_VOUCH_W_META * b_metadata)
    a_final = a_weighted
    vouch_note: str | None = None

    if trusted:
        a_final = min(1.0, a_weighted + METADATA_VOUCH_TRUST_BOOST)
        vouch_note = METADATA_VOUCH_NOTE

    return a_final, {
        "camera_make": make,
        "camera_model": model,
        "trusted_hardware": trusted,
        "b_metadata": b_metadata,
        "s_model": round(s_model, 4),
        "a_weighted": round(a_weighted, 4),
        "a_final": round(a_final, 4),
        "w_ai": METADATA_VOUCH_W_AI,
        "w_meta": METADATA_VOUCH_W_META,
        "trust_boost_applied": METADATA_VOUCH_TRUST_BOOST if trusted else 0.0,
        "vouch_note": vouch_note,
    }
