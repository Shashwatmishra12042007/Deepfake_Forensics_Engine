"""Forensic Reasoning Report — rule-based detection factor synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import librosa
import numpy as np
from PIL import Image

from src.utils.errors import ProcessingError

if TYPE_CHECKING:
    from src.analysis.forensic_score import EvidenceBreakdown
    from src.processors.metadata_extractor import MetadataReport

MSG_SPATIAL = (
    "Spatial Inconsistency: Detected irregular pixel blending in high-detail "
    "facial regions (Eyes/Mouth)."
)
MSG_SPECTRAL = (
    "Spectral Artifacts: Detected synthetic frequency patterns common in "
    "GAN-based voice synthesis."
)
MSG_TEMPORAL = (
    "Temporal Jitter: Detected frame-to-frame variance suggesting "
    "patch-based manipulation."
)
MSG_METADATA = (
    "Metadata Warning: Standard camera EXIF data is missing, suggesting the "
    "file was processed by external software."
)

# Grad-CAM heatmap activation in upper / lower face bands (normalized 0–1)
FACIAL_CAM_INTENSITY_THRESHOLD = 0.52
# Frame authenticity % std dev across 1 fps samples
TEMPORAL_JITTER_STD_THRESHOLD = 7.5
# High-frequency blockiness in mel-spectrogram
HF_BLOCK_VARIANCE_THRESHOLD = 18.0

EXIF_TAG_MAKE = 271
EXIF_TAG_MODEL = 272
EXIF_TAG_DATETIME = 306
EXIF_TAG_DATETIME_ORIGINAL = 36867


@dataclass
class AnalysisContext:
    media_type: str
    authenticity_pct: float
    file_path: Path | None = None
    gradcam_meta: dict | None = None
    frame_scores: list[float] = field(default_factory=list)
    audio_waveform: np.ndarray | None = None
    audio_sample_rate: int | None = None
    ai_authenticity_pct: float | None = None
    metadata_vouching: dict | None = None
    metadata_report: "MetadataReport | None" = None
    evidence: "EvidenceBreakdown | None" = None


def _facial_gradcam_high_intensity(heatmap: np.ndarray, height: int) -> bool:
    """True when top activation mass sits in eye or mouth vertical bands."""
    try:
        h = heatmap.shape[0]
        if h < 8:
            return False

        peak = float(np.max(heatmap))
        if peak < 1e-6:
            return False

        eye_band = heatmap[: int(h * 0.42), :]
        mouth_band = heatmap[int(h * 0.58) :, :]

        eye_ratio = float(np.mean(eye_band)) / peak
        mouth_ratio = float(np.mean(mouth_band)) / peak

        return (
            eye_ratio >= FACIAL_CAM_INTENSITY_THRESHOLD
            or mouth_ratio >= FACIAL_CAM_INTENSITY_THRESHOLD
        )
    except (ValueError, TypeError):
        return False


def _gradcam_focuses_facial_detail(focus_regions: list[str]) -> bool:
    text = " ".join(focus_regions).lower()
    eye_region = any(k in text for k in ("upper face", "mid face", "forehead"))
    mouth_region = any(k in text for k in ("lower face", "mouth", "chin"))
    return eye_region or mouth_region


def check_spatial_inconsistency(gradcam_meta: dict | None) -> bool:
    if not gradcam_meta:
        return False

    focus = gradcam_meta.get("focus_regions") or []
    if not _gradcam_focuses_facial_detail(focus):
        return False

    heatmap = gradcam_meta.get("activation_heatmap")
    if heatmap is not None:
        h = int(gradcam_meta.get("frame_height", heatmap.shape[0]))
        return _facial_gradcam_high_intensity(np.asarray(heatmap), h)

    return True


def check_spectral_artifacts(y: np.ndarray | None, sr: int | None) -> bool:
    if y is None or sr is None or y.size == 0:
        return False
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        hf_rows = mel_db[mel_db.shape[0] // 2 :, :]
        if hf_rows.shape[1] < 4:
            return False

        temporal_diff = np.diff(hf_rows, axis=1)
        block_variance = float(np.var(temporal_diff))

        hf_flatness = librosa.feature.spectral_flatness(y=y, sr=sr)
        flat_mean = float(np.mean(hf_flatness))

        return (
            block_variance >= HF_BLOCK_VARIANCE_THRESHOLD
            or flat_mean >= 0.42
        )
    except Exception:
        return False


def check_temporal_jitter(frame_scores: list[float]) -> bool:
    if len(frame_scores) < 2:
        return False
    try:
        return float(np.std(frame_scores)) >= TEMPORAL_JITTER_STD_THRESHOLD
    except (ValueError, TypeError):
        return False


def check_metadata_stripped(path: Path | None, media_type: str) -> bool:
    if path is None or not path.is_file():
        return True

    suffix = path.suffix.lower()
    try:
        if suffix in {".jpg", ".jpeg"}:
            with Image.open(path) as img:
                exif = img.getexif()
                if not exif:
                    return True
                keys = set(exif.keys())
                camera_keys = {
                    EXIF_TAG_MAKE,
                    EXIF_TAG_MODEL,
                    EXIF_TAG_DATETIME,
                    EXIF_TAG_DATETIME_ORIGINAL,
                }
                return len(keys & camera_keys) == 0

        if suffix == ".png":
            with Image.open(path) as img:
                exif = img.getexif()
                if exif and len(exif) >= 2:
                    return False
            return True

        if suffix in {".mp4", ".wav", ".mp3"}:
            return _container_metadata_stripped(path, suffix)

        return True
    except (OSError, ValueError, ProcessingError):
        return True


def _container_metadata_stripped(path: Path, suffix: str) -> bool:
    """Heuristic metadata check for non-JPEG containers."""
    try:
        if suffix == ".mp4":
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                cap.release()
                return True
            meta_keys = [
                cv2.CAP_PROP_CODEC_PIXEL_FORMAT,
            ]
            has_signal = any(cap.get(k) not in (-1, 0) for k in meta_keys)
            cap.release()
            return not has_signal

        if suffix == ".wav":
            import wave

            with wave.open(str(path), "rb") as wf:
                return wf.getnchannels() <= 0

        if suffix == ".mp3":
            header = path.read_bytes()[:4096]
            return b"ID3" not in header and b"TAG" not in header[-128:]
    except (OSError, ValueError, cv2.error):
        return True


def build_detection_factors(ctx: AnalysisContext) -> list[str]:
    """Return bulleted detection factor messages for the forensic summary."""
    factors: list[str] = []

    try:
        if ctx.media_type in {"image", "video"} and check_spatial_inconsistency(
            ctx.gradcam_meta
        ):
            factors.append(MSG_SPATIAL)

        if ctx.media_type == "audio" and check_spectral_artifacts(
            ctx.audio_waveform, ctx.audio_sample_rate
        ):
            factors.append(MSG_SPECTRAL)

        if ctx.media_type == "video" and check_temporal_jitter(ctx.frame_scores):
            factors.append(MSG_TEMPORAL)

        if check_metadata_stripped(ctx.file_path, ctx.media_type):
            factors.append(MSG_METADATA)

        if ctx.metadata_vouching and ctx.metadata_vouching.get("vouch_note"):
            factors.append(str(ctx.metadata_vouching["vouch_note"]))
    except Exception:
        pass

    return factors
