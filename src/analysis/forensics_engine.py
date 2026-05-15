"""Unified forensic analysis orchestrator for image, video, and audio."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import DEEPFAKE_MODEL_REPO, HIGH_CONFIDENCE_THRESHOLD, SYNTHETIC_THRESHOLD
from src.models.detector import analyze_media, get_last_forensic_details
from src.processors.audio_processor import analyze_audio, load_audio
from src.processors.video_processor import analyze_video, load_video_metadata
from src.utils.errors import ForensicsError, MediaLoadError, ProcessingError


@dataclass
class ForensicsResult:
    media_type: str
    synthetic_probability: float
    verdict: str
    confidence: str
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class ForensicsEngine:
    """Synthetic media detection pipeline with graceful error aggregation."""

    def _verdict(self, score: float) -> tuple[str, str]:
        if score >= HIGH_CONFIDENCE_THRESHOLD:
            return "LIKELY SYNTHETIC", "high"
        if score >= SYNTHETIC_THRESHOLD:
            return "SUSPICIOUS", "medium"
        return "LIKELY AUTHENTIC", "low"

    def analyze_image(self, path: str | Path) -> ForensicsResult:
        errors: list[str] = []
        try:
            authenticity_pct = analyze_media(path)
            score = 1.0 - (authenticity_pct / 100.0)
            verdict, confidence = self._verdict(score)
            return ForensicsResult(
                media_type="image",
                synthetic_probability=round(score, 4),
                verdict=verdict,
                confidence=confidence,
                details={
                    "path": str(path),
                    "authenticity_confidence_pct": authenticity_pct,
                    "backbone": DEEPFAKE_MODEL_REPO,
                    "forensics": get_last_forensic_details(),
                },
                errors=errors,
            )
        except ForensicsError as exc:
            errors.append(str(exc))
            return ForensicsResult(
                media_type="image",
                synthetic_probability=0.0,
                verdict="ANALYSIS FAILED",
                confidence="none",
                details={"path": str(path)},
                errors=errors,
            )

    def analyze_video(self, path: str | Path) -> ForensicsResult:
        errors: list[str] = []
        try:
            meta = load_video_metadata(path)
            authenticity_pct = analyze_video(path)
            score = 1.0 - (authenticity_pct / 100.0)
            verdict, confidence = self._verdict(score)

            return ForensicsResult(
                media_type="video",
                synthetic_probability=round(score, 4),
                verdict=verdict,
                confidence=confidence,
                details={
                    **meta,
                    "authenticity_confidence_pct": authenticity_pct,
                    "sampling": "1_frame_per_second",
                    "backbone": DEEPFAKE_MODEL_REPO,
                },
                errors=errors,
            )
        except ForensicsError as exc:
            errors.append(str(exc))
            return ForensicsResult(
                media_type="video",
                synthetic_probability=0.0,
                verdict="ANALYSIS FAILED",
                confidence="none",
                details={"path": str(path)},
                errors=errors,
            )

    def analyze_audio(self, path: str | Path) -> ForensicsResult:
        errors: list[str] = []
        try:
            y, sr = load_audio(path)
            features = analyze_audio(y, sr)
            score = features["heuristic_synthetic_score"]
            verdict, confidence = self._verdict(score)
            return ForensicsResult(
                media_type="audio",
                synthetic_probability=score,
                verdict=verdict,
                confidence=confidence,
                details=features,
                errors=errors,
            )
        except ForensicsError as exc:
            errors.append(str(exc))
            return ForensicsResult(
                media_type="audio",
                synthetic_probability=0.0,
                verdict="ANALYSIS FAILED",
                confidence="none",
                details={"path": str(path)},
                errors=errors,
            )

    def analyze(self, path: str | Path) -> ForensicsResult:
        """Auto-detect media type from extension."""
        try:
            path = Path(path)
            suffix = path.suffix.lower()
            image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
            video_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}
            audio_ext = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

            if suffix in image_ext:
                return self.analyze_image(path)
            if suffix in video_ext:
                return self.analyze_video(path)
            if suffix in audio_ext:
                return self.analyze_audio(path)
            raise MediaLoadError("Unsupported file extension", suffix)
        except MediaLoadError as exc:
            return ForensicsResult(
                media_type="unknown",
                synthetic_probability=0.0,
                verdict="ANALYSIS FAILED",
                confidence="none",
                details={"path": str(path)},
                errors=[str(exc)],
            )
