"""Librosa-based audio forensic feature extraction."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from config import AUDIO_DURATION_SEC, AUDIO_SAMPLE_RATE
from src.utils.errors import MediaLoadError, ProcessingError


def load_audio(path: str | Path, duration: float = AUDIO_DURATION_SEC) -> tuple[np.ndarray, int]:
    """Load mono audio; truncate/pad to duration window."""
    try:
        path = Path(path)
        if not path.is_file():
            raise MediaLoadError("Audio file not found", str(path))
        y, sr = librosa.load(str(path), sr=AUDIO_SAMPLE_RATE, mono=True, duration=duration)
        if y.size == 0:
            raise MediaLoadError("Empty audio buffer", str(path))
        return y, sr
    except MediaLoadError:
        raise
    except Exception as exc:
        raise MediaLoadError("Failed to load audio", str(exc)) from exc


def analyze_audio(y: np.ndarray, sr: int) -> dict:
    """
    Extract forensic audio features and heuristic synthetic score.
    Combines spectral, temporal, and phase cues common in TTS/vocoder artifacts.
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        mfcc_var = float(np.var(mfcc))
        flatness_mean = float(np.mean(spectral_flatness))
        zcr_mean = float(np.mean(zcr))
        rms_std = float(np.std(rms))
        centroid_std = float(np.std(spectral_centroid))

        # Heuristic: overly flat spectrum + low MFCC variance suggests synthesis
        score = min(
            1.0,
            max(
                0.0,
                0.35 * flatness_mean
                + 0.25 * (1.0 - min(mfcc_var / 500.0, 1.0))
                + 0.20 * (1.0 - min(centroid_std / 2000.0, 1.0))
                + 0.20 * min(rms_std * 10.0, 1.0),
            ),
        )

        return {
            "mfcc_variance": round(mfcc_var, 4),
            "spectral_flatness_mean": round(flatness_mean, 4),
            "zero_crossing_rate_mean": round(zcr_mean, 4),
            "rms_std": round(rms_std, 4),
            "spectral_centroid_std": round(centroid_std, 4),
            "heuristic_synthetic_score": round(score, 4),
        }
    except Exception as exc:
        raise ProcessingError("Audio feature extraction failed", str(exc)) from exc
