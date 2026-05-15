"""Forensic pre-processing: sharpening, Moiré patterns, spectral residuals."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from config import (
    DENOISE_H,
    DENOISE_H_COLOR,
    DENOISE_SEARCH_WINDOW_SIZE,
    DENOISE_TEMPLATE_WINDOW_SIZE,
)
from src.utils.errors import ProcessingError


def apply_denoise_colored(bgr: np.ndarray) -> np.ndarray:
    """Light non-local means denoising to reduce smartphone sensor noise before CNN."""
    try:
        if bgr is None or bgr.size == 0:
            raise ProcessingError("Empty or invalid frame array")
        denoised = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            DENOISE_H,
            DENOISE_H_COLOR,
            DENOISE_TEMPLATE_WINDOW_SIZE,
            DENOISE_SEARCH_WINDOW_SIZE,
        )
        return np.clip(denoised, 0, 255).astype(np.uint8)
    except cv2.error as exc:
        raise ProcessingError("Colored denoising failed", str(exc)) from exc


def apply_gaussian_normalize(bgr: np.ndarray, sigma: float = 0.85) -> np.ndarray:
    """Very slight Gaussian blur to reduce smartphone oversharpening before CNN input."""
    try:
        if bgr is None or bgr.size == 0:
            raise ProcessingError("Empty or invalid frame array")
        blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return np.clip(blurred, 0, 255).astype(np.uint8)
    except cv2.error as exc:
        raise ProcessingError("Gaussian normalization failed", str(exc)) from exc


def sharpen_image(bgr: np.ndarray, amount: float = 1.25) -> np.ndarray:
    """Unsharp mask to accentuate compression / synthesis boundary artifacts."""
    try:
        blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=3.0)
        sharpened = cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except cv2.error as exc:
        raise ProcessingError("Image sharpening failed", str(exc)) from exc


def detect_moire_score(bgr: np.ndarray) -> float:
    """
    Estimate Moiré strength via periodic peaks in the 2D FFT magnitude spectrum.
    Returns synthetic-likelihood score in [0, 1].
    """
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        spectrum = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.log1p(np.abs(spectrum))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        radius = int(min(h, w) * 0.08)
        magnitude[cy - radius : cy + radius, cx - radius : cx + radius] = 0.0

        mean_val = float(np.mean(magnitude))
        peak_val = float(np.max(magnitude))
        if mean_val < 1e-6:
            return 0.0

        peak_ratio = peak_val / mean_val
        return float(min(1.0, max(0.0, (peak_ratio - 4.0) / 12.0)))
    except (cv2.error, ValueError) as exc:
        raise ProcessingError("Moiré analysis failed", str(exc)) from exc


def spectral_residual_score(bgr: np.ndarray) -> float:
    """
    Spectral residual energy (log-amplitude minus smoothed envelope).
    Elevated residuals often correlate with synthetic re-sampling artifacts.
    """
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        spectrum = np.fft.fft2(gray)
        log_amplitude = np.log(np.abs(spectrum) + 1e-8)
        smoothed = uniform_filter(log_amplitude, size=5)
        residual = log_amplitude - smoothed
        energy = float(np.mean(np.abs(residual)))
        return float(min(1.0, max(0.0, (energy - 0.35) / 0.45)))
    except (ValueError, TypeError) as exc:
        raise ProcessingError("Spectral residual analysis failed", str(exc)) from exc


def run_forensic_preprocessing(bgr: np.ndarray, *, sharpen: bool = True) -> dict:
    """
    Pre-process frame for deepfake forensics.
    Returns working BGR and heuristic synthetic scores.
    """
    try:
        if bgr is None or bgr.size == 0:
            raise ProcessingError("Empty or invalid frame array")

        sharpened = sharpen_image(bgr) if sharpen else bgr.copy()
        moire = detect_moire_score(sharpened)
        spectral = spectral_residual_score(sharpened)
        synthetic = min(1.0, max(0.0, 0.45 * moire + 0.55 * spectral))

        return {
            "sharpened_bgr": sharpened,
            "moire_score": round(moire, 4),
            "spectral_residual_score": round(spectral, 4),
            "heuristic_synthetic_score": round(synthetic, 4),
        }
    except ProcessingError:
        raise
    except Exception as exc:
        raise ProcessingError("Forensic pre-processing failed", str(exc)) from exc
