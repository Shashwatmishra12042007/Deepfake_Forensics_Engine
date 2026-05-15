"""
Deepfake detector: FaceForensics++ EfficientNet-B0 + forensic pre-processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from src.models.loader import (
    ensure_model,
    get_model_source,
    load_detector,
    predict_authenticity_probability,
    predict_synthetic_score,
)
from src.utils.calibration import sigmoid_calibrate_synthetic
from src.utils.errors import MediaLoadError, ModelError, ProcessingError
from src.utils.sensor_profile import (
    get_active_sensor_profile_key,
    get_sensor_profile_config,
    is_synthetic_label,
)
from src.processors.metadata_extractor import MetadataReport
from src.utils.metadata_vouching import compute_metadata_vouching
from src.utils.vision import preprocess_bgr_for_model

_last_forensic_details: dict | None = None
_last_vouching_details: dict | None = None
_denoise_enabled: bool = False


def set_denoise_enabled(enabled: bool) -> None:
    """Enable/disable fastNlMeans denoising before visual analysis."""
    global _denoise_enabled
    _denoise_enabled = bool(enabled)


def is_denoise_enabled() -> bool:
    return _denoise_enabled


def _fuse_authenticity(
    model_authentic_cal: float,
    forensic_synthetic: float,
    forensic_blend_weight: float,
) -> float:
    """Blend calibrated CNN authenticity with forensic heuristic authenticity."""
    forensic_authentic = 1.0 - forensic_synthetic
    w = forensic_blend_weight
    blended = (1.0 - w) * model_authentic_cal + w * forensic_authentic
    return min(1.0, max(0.0, blended))


def _score_bgr_frame(
    bgr: np.ndarray,
    sensor_profile: str | None = None,
    denoise: bool | None = None,
) -> float:
    """Forensic pre-process, calibrated CNN inference, fused authenticity %."""
    global _last_forensic_details
    profile_key = sensor_profile or get_active_sensor_profile_key()
    profile = get_sensor_profile_config(profile_key)
    use_denoise = _denoise_enabled if denoise is None else denoise

    try:
        tensor, meta = preprocess_bgr_for_model(
            bgr,
            sensor_profile=profile_key,
            denoise=use_denoise,
        )
        _last_forensic_details = {
            **meta,
            "model_source": get_model_source(),
            "sensor_profile_key": profile_key,
        }

        try:
            model, device = ensure_model()
            p_authentic_raw = predict_authenticity_probability(model, tensor, device)
            p_fake_raw = 1.0 - p_authentic_raw
        except ModelError:
            synthetic = meta.get("heuristic_synthetic_score", 0.5)
            p_authentic_raw = 1.0 - synthetic
            p_fake_raw = synthetic
            _last_forensic_details["model_source"] = "forensic_only (weights unavailable)"

        p_fake_cal = sigmoid_calibrate_synthetic(
            p_fake_raw,
            threshold=float(profile["sigmoid_threshold"]),
            steepness=float(profile["sigmoid_steepness"]),
        )
        model_authentic_cal = 1.0 - p_fake_cal

        synthetic_score = meta.get("heuristic_synthetic_score", 0.0)
        fused = _fuse_authenticity(
            model_authentic_cal,
            synthetic_score,
            float(profile["forensic_blend_weight"]),
        )

        _last_forensic_details.update(
            {
                "p_authentic_raw": round(p_authentic_raw, 4),
                "p_fake_raw": round(p_fake_raw, 4),
                "p_fake_calibrated": round(p_fake_cal, 4),
                "model_authenticity_calibrated_pct": round(model_authentic_cal * 100.0, 2),
                "fused_authenticity_pct": round(fused * 100.0, 2),
                "labeled_synthetic": is_synthetic_label(p_fake_raw, profile_key),
                "calibration": "sigmoid",
            }
        )

        return round(fused * 100.0, 2)
    except (ModelError, ProcessingError):
        raise
    except (cv2.error, ValueError, TypeError) as exc:
        raise ProcessingError("Frame scoring failed", str(exc)) from exc


def get_last_forensic_details() -> dict | None:
    """Metadata from the most recent analyze_media call."""
    return _last_forensic_details


def get_last_vouching_details() -> dict | None:
    """Metadata vouching details from the most recent finalize call."""
    return _last_vouching_details


def finalize_authenticity_with_vouching(
    ai_authenticity_pct: float,
    metadata_report: MetadataReport | None = None,
) -> tuple[float, dict]:
    """
    Combine AI authenticity with metadata hardware vouching.

    A_final = (w_ai * S_model) + (w_meta * B_metadata), plus +20% real boost if trusted.
    Returns (final_authenticity_pct, vouching_details).
    """
    global _last_vouching_details
    try:
        s_model = min(1.0, max(0.0, ai_authenticity_pct / 100.0))
        a_final, details = compute_metadata_vouching(s_model, metadata_report)
        _last_vouching_details = details
        if _last_forensic_details is not None:
            _last_forensic_details["metadata_vouching"] = details
            _last_forensic_details["ai_authenticity_pct"] = round(ai_authenticity_pct, 2)
            _last_forensic_details["final_authenticity_pct"] = round(a_final * 100.0, 2)
        return round(a_final * 100.0, 2), details
    except (ValueError, TypeError) as exc:
        raise ProcessingError("Metadata vouching failed", str(exc)) from exc


def analyze_media(
    source: Union[str, Path, np.ndarray],
    sensor_profile: str | None = None,
    denoise: bool | None = None,
) -> float:
    """
    Analyze image or BGR frame; return fused authenticity confidence [0, 100].

    Applies Gaussian normalization, sigmoid calibration on P(fake) (synthetic
    label only when raw P(fake) > profile threshold, default 0.85), and
  sensor-profile blending.
    """
    try:
        if isinstance(source, np.ndarray):
            return _score_bgr_frame(
                source, sensor_profile=sensor_profile, denoise=denoise
            )

        path = Path(source)
        if not path.is_file():
            raise MediaLoadError("Image file not found", str(path))

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise MediaLoadError("OpenCV could not decode image", str(path))
        return _score_bgr_frame(bgr, sensor_profile=sensor_profile, denoise=denoise)
    except (MediaLoadError, ProcessingError, ModelError):
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ProcessingError("analyze_media failed", str(exc)) from exc


from src.models.loader import (  # noqa: E402
    build_deepfake_efficientnet,
    build_efficientnet_binary,
    download_deepfake_weights,
)
from src.utils.constants import CLASS_FAKE, CLASS_REAL  # noqa: E402
from src.utils.transforms import get_image_transform  # noqa: E402

_ensure_model = ensure_model
