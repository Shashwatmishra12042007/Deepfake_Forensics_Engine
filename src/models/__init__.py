"""Model package — import detector before gradcam to avoid init-order issues."""

from src.models.detector import (
    CLASS_FAKE,
    CLASS_REAL,
    analyze_media,
    build_deepfake_efficientnet,
    build_efficientnet_binary,
    download_deepfake_weights,
    get_image_transform,
    finalize_authenticity_with_vouching,
    get_last_forensic_details,
    get_last_vouching_details,
    is_denoise_enabled,
    load_detector,
    set_denoise_enabled,
    predict_authenticity_probability,
    predict_synthetic_score,
)
from src.models.gradcam import CLASS_LABELS, generate_gradcam

__all__ = [
    "CLASS_FAKE",
    "CLASS_REAL",
    "CLASS_LABELS",
    "analyze_media",
    "build_deepfake_efficientnet",
    "build_efficientnet_binary",
    "download_deepfake_weights",
    "generate_gradcam",
    "is_denoise_enabled",
    "set_denoise_enabled",
    "get_image_transform",
    "finalize_authenticity_with_vouching",
    "get_last_forensic_details",
    "get_last_vouching_details",
    "load_detector",
    "predict_authenticity_probability",
    "predict_synthetic_score",
]
