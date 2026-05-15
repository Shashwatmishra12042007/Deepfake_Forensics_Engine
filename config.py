"""Configuration for the Synthetic Media Detection Engine."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Sticky footer attribution (edit with your name)
FOOTER_AUTHOR_NAME = "Shashwat Mishra"
FOOTER_UNIVERSITY = "VIT Bhopal University"

# Optional sidebar brand image (first existing file is used)
SIDEBAR_LOGO_CANDIDATES: tuple[Path, ...] = (
    ASSETS_DIR / "sidebar_logo.png",
    ASSETS_DIR / "logo.png",
    ASSETS_DIR / "logo.jpg",
    ASSETS_DIR / "logo.jpeg",
    ASSETS_DIR / "logo.webp",
)
DEFAULT_MODEL_PATH = MODELS_DIR / "synthetic_detector.pt"

# FaceForensics++ C23 fine-tuned EfficientNet-B0 (Xicor9 / Hugging Face)
DEEPFAKE_MODEL_URL = (
    "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/resolve/main/"
    "efficientnet_b0_ffpp_c23.pth"
)
DEEPFAKE_MODEL_PATH = MODELS_DIR / "efficientnet_b0_ffpp_c23.pth"
DEEPFAKE_MODEL_REPO = "Xicor9/efficientnet-b0-ffpp-c23"

IMAGE_SIZE = (224, 224)
VIDEO_SAMPLE_FRAMES = 16
AUDIO_SAMPLE_RATE = 22050
AUDIO_DURATION_SEC = 10.0

SYNTHETIC_THRESHOLD = 0.55
HIGH_CONFIDENCE_THRESHOLD = 0.75

# Blend: (1 - w) * deepfake CNN + w * forensic heuristic authenticity (default)
FORENSIC_BLEND_WEIGHT = 0.25

# Sigmoid calibration: raw P(fake) gated — synthetic label only if raw P > threshold
SIGMOID_CALIBRATION_THRESHOLD = 0.85
SIGMOID_CALIBRATION_STEEPNESS = 14.0

# Slight Gaussian blur before CNN (sigma) — neutralizes smartphone oversharpening
GAUSSIAN_NORMALIZE_SIGMA_DEFAULT = 0.85

# Metadata vouching: A_final = (w_ai * S_model) + (w_meta * B_metadata) [+ trust boost]
METADATA_VOUCH_W_AI = 0.8
METADATA_VOUCH_W_META = 0.2
METADATA_VOUCH_TRUST_BOOST = 0.20

TRUSTED_CAMERA_MANUFACTURERS: tuple[str, ...] = (
    "apple",
    "samsung",
    "google",
    "canon",
    "sony",
    "nikon",
    "fujifilm",
    "huawei",
    "oneplus",
    "xiaomi",
)

METADATA_VOUCH_NOTE = (
    "Device Hardware Verified: Reducing sensitivity for computational photography artifacts."
)

# Forensic Summary — weighted evidence channels (renormalized when a channel is N/A)
FORENSIC_EVIDENCE_W_VISUAL = 0.55
FORENSIC_EVIDENCE_W_AUDIO = 0.30
FORENSIC_EVIDENCE_W_METADATA = 0.15

# fastNlMeansDenoisingColored — low strength for smartphone sensor noise
DENOISE_H = 3
DENOISE_H_COLOR = 3
DENOISE_TEMPLATE_WINDOW_SIZE = 7
DENOISE_SEARCH_WINDOW_SIZE = 21

SENSOR_PROFILES: dict[str, dict] = {
    "standard_camera": {
        "display_name": "Standard Camera (Low Sensitivity)",
        "synthetic_label_threshold": 0.85,
        "sigmoid_threshold": 0.85,
        "sigmoid_steepness": 14.0,
        "gauge_warn_pct": 30.0,
        "high_confidence_synthetic_pct": 15.0,
        "forensic_blend_weight": 0.15,
        "gaussian_sigma": 0.85,
        "use_sharpen_heuristics": False,
    },
    "pro_raw": {
        "display_name": "Pro/Raw Image (High Sensitivity)",
        "synthetic_label_threshold": 0.55,
        "sigmoid_threshold": 0.55,
        "sigmoid_steepness": 10.0,
        "gauge_warn_pct": 45.0,
        "high_confidence_synthetic_pct": 25.0,
        "forensic_blend_weight": 0.30,
        "gaussian_sigma": 0.35,
        "use_sharpen_heuristics": True,
    },
}
