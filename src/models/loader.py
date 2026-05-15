"""EfficientNet-B0 weight loading and inference (no processor / gradcam imports)."""

from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from config import (
    DEEPFAKE_MODEL_PATH,
    DEEPFAKE_MODEL_REPO,
    DEEPFAKE_MODEL_URL,
    DEFAULT_MODEL_PATH,
    MODELS_DIR,
)
from src.utils.constants import CLASS_REAL
from src.utils.errors import ModelError

_model: nn.Module | None = None
_device: torch.device | None = None
_model_source: str = "unknown"


def get_model_source() -> str:
    return _model_source


def download_deepfake_weights(dest: Path | None = None) -> Path:
    """Download FF++ C23 EfficientNet-B0 weights from Hugging Face."""
    try:
        target = dest or DEEPFAKE_MODEL_PATH
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if target.is_file() and target.stat().st_size > 1_000_000:
            return target

        try:
            torch.hub.download_url_to_file(DEEPFAKE_MODEL_URL, str(target), progress=True)
        except (URLError, OSError, RuntimeError) as exc:
            raise ModelError(
                "Could not download deepfake weights",
                f"{DEEPFAKE_MODEL_REPO}: {exc}",
            ) from exc

        if not target.is_file():
            raise ModelError("Download completed but weights file is missing", str(target))
        return target
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError("Weight download failed", str(exc)) from exc


def build_deepfake_efficientnet() -> nn.Module:
    """EfficientNet-B0 with 2-class head (0=real, 1=fake) for FF++ weights."""
    try:
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        return model
    except (RuntimeError, ValueError) as exc:
        raise ModelError("Failed to build EfficientNet-B0 architecture", str(exc)) from exc


def load_detector(
    weights_path: Path | str | None = None,
    device: str | None = None,
) -> tuple[nn.Module, torch.device]:
    """Load FaceForensics++ C23 deepfake detector."""
    global _model_source
    try:
        dev = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = build_deepfake_efficientnet().to(dev)
        model.eval()

        custom = Path(weights_path) if weights_path else DEFAULT_MODEL_PATH
        if custom.is_file() and custom != DEEPFAKE_MODEL_PATH:
            state_path = custom
            _model_source = f"local:{custom.name}"
        else:
            state_path = download_deepfake_weights()
            _model_source = DEEPFAKE_MODEL_REPO

        try:
            state = torch.load(state_path, map_location=dev, weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location=dev)

        model.load_state_dict(state, strict=True)
        return model, dev
    except ModelError:
        raise
    except (OSError, RuntimeError, KeyError) as exc:
        raise ModelError("Failed to load deepfake detector weights", str(exc)) from exc


def ensure_model() -> tuple[nn.Module, torch.device]:
    """Return cached model singleton."""
    global _model, _device
    if _model is None or _device is None:
        _model, _device = load_detector()
    return _model, _device


@torch.inference_mode()
def predict_authenticity_probability(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> float:
    """Return P(real) in [0, 1] — FF++ label 0 = authentic."""
    try:
        batch = tensor.unsqueeze(0).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        return float(probs[0, CLASS_REAL].cpu().item())
    except (RuntimeError, ValueError) as exc:
        raise ModelError("Inference failed", str(exc)) from exc


@torch.inference_mode()
def predict_synthetic_score(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> float:
    """Return P(fake) in [0, 1]."""
    try:
        return 1.0 - predict_authenticity_probability(model, tensor, device)
    except ModelError:
        raise


build_efficientnet_binary = build_deepfake_efficientnet
