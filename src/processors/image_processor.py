"""Image loading and preprocessing for forensic CNN inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from config import IMAGE_SIZE
from src.utils.errors import MediaLoadError, ProcessingError
from src.utils.transforms import get_image_transform


def load_image(path: str | Path) -> np.ndarray:
    """Load image as BGR uint8 array."""
    try:
        path = Path(path)
        if not path.is_file():
            raise MediaLoadError("Image file not found", str(path))
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise MediaLoadError("OpenCV could not decode image", str(path))
        return img
    except MediaLoadError:
        raise
    except (OSError, ValueError) as exc:
        raise MediaLoadError("Failed to load image", str(exc)) from exc


def preprocess_image(bgr: np.ndarray) -> torch.Tensor:
    """BGR ndarray -> normalized CHW tensor."""
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        return get_image_transform()(pil)
    except (cv2.error, ValueError, TypeError) as exc:
        raise ProcessingError("Image preprocessing failed", str(exc)) from exc
