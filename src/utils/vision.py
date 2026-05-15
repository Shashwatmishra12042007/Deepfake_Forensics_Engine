"""BGR frame preparation for CNN input (normalization + forensic heuristics)."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from config import GAUSSIAN_NORMALIZE_SIGMA_DEFAULT, IMAGE_SIZE
from src.processors.forensic_preprocess import (
    apply_denoise_colored,
    apply_gaussian_normalize,
    run_forensic_preprocessing,
)
from src.utils.errors import ProcessingError
from src.utils.sensor_profile import get_sensor_profile_config
from src.utils.transforms import get_image_transform


def preprocess_bgr_for_model(
    bgr: np.ndarray,
    *,
    sensor_profile: str | None = None,
    sharpen: bool | None = None,
    denoise: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Convert BGR frame to model input tensor.

    Pipeline:
      1. Optional fastNlMeans denoising (smartphone sensor noise)
      2. Forensic heuristics (optional sharpen)
      3. Slight Gaussian blur for CNN (normalization layer)
      4. Resize + tensor
    """
    try:
        if bgr is None or bgr.size == 0:
            raise ProcessingError("Empty or invalid frame array")

        profile = get_sensor_profile_config(sensor_profile)
        use_sharpen = profile["use_sharpen_heuristics"] if sharpen is None else sharpen
        sigma = float(profile.get("gaussian_sigma", GAUSSIAN_NORMALIZE_SIGMA_DEFAULT))

        work_bgr = apply_denoise_colored(bgr) if denoise else bgr
        forensic = run_forensic_preprocessing(work_bgr, sharpen=use_sharpen)
        model_bgr = apply_gaussian_normalize(work_bgr, sigma=sigma)

        meta = {
            "moire_score": forensic["moire_score"],
            "spectral_residual_score": forensic["spectral_residual_score"],
            "heuristic_synthetic_score": forensic["heuristic_synthetic_score"],
            "gaussian_sigma": sigma,
            "sharpen_heuristics": use_sharpen,
            "denoise_applied": denoise,
            "sensor_profile": sensor_profile or profile.get("display_name"),
        }

        rgb = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        tensor = get_image_transform()(pil)
        return tensor, meta
    except ProcessingError:
        raise
    except (cv2.error, ValueError, TypeError) as exc:
        raise ProcessingError("BGR preprocessing failed", str(exc)) from exc
