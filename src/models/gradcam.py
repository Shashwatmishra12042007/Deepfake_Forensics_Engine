"""Grad-CAM explainability for the FF++ EfficientNet-B0 deepfake detector."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.detector import is_denoise_enabled
from src.models.loader import ensure_model
from src.utils.constants import CLASS_FAKE, CLASS_LABELS, CLASS_REAL
from src.utils.errors import ModelError, ProcessingError
from src.utils.sensor_profile import get_active_sensor_profile_key
from src.utils.vision import preprocess_bgr_for_model

__all__ = ["CLASS_FAKE", "CLASS_REAL", "CLASS_LABELS", "generate_gradcam"]


def _get_gradcam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Last feature block — high-level spatial maps before global pooling."""
    try:
        return model.features[-1]
    except (AttributeError, IndexError) as exc:
        raise ModelError("Could not locate Grad-CAM target layer", str(exc)) from exc


def generate_gradcam(
    bgr: np.ndarray,
    target_class: int | None = None,
    overlay_alpha: float = 0.45,
) -> dict:
    """
    Compute Grad-CAM heatmap and overlay for a BGR image.

    Returns dict with overlay_bgr, heatmap_bgr, target_class, target_label,
    class probabilities, and focus_regions hint.
    """
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None

    try:
        if bgr is None or bgr.size == 0:
            raise ProcessingError("Empty or invalid frame for Grad-CAM")

        model, device = ensure_model()
        model.eval()

        target_layer = _get_gradcam_target_layer(model)

        def _forward_hook(_module, _inputs, output: torch.Tensor) -> None:
            nonlocal activations
            activations = output

        def _backward_hook(_module, _grad_input, grad_output) -> None:
            nonlocal gradients
            gradients = grad_output[0]

        fwd_handle = target_layer.register_forward_hook(_forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(_backward_hook)

        try:
            profile_key = get_active_sensor_profile_key()
            tensor, _meta = preprocess_bgr_for_model(
                bgr,
                sensor_profile=profile_key,
                denoise=is_denoise_enabled(),
            )
            tensor = tensor.unsqueeze(0).to(device)
            tensor.requires_grad_(True)

            model.zero_grad(set_to_none=True)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

            if target_class is None:
                target_class = int(np.argmax(probs))

            score = logits[0, target_class]
            score.backward()

            if activations is None or gradients is None:
                raise ProcessingError("Grad-CAM hooks did not capture tensors")

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            cam_np = cam.squeeze().detach().cpu().numpy()
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

            h, w = bgr.shape[:2]
            heatmap = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_CUBIC)
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            overlay_bgr = cv2.addWeighted(
                bgr, 1.0 - overlay_alpha, heatmap_bgr, overlay_alpha, 0
            )

            focus = _describe_focus_regions(heatmap, w, h)

            return {
                "overlay_bgr": overlay_bgr,
                "heatmap_bgr": heatmap_bgr,
                "original_bgr": bgr,
                "target_class": target_class,
                "target_label": CLASS_LABELS.get(target_class, "unknown"),
                "prob_real": round(float(probs[CLASS_REAL]), 4),
                "prob_fake": round(float(probs[CLASS_FAKE]), 4),
                "focus_regions": focus,
                "activation_heatmap": heatmap,
                "frame_height": h,
                "frame_width": w,
            }
        finally:
            fwd_handle.remove()
            bwd_handle.remove()
            model.zero_grad(set_to_none=True)

    except (ModelError, ProcessingError):
        raise
    except (cv2.error, RuntimeError, ValueError) as exc:
        raise ProcessingError("Grad-CAM generation failed", str(exc)) from exc


def _describe_focus_regions(heatmap: np.ndarray, width: int, height: int) -> list[str]:
    """Rough spatial buckets where activation mass is highest."""
    try:
        regions: list[str] = []
        threshold = float(np.percentile(heatmap, 85))
        mask = heatmap >= threshold
        if not mask.any():
            return ["distributed across frame"]

        ys, xs = np.where(mask)
        cx, cy = float(np.mean(xs)), float(np.mean(ys))

        if cy < height * 0.4:
            regions.append("upper face / forehead")
        elif cy > height * 0.62:
            regions.append("lower face / mouth-chin")
        else:
            regions.append("mid face")

        if cx < width * 0.38:
            regions.append("left side")
        elif cx > width * 0.62:
            regions.append("right side")
        else:
            regions.append("central region")

        return regions
    except (ValueError, TypeError):
        return ["unknown"]
