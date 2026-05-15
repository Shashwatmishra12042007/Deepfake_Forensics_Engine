"""Sigmoid calibration for synthetic (fake) probability scores."""

from __future__ import annotations

import math


def sigmoid_calibrate_synthetic(
    p_fake_raw: float,
    *,
    threshold: float = 0.85,
    steepness: float = 14.0,
) -> float:
    """
    Map raw P(synthetic) through a logistic gate centered at ``threshold``.

  Only scores with raw P(fake) well above ``threshold`` produce high calibrated
    synthetic confidence — reduces smartphone / sharpening false positives.
    """
    try:
        p = min(1.0, max(0.0, float(p_fake_raw)))
        z = steepness * (p - threshold)
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))
    except (ValueError, TypeError, OverflowError):
        return 0.0


def calibrated_authenticity(p_fake_raw: float, **kwargs) -> float:
    """Return calibrated P(authentic) in [0, 1]."""
    p_fake_cal = sigmoid_calibrate_synthetic(p_fake_raw, **kwargs)
    return min(1.0, max(0.0, 1.0 - p_fake_cal))
