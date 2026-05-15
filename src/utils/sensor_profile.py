"""Sensor profile presets — detection sensitivity and preprocessing."""

from __future__ import annotations

from config import SENSOR_PROFILES

DEFAULT_PROFILE_KEY = "standard_camera"

_active_profile_key: str = DEFAULT_PROFILE_KEY


def list_profile_keys() -> list[str]:
    return list(SENSOR_PROFILES.keys())


def get_profile_display_name(key: str) -> str:
    return SENSOR_PROFILES.get(key, SENSOR_PROFILES[DEFAULT_PROFILE_KEY])["display_name"]


def set_active_sensor_profile(key: str) -> None:
    global _active_profile_key
    if key not in SENSOR_PROFILES:
        raise ValueError(f"Unknown sensor profile: {key}")
    _active_profile_key = key


def get_active_sensor_profile_key() -> str:
    return _active_profile_key


def get_sensor_profile_config(key: str | None = None) -> dict:
    resolved = key or _active_profile_key
    if resolved not in SENSOR_PROFILES:
        resolved = DEFAULT_PROFILE_KEY
    return SENSOR_PROFILES[resolved].copy()


def get_authenticity_warn_pct(key: str | None = None) -> float:
    cfg = get_sensor_profile_config(key)
    return float(cfg["gauge_warn_pct"])


def is_synthetic_label(p_fake_raw: float, key: str | None = None) -> bool:
    """True when raw P(fake) exceeds profile gate (e.g. 0.85 for Standard Camera)."""
    cfg = get_sensor_profile_config(key)
    return p_fake_raw > float(cfg["sigmoid_threshold"])
