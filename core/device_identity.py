"""Stable physical camera identity used by per-device calibration profiles."""

from __future__ import annotations

import re
from typing import Any


def sanitize_identity_component(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def legacy_camera_key(manufacturer: str, model: str) -> str:
    manufacturer_key = sanitize_identity_component(manufacturer)
    model_key = sanitize_identity_component(model)
    if manufacturer_key and model_key:
        return f"{manufacturer_key}_{model_key}"
    return ""


def physical_camera_key(
    manufacturer: str,
    model: str,
    serial_number: str = "",
    mac_address: str = "",
    host: str = "",
) -> str:
    """Build a key that cannot collide for two cameras of the same model."""
    prefix = legacy_camera_key(manufacturer, model) or "camera"
    serial = sanitize_identity_component(serial_number)
    mac = sanitize_identity_component(mac_address).replace("_", "")
    host_key = sanitize_identity_component(host)
    unique = serial or mac or host_key
    return f"{prefix}_{unique}" if unique else prefix


def camera_identity(client: Any) -> dict[str, str]:
    return {
        "manufacturer": str(getattr(client, "manufacturer", "") or ""),
        "model": str(getattr(client, "model", "") or ""),
        "serial_number": str(getattr(client, "serial_number", "") or ""),
        "mac_address": str(getattr(client, "mac_address", "") or ""),
        "firmware_version": str(getattr(client, "firmware_version", "") or ""),
        "hardware_id": str(getattr(client, "hardware_id", "") or ""),
    }
