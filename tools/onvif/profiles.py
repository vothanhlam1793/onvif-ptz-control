"""Persistent ONVIF device profiles shared by human and agent runtimes."""

from __future__ import annotations

import json
from pathlib import Path

from tools.onvif.ptz_tool import PtzConnection, PtzToolError


def load_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PtzToolError(f"PTZ_PROFILE_CONFIG_MISSING: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PtzToolError(f"PTZ_PROFILE_CONFIG_INVALID: {path}") from exc
    if not isinstance(config.get("devices"), list):
        raise PtzToolError("PTZ_PROFILE_CONFIG_INVALID")
    return config


def selected_profile(config: dict, name: str = "") -> dict:
    selected = name or config.get("selected", "")
    matches = [device for device in config["devices"] if device.get("name", "").casefold() == selected.casefold()]
    if len(matches) != 1:
        raise PtzToolError("PTZ_PROFILE_NOT_SELECTED")
    device = matches[0]
    required = ("name", "host", "port", "username", "password")
    if not all(device.get(field) for field in required):
        raise PtzToolError("PTZ_PROFILE_CREDENTIAL_REQUIRED")
    return device


def selected_connection(path: Path, name: str = "") -> tuple[dict, PtzConnection]:
    device = selected_profile(load_config(path), name)
    try:
        port = int(device["port"])
    except (TypeError, ValueError) as exc:
        raise PtzToolError("PTZ_PROFILE_PORT_INVALID") from exc
    if not 1 <= port <= 65535:
        raise PtzToolError("PTZ_PROFILE_PORT_INVALID")
    return device, PtzConnection(device["host"], port, device["username"], device["password"])
