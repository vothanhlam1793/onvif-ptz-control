"""Deterministic PTZ operations shared by the CLI, Web API, and AI agent."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from core.onvif_client import OnvifClient
from streaming.stream_relay import snapshot


class PtzToolError(RuntimeError):
    """A stable error raised by the shared PTZ tool."""


@dataclass(frozen=True)
class PtzConnection:
    host: str
    port: int
    username: str
    password: str


class PtzTool:
    """Single owner of vendor-compatible ONVIF PTZ commands.

    This device requires a Timeout on ContinuousMove even when the caller sends
    Stop explicitly. The transport client applies that compatibility rule.
    """

    _DIRECTIONS = {
        "left": (-1.0, 0.0, 0.0),
        "right": (1.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "down": (0.0, -1.0, 0.0),
        "up-left": (-1.0, 1.0, 0.0),
        "up-right": (1.0, 1.0, 0.0),
        "down-left": (-1.0, -1.0, 0.0),
        "down-right": (1.0, -1.0, 0.0),
        "zoom-in": (0.0, 0.0, 1.0),
        "zoom-out": (0.0, 0.0, -1.0),
    }

    def __init__(self, connection: Optional[PtzConnection] = None,
                 client: Optional[OnvifClient] = None, rtsp_url: str = ""):
        if client is None:
            if connection is None:
                raise ValueError("connection or client is required")
            client = OnvifClient(connection.host, connection.port, connection.username, connection.password)
        self.client = client
        self.rtsp_url = rtsp_url
        self.info: dict = {}

    @classmethod
    def from_environment(cls) -> "PtzTool":
        return cls(PtzConnection(
            host=os.environ["CAMERA_HOST"],
            port=int(os.getenv("CAMERA_PORT", "80")),
            username=os.environ["CAMERA_USER"],
            password=os.environ["CAMERA_PASS"],
        ))

    def connect(self) -> dict:
        try:
            self.info = self.client.discover()
            self.rtsp_url = self.rtsp_url or self.client.get_stream_uri()
            return self.info
        except Exception as exc:
            raise PtzToolError(f"PTZ_CONNECT_FAILED: {exc}") from exc

    def ensure_connected(self) -> None:
        if not self.rtsp_url or not self.client.profile_token:
            self.connect()

    def probe(self) -> dict:
        self.ensure_connected()
        return {
            "device": self.info,
            "profile_token": self.client.profile_token,
            "ptz_url": self.client.ptz_url,
            "rtsp_ready": bool(self.rtsp_url),
            "absolute_move_supported": False,
            "position_status_reliable": False,
        }

    def start_move(self, direction: str, speed: float = 0.3, timeout_s: float = 5.0) -> None:
        self.ensure_connected()
        if direction not in self._DIRECTIONS:
            raise PtzToolError(f"PTZ_INVALID_DIRECTION: {direction}")
        if not 0.05 <= speed <= 1.0:
            raise PtzToolError("PTZ_INVALID_SPEED")
        if not 0.1 <= timeout_s <= 30.0:
            raise PtzToolError("PTZ_INVALID_TIMEOUT")
        pan, tilt, zoom = self._DIRECTIONS[direction]
        try:
            self.client.continuous_move(pan * speed, tilt * speed, zoom * speed, timeout_s=timeout_s)
        except Exception as exc:
            raise PtzToolError(f"PTZ_MOVE_FAILED: {exc}") from exc

    def stop(self) -> None:
        self.ensure_connected()
        try:
            self.client.stop(pan_tilt=True, zoom=True)
        except Exception as exc:
            raise PtzToolError(f"PTZ_STOP_FAILED: {exc}") from exc

    def nudge(self, direction: str, speed: float = 0.3, duration_s: float = 1.0) -> None:
        if not 0.1 <= duration_s <= 2.0:
            raise PtzToolError("PTZ_INVALID_DURATION")
        self.start_move(direction, speed=speed, timeout_s=duration_s + 1.0)
        try:
            time.sleep(duration_s)
        finally:
            self.stop()

    def relative_move(self, pan: float, tilt: float, speed: float = 0.3) -> None:
        self.ensure_connected()
        if not 0.05 <= speed <= 1.0:
            raise PtzToolError("PTZ_INVALID_SPEED")
        try:
            self.client.relative_move(pan, tilt, pan_speed=speed, tilt_speed=speed)
        except Exception as exc:
            raise PtzToolError(f"PTZ_RELATIVE_MOVE_FAILED: {exc}") from exc

    def take_snapshot(self, width: int = 1280, height: int = 720) -> bytes:
        self.ensure_connected()
        frame = snapshot(self.rtsp_url, width=width, height=height)
        if not frame:
            raise PtzToolError("PTZ_SNAPSHOT_FAILED")
        return frame

    def probe_motion(self, direction: str, speed: float = 0.3, duration_s: float = 1.0) -> dict:
        before = self.take_snapshot(640, 360)
        self.nudge(direction, speed=speed, duration_s=duration_s)
        time.sleep(0.5)
        after = self.take_snapshot(640, 360)
        before_img = cv2.imdecode(np.frombuffer(before, np.uint8), cv2.IMREAD_GRAYSCALE)
        after_img = cv2.imdecode(np.frombuffer(after, np.uint8), cv2.IMREAD_GRAYSCALE)
        if before_img is None or after_img is None:
            raise PtzToolError("PTZ_SNAPSHOT_DECODE_FAILED")
        before_img = cv2.resize(before_img, (160, 90)).astype(np.float32)
        after_img = cv2.resize(after_img, (160, 90)).astype(np.float32)
        difference = float(np.abs(before_img - after_img).mean())
        return {"direction": direction, "difference": difference, "motion_detected": difference > 8.0}


class PtzRuntimeAdapter:
    """Login-gated runtime session for manual PTZ control."""

    DISCONNECTED = "DISCONNECTED"
    READY = "READY"
    FAILED = "FAILED"

    def __init__(self):
        self.state = self.DISCONNECTED
        self.tool: Optional[PtzTool] = None
        self.last_error = ""

    def login(self, connection: PtzConnection) -> dict:
        self.logout()
        candidate = PtzTool(connection=connection)
        try:
            info = candidate.connect()
        except PtzToolError as exc:
            self.state = self.FAILED
            self.last_error = str(exc)
            raise
        self.tool = candidate
        self.state = self.READY
        self.last_error = ""
        return info

    def logout(self) -> None:
        if self.tool:
            try:
                self.tool.stop()
            except PtzToolError:
                pass
            self.tool.client._session.close()
        self.tool = None
        self.state = self.DISCONNECTED
        self.last_error = ""

    def status(self) -> dict:
        return {
            "state": self.state,
            "camera_key": self.tool.info.get("camera_key") if self.tool else None,
            "last_error": self.last_error or None,
        }

    def _require_ready(self) -> PtzTool:
        if self.state != self.READY or not self.tool:
            raise PtzToolError("PTZ_NOT_LOGGED_IN")
        return self.tool

    def move(self, direction: str, speed: float, duration_s: float) -> None:
        self._require_ready().nudge(direction, speed=speed, duration_s=duration_s)

    def stop(self) -> None:
        self._require_ready().stop()

    def snapshot(self) -> bytes:
        return self._require_ready().take_snapshot()

    def probe_motion(self, direction: str, speed: float, duration_s: float) -> dict:
        return self._require_ready().probe_motion(direction, speed=speed, duration_s=duration_s)
