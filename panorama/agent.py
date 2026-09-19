"""
Panorama module - agent.py
LangGraph agent điều phối toàn bộ workflow:
  1. Calibrate - VLM (Gemini 3.7 via 9Router) xác nhận góc xoay
  2. Sweep & Capture - Camera quét + chụp frames
  3. Stitch - Ghép panorama
  
Model: ag/gemini-3.7-flash-high via 9Router
"""

import base64
import json
import os
import time
import threading
from typing import Annotated, Any, Optional, TypedDict
import numpy as np
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from core.onvif_client import OnvifClient
from streaming.stream_relay import snapshot
from panorama.sweeper import sweep_and_capture, wait_idle
from panorama.stitcher import stitch, save_results

load_dotenv()

# ── 9Router config (Gemini 3.7 Flash High)
NINEROUTER_BASE_URL = os.getenv("NINEROUTER_BASE_URL", "https://9router.camerangochoang.com/v1")
NINEROUTER_API_KEY  = os.getenv("NINEROUTER_API_KEY", "sk-1aa6a2183c3f40e1-6zg43d-fcff8a05")
VLM_MODEL           = os.getenv("VLM_MODEL", "ag/gemini-3.7-flash-high")

DEFAULT_NUM_STEPS = 8
MIN_STEPS = 4
MAX_STEPS = 14


# ── LangGraph State
class PanoramaState(TypedDict):
    # Config
    client: Any           # OnvifClient
    rtsp_url: str
    tracker: Any          # VirtualPTZTracker (optional)
    mode: str             # "horizon" (1 tầng) | "full_space" (2 tầng)
    # Calibration result
    num_steps: int
    pan_min: float
    pan_max: float
    tilt_pos: float
    tilt_rows: list[float]
    calibration_note: str
    # Sweep result
    frames: list[bytes]
    # Stitch result
    panorama: Optional[bytes]
    grid: Optional[bytes]
    saved_paths: dict
    # Progress callback
    on_progress: Any       # Callable | None
    # Error
    error: Optional[str]


def _encode_image(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


# ── Node 1: Calibrate
def node_calibrate(state: PanoramaState) -> dict:
    client: OnvifClient = state["client"]
    rtsp_url = state["rtsp_url"]
    tracker = state.get("tracker")
    mode = state.get("mode", "full_space")

    from core.auto_calibration import load_camera_profile
    camera_key = getattr(client, "camera_key", "")
    profile = load_camera_profile(camera_key) if camera_key else None

    if tracker and not tracker.is_homed:
        print("[calibrate] Performing homing routine on Virtual PTZ Tracker...")
        try:
            tracker.home(speed=0.8)
        except Exception as e:
            print(f"[calibrate] Homing error: {e}")

    opt_steps = 8
    opt_tilt = -0.80
    tilt_rows = [-0.80] if mode == "horizon" else [-0.80, -0.10]

    if profile:
        opt_steps = profile.get("pan", {}).get("optimal_pan_steps", 8)
        opt_tilt = profile.get("tilt", {}).get("optimal_horizon_tilt_val", -0.80)
        total_pan = profile.get("pan", {}).get("total_pan_range_deg", 360.0)
        note = f"Profile '{camera_key}': Mode={mode}, {len(tilt_rows)} rows x {opt_steps} cols = {len(tilt_rows)*opt_steps} frames."
        print(f"[calibrate] {note}")
        return {
            "num_steps": opt_steps,
            "pan_min": -0.90,
            "pan_max": 0.90,
            "tilt_pos": opt_tilt,
            "tilt_rows": tilt_rows,
            "calibration_note": note,
            "error": None
        }

    return {
        "num_steps": opt_steps,
        "pan_min": -0.90,
        "pan_max": 0.90,
        "tilt_pos": opt_tilt,
        "tilt_rows": tilt_rows,
        "calibration_note": f"Mode={mode} default",
        "error": None
    }


# ── Node 2: Sweep & Capture
def node_sweep(state: PanoramaState) -> dict:
    if state.get("error"):
        return {"frames": []}

    client: OnvifClient = state["client"]
    rtsp_url = state["rtsp_url"]
    tracker = state.get("tracker")
    num_steps = state["num_steps"]
    pan_min = state["pan_min"]
    pan_max = state["pan_max"]
    tilt_rows = state.get("tilt_rows", [-0.80, -0.10])
    on_progress = state.get("on_progress")

    print(f"[sweep] Quét 2D Grid: {len(tilt_rows)} rows x {num_steps} cols, pan [{pan_min:.2f} -> {pan_max:.2f}]")
    from panorama.sweeper import sweep_2d_grid
    frames = sweep_2d_grid(
        client=client,
        rtsp_url=rtsp_url,
        tracker=tracker,
        tilt_rows=tilt_rows,
        cols=num_steps,
        pan_min=pan_min,
        pan_max=pan_max,
        dwell_sec=0.7,
        snapshot_w=1920,
        snapshot_h=1080,
        on_progress=on_progress,
    )
    print(f"[sweep] Đã chụp tổng cộng {len(frames)} frames.")
    return {"frames": frames}


# ── Node 3: Stitch
def node_stitch(state: PanoramaState) -> dict:
    frames = state.get("frames", [])
    if not frames:
        return {"panorama": None, "grid": None, "saved_paths": {}, "error": "No frames captured"}

    client: OnvifClient = state.get("client")
    from core.auto_calibration import load_camera_profile
    camera_key = getattr(client, "camera_key", "") if client else ""
    profile = load_camera_profile(camera_key) if camera_key else None

    dist_coeffs = None
    hfov_deg = 85.0
    pan_angles = None
    tilt_angles = None

    if profile:
        dist_coeffs = np.array(profile.get("optical", {}).get("dist_coeffs", [-0.14, 0.02, 0.0, 0.0, 0.0]), dtype=np.float64)
        hfov_deg = profile.get("optical", {}).get("hfov_deg", 85.0)

    # Tính toán toạ độ pan_angles & tilt_angles nếu quét theo grid
    tilt_rows = state.get("tilt_rows", [-0.80, -0.10])
    num_steps = state.get("num_steps", 8)
    if len(frames) == len(tilt_rows) * num_steps:
        pan_angles = []
        tilt_angles = []
        pan_min = state.get("pan_min", -0.90)
        pan_max = state.get("pan_max", 0.90)
        # Chuyển toạ độ ONVIF pan [-1, 1] sang góc độ [0, 360]
        # pan_step: linspace(pan_min, pan_max, num_steps)
        pan_vals = np.linspace(pan_min, pan_max, num_steps)
        for r_idx, t_val in enumerate(tilt_rows):
            # ONVIF tilt_val sang tilt_deg: min_deg=-5, max_deg=80
            # tilt_deg = -5.0 + ((t_val - (-1.0)) / 2.0) * (80.0 - (-5.0))
            t_deg = -5.0 + ((t_val + 1.0) / 2.0) * 85.0
            for c_idx, p_val in enumerate(pan_vals):
                p_deg = ((p_val + 1.0) / 2.0) * 360.0
                pan_angles.append(float(p_deg))
                tilt_angles.append(float(t_deg))

    print(f"[stitch] Stitching {len(frames)} frames...")
    panorama, grid = stitch(
        frames_bytes=frames,
        grid_rows=len(tilt_rows),
        pan_angles=pan_angles,
        tilt_angles=tilt_angles,
        hfov_deg=hfov_deg,
        dist_coeffs=dist_coeffs
    )
    paths = save_results(frames, panorama, grid)
    print(f"[stitch] Done. Saved to: {paths.get('panorama')}")
    return {"panorama": panorama, "grid": grid, "saved_paths": paths}


# ── Build LangGraph
def build_graph():
    g = StateGraph(PanoramaState)
    g.add_node("calibrate", node_calibrate)
    g.add_node("sweep", node_sweep)
    g.add_node("stitch", node_stitch)
    g.add_edge(START, "calibrate")
    g.add_edge("calibrate", "sweep")
    g.add_edge("sweep", "stitch")
    g.add_edge("stitch", END)
    return g.compile()


# ── Public API (gọi từ FastAPI route)
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_panorama(
    client: OnvifClient,
    rtsp_url: str,
    tracker: Optional[Any] = None,
    mode: str = "full_space",
    on_progress=None,
) -> dict:
    """
    Chạy toàn bộ pipeline panorama với VirtualPTZTracker & chế độ (horizon | full_space).
    Trả về dict: {panorama, grid, saved_paths, calibration_note, num_steps, error}
    """
    initial_state: PanoramaState = {
        "client": client,
        "rtsp_url": rtsp_url,
        "tracker": tracker,
        "mode": mode,
        "num_steps": DEFAULT_NUM_STEPS,
        "pan_min": -0.9,
        "pan_max": 0.9,
        "tilt_pos": -0.80,
        "tilt_rows": [-0.80] if mode == "horizon" else [-0.80, -0.10],
        "calibration_note": "",
        "frames": [],
        "panorama": None,
        "grid": None,
        "saved_paths": {},
        "on_progress": on_progress,
        "error": None,
    }
    result = get_graph().invoke(initial_state)
    return {
        "panorama": result.get("panorama"),
        "grid": result.get("grid"),
        "saved_paths": result.get("saved_paths", {}),
        "calibration_note": result.get("calibration_note", ""),
        "num_steps": result.get("num_steps", DEFAULT_NUM_STEPS),
        "frame_count": len(result.get("frames", [])),
        "error": result.get("error"),
    }
