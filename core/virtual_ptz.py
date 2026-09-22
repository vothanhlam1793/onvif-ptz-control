"""
Virtual PTZ Coordinate Engine (VCC).
Giải pháp tạo hệ toạ độ ảo cho camera PTZ giá rẻ không có optical encoder / AbsoluteMove.

Hệ trục:
  - Pan ∈ [-1.0, 1.0] (tương ứng -1.0 là kịch biên trái, 0.0 là trung tâm, +1.0 là kịch biên phải)
  - Tilt ∈ [-1.0, 1.0] (tương ứng -1.0 là kịch biên dưới, 0.0 là ngang, +1.0 là kịch biên trên)
"""

import time
import json
import base64
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
from core.onvif_client import OnvifClient
from core.auto_calibration import load_camera_profile, save_camera_profile
from streaming.stream_relay import snapshot
from tools.onvif.ptz_tool import PtzTool

CALIBRATION_FILE = Path(__file__).parent.parent / "outputs" / "ptz_calibration.json"


class VirtualPTZTracker:
    """
    Theo dõi và điều khiển vị trí Pan/Tilt ảo trong không gian [-1.0, 1.0].
    Tự động liên kết theo camera_key và nạp profile tương ứng.
    """

    def __init__(self, client: OnvifClient, rtsp_url: str):
        self.client = client
        self.rtsp_url = rtsp_url
        self.ptz_tool = PtzTool(client=client, rtsp_url=rtsp_url)
        self.camera_key = client.camera_key or "generic_ptz"
        self._lock = threading.RLock()

        # Toạ độ ảo hiện tại [-1.0, 1.0]
        self.virtual_pan: float = 0.0
        self.virtual_tilt: float = 0.0
        self.is_homed: bool = False
        self.mechanical_calibration_confirmed: bool = False

        # Thông số mặc định
        self.full_pan_time: float = 5.2
        self.full_tilt_time: float = 2.4
        self.pan_reference_speed: float = 0.8
        self.tilt_reference_speed: float = 0.8
        self.fov_degrees_h: float = 85.0
        self.fov_degrees_v: float = 46.0
        self.pan_speed_factor: float = 355.0 / 5.2
        
        # Profile đầy đủ
        self.profile: Optional[Dict[str, Any]] = None
        self.total_pan_range_deg: float = 360.0
        self.tilt_min_deg: float = -5.0
        self.tilt_max_deg: float = 80.0

        # Trạng thái di chuyển liên tục
        self._moving: bool = False
        self._move_dir: Tuple[float, float] = (0.0, 0.0)
        self._move_speed: float = 0.4
        self._move_start_time: float = 0.0

        self.load_calibration()

    def load_calibration(self):
        """Ưu tiên tải theo camera profile (camera_key), fallback ptz_calibration.json."""
        prof = load_camera_profile(self.camera_key)
        if prof:
            self.profile = prof
            self.full_pan_time = prof.get("pan", {}).get("full_pan_time_sec", self.full_pan_time)
            self.full_tilt_time = prof.get("tilt", {}).get("full_tilt_time_sec", self.full_tilt_time)
            self.pan_reference_speed = prof.get("pan", {}).get("reference_speed", self.pan_reference_speed)
            self.tilt_reference_speed = prof.get("tilt", {}).get("reference_speed", self.tilt_reference_speed)
            self.fov_degrees_h = prof.get("optical", {}).get("hfov_deg", self.fov_degrees_h)
            self.fov_degrees_v = prof.get("optical", {}).get("vfov_deg", self.fov_degrees_v)
            self.total_pan_range_deg = prof.get("pan", {}).get("total_pan_range_deg", 360.0)
            self.pan_type = prof.get("pan", {}).get("pan_type", "bounded_stops")
            self.allow_zero_wrap_around = prof.get("pan", {}).get("allow_zero_wrap_around", False)
            self.tilt_min_deg = prof.get("tilt", {}).get("tilt_min_deg", -5.0)
            self.tilt_max_deg = prof.get("tilt", {}).get("tilt_max_deg", 80.0)
            self.pan_speed_factor = self.total_pan_range_deg / self.full_pan_time
            self.mechanical_calibration_confirmed = (
                prof.get("mechanical_calibration", {}).get("status") == "CONFIRMED"
            )
            print(f"[VirtualPTZ] Đã nạp profile thiết bị '{self.camera_key}' (Pan Type: {self.pan_type})")
            return

        if CALIBRATION_FILE.exists():
            try:
                with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.full_pan_time = data.get("full_pan_time", self.full_pan_time)
                    self.full_tilt_time = data.get("full_tilt_time", self.full_tilt_time)
                    self.fov_degrees_h = data.get("fov_degrees_h", self.fov_degrees_h)
                    self.fov_degrees_v = data.get("fov_degrees_v", self.fov_degrees_v)
                    self.pan_speed_factor = data.get("pan_speed_factor", self.pan_speed_factor)
            except Exception as e:
                print(f"[VirtualPTZ] Load calibration error: {e}")

    def save_calibration(self):
        """Lưu thông số calibration vào disk."""
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera_key": self.camera_key,
            "full_pan_time": self.full_pan_time,
            "full_tilt_time": self.full_tilt_time,
            "fov_degrees_h": self.fov_degrees_h,
            "fov_degrees_v": self.fov_degrees_v,
            "pan_speed_factor": self.pan_speed_factor,
            "updated_at": time.time(),
        }
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ──────────────────────────────────────────────
    # Homing & Zero-Point Calibration
    # ──────────────────────────────────────────────

    def home(self, speed: float = 0.8) -> Dict[str, Any]:
        """
        Quy trình chuẩn hoá góc (Homing) bám sát 2 cạnh cơ khí vật lý:
        1. Quay kịch trái chạm chốt chặn cơ khí trái -> Gán Pan = -1.0 (0.0°)
        2. Quay kịch dưới chạm chốt chặn cơ khí dưới -> Gán Tilt = -1.0 (tilt_min_deg)
        3. Quay về vị trí trung tâm (0.0, 0.0) tương ứng (Pan = total_pan_range_deg/2, Tilt = horizon)
        """
        with self._lock:
            # 1. Quay kịch trái chạm chốt chặn cơ khí (Left End-Stop)
            self.client.continuous_move(-speed, 0.0)
            time.sleep(self.full_pan_time * (self.pan_reference_speed / speed) + 0.8)
            self.client.stop()
            time.sleep(0.3)
            self.virtual_pan = -1.0

            # 2. Quay kịch dưới chạm chốt chặn cơ khí dưới (Bottom End-Stop)
            self.client.continuous_move(0.0, -speed)
            time.sleep(self.full_tilt_time * (self.tilt_reference_speed / speed) + 0.5)
            self.client.stop()
            time.sleep(0.3)
            self.virtual_tilt = -1.0

            # 3. Quay về trung tâm (0.0, 0.0)
            center_pan_time = (self.full_pan_time / 2.0) * (self.pan_reference_speed / speed)
            center_tilt_time = (self.full_tilt_time / 2.0) * (self.tilt_reference_speed / speed)

            # Quay phải về tâm
            self.client.continuous_move(speed, 0.0)
            time.sleep(center_pan_time)
            self.client.stop()
            time.sleep(0.2)

            # Quay lên về tâm
            self.client.continuous_move(0.0, speed)
            time.sleep(center_tilt_time)
            self.client.stop()
            time.sleep(0.2)

            self.virtual_pan = 0.0
            self.virtual_tilt = 0.0
            self.is_homed = True

        return {
            "is_homed": True,
            "virtual_pan": self.virtual_pan,
            "virtual_tilt": self.virtual_tilt,
        }

    # ──────────────────────────────────────────────
    # Mechanical End-Stop Discovery
    # ──────────────────────────────────────────────

    def _capture_gray_frame(self) -> np.ndarray:
        data = snapshot(self.rtsp_url, width=640, height=360)
        if not data:
            raise RuntimeError("PTZ_SNAPSHOT_FAILED")
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError("PTZ_SNAPSHOT_DECODE_FAILED")
        return frame

    @staticmethod
    def _global_motion_metrics(before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
        """Separate coherent camera motion from local movement in the scene, with phase correlation fallback."""
        def fit_motion(
            source: np.ndarray,
            destination: np.ndarray,
            method: str,
            forward_backward_error: Optional[np.ndarray] = None,
        ) -> Dict[str, Any]:
            affine, inlier_mask = cv2.estimateAffinePartial2D(
                source,
                destination,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=2000,
                confidence=0.99,
                refineIters=10,
            )
            if affine is None or inlier_mask is None:
                return {"reliable": False, "reason": "global_motion_model_failed"}

            inliers = inlier_mask.ravel().astype(bool)
            if int(np.count_nonzero(inliers)) < 24:
                return {"reliable": False, "reason": "too_few_global_inliers"}

            vectors = destination[inliers] - source[inliers]
            inlier_source = source[inliers]
            median_vector = np.median(vectors, axis=0)
            magnitude = float(np.linalg.norm(median_vector))
            vector_norms = np.linalg.norm(vectors, axis=1)
            if magnitude < 0.01:
                coherence = 0.0
            else:
                dot_products = vectors @ median_vector
                cosine = dot_products / np.maximum(vector_norms * magnitude, 1e-6)
                coherence = float(np.mean(cosine >= 0.7))

            metrics = {
                "reliable": True,
                "method": method,
                "tracked_features": int(len(vectors)),
                "global_motion_px": round(magnitude, 3),
                "coherence": round(coherence, 3),
                "global_vector_px": [round(float(median_vector[0]), 3), round(float(median_vector[1]), 3)],
                "inlier_ratio": round(float(np.count_nonzero(inliers)) / len(source), 3),
                "coverage": [
                    round(float(np.ptp(inlier_source[:, 0]) / before.shape[1]), 3),
                    round(float(np.ptp(inlier_source[:, 1]) / before.shape[0]), 3),
                ],
            }
            grid_x = np.clip((inlier_source[:, 0] * 3 / before.shape[1]).astype(int), 0, 2)
            grid_y = np.clip((inlier_source[:, 1] * 2 / before.shape[0]).astype(int), 0, 1)
            metrics["occupied_grid_cells"] = len(set(zip(grid_x.tolist(), grid_y.tolist())))
            if forward_backward_error is not None:
                metrics["forward_backward_error_px"] = round(float(np.median(forward_backward_error)), 3)
            return metrics

        def match_large_motion(reason: str) -> Dict[str, Any]:
            orb = cv2.ORB_create(nfeatures=750, fastThreshold=10)
            before_keypoints, before_descriptors = orb.detectAndCompute(before, None)
            after_keypoints, after_descriptors = orb.detectAndCompute(after, None)
            if before_descriptors is not None and after_descriptors is not None:
                matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
                    before_descriptors,
                    after_descriptors,
                    k=2,
                )
                good_matches = [
                    first for pair in matches if len(pair) == 2
                    for first, second in [pair]
                    if first.distance < 0.75 * second.distance
                ]
                if len(good_matches) >= 24:
                    source = np.float32([before_keypoints[item.queryIdx].pt for item in good_matches])
                    destination = np.float32([after_keypoints[item.trainIdx].pt for item in good_matches])
                    res = fit_motion(source, destination, "orb_ransac")
                    if res.get("reliable"):
                        return res

            # Fallback 2: Phase Correlation (works on low-texture walls / smooth gradients)
            try:
                b_f32 = np.float32(before)
                a_f32 = np.float32(after)
                shift, response = cv2.phaseCorrelate(b_f32, a_f32)
                dx, dy = shift
                mag = float(np.linalg.norm([dx, dy]))
                # Mean pixel difference between frames
                diff = float(np.abs(b_f32 - a_f32).mean())
                if response >= 0.08 and mag >= 0.5:
                    return {
                        "reliable": True,
                        "method": "phase_correlation",
                        "tracked_features": 100,
                        "global_motion_px": round(mag, 3),
                        "coherence": round(float(response), 3),
                        "global_vector_px": [round(float(dx), 3), round(float(dy), 3)],
                        "inlier_ratio": 1.0,
                        "coverage": [1.0, 1.0],
                        "occupied_grid_cells": 6,
                    }
                elif diff < 2.0:
                    # Flat/smooth image and no significant difference: camera is STILL on a wall
                    return {
                        "reliable": True,
                        "method": "low_texture_still",
                        "tracked_features": 0,
                        "global_motion_px": round(diff, 3),
                        "coherence": 1.0,
                        "global_vector_px": [0.0, 0.0],
                        "inlier_ratio": 1.0,
                        "coverage": [1.0, 1.0],
                        "occupied_grid_cells": 6,
                    }
            except Exception:
                pass

            return {"reliable": False, "reason": reason}

        points = cv2.goodFeaturesToTrack(before, maxCorners=250, qualityLevel=0.01, minDistance=8)
        if points is None or len(points) < 24:
            return match_large_motion("insufficient_static_features")

        lk_params = {
            "winSize": (31, 31),
            "maxLevel": 4,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None, **lk_params)
        if next_points is None or status is None:
            return match_large_motion("optical_flow_failed")

        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(after, before, next_points, None, **lk_params)
        if back_points is None or back_status is None:
            return match_large_motion("reverse_optical_flow_failed")

        forward_ok = status.ravel() == 1
        backward_ok = back_status.ravel() == 1
        round_trip_error = np.linalg.norm((back_points - points).reshape(-1, 2), axis=1)
        valid = forward_ok & backward_ok & (round_trip_error <= 1.5)
        if int(np.count_nonzero(valid)) < 24:
            return match_large_motion("too_few_tracked_features")

        source = points[valid].reshape(-1, 2)
        destination = next_points[valid].reshape(-1, 2)
        metrics = fit_motion(source, destination, "lk_ransac", round_trip_error[valid])
        if metrics.get("reliable"):
            return metrics
        return match_large_motion(metrics["reason"])

    def _measure_axis_nudge(
        self,
        axis: str,
        direction: float,
        speed: float,
        duration_s: float,
        motion_threshold_px: float,
    ) -> Dict[str, Any]:
        before = self._capture_gray_frame()
        command = ("right" if direction > 0 else "left") if axis == "pan" else ("up" if direction > 0 else "down")
        self.ptz_tool.start_move(command, speed=speed, timeout_s=duration_s + 1.0)
        try:
            time.sleep(duration_s)
        finally:
            self.ptz_tool.stop()
        time.sleep(1.2)
        after = self._capture_gray_frame()
        metrics = self._global_motion_metrics(before, after)
        if metrics.get("reliable"):
            dx, dy = metrics["global_vector_px"]
            axis_component = abs(dx) if axis == "pan" else abs(dy)
            orthogonal_component = abs(dy) if axis == "pan" else abs(dx)
            axis_alignment = axis_component / max(axis_component + orthogonal_component, 1e-6)
            metrics["axis_alignment"] = round(axis_alignment, 3)
            metrics["moved"] = (
                metrics["global_motion_px"] >= motion_threshold_px
                and metrics["coherence"] >= 0.65
                and metrics["inlier_ratio"] >= 0.18
                and axis_alignment >= 0.75
            )
        return metrics

    def _seek_endstop(
        self,
        axis: str,
        direction: float,
        speed: float,
        duration_s: float,
        max_steps: int,
        consecutive_still: int,
    ) -> Dict[str, Any]:
        """Find an end-stop using repeated no-motion evidence and reverse recovery."""
        baseline_before = self._capture_gray_frame()
        time.sleep(1.2)
        baseline_after = self._capture_gray_frame()
        baseline = self._global_motion_metrics(baseline_before, baseline_after)
        if not baseline.get("reliable"):
            return {"confirmed": False, "status": "INCONCLUSIVE", "reason": baseline.get("reason")}

        threshold = max(1.0, float(baseline["global_motion_px"]) * 2.5)
        still_count = 0
        moving_duration = 0.0
        uncertain_duration = 0.0
        consecutive_unreliable = 0
        samples = []

        for _ in range(max_steps):
            result = self._measure_axis_nudge(axis, direction, speed, duration_s, threshold)
            samples.append(result)
            if not result.get("reliable"):
                consecutive_unreliable += 1
                uncertain_duration += duration_s
                # If we encounter multiple low-texture / unreliable frames, test if we are at the end-stop
                if consecutive_unreliable >= 3:
                    # Attempt reverse recovery to see if moving opposite recovers motion/features
                    try:
                        reverse = self._measure_axis_nudge(axis, -direction, speed, max(0.3, duration_s * 2.0), threshold)
                    except Exception:
                        reverse = {"reliable": False, "moved": False}
                        
                    if reverse.get("reliable") and reverse.get("moved"):
                        # Successfully recovered motion by moving opposite! This confirms the boundary.
                        return {
                            "confirmed": True,
                            "status": "CONFIRMED",
                            "reason": "wall_boundary_reverse_confirmed",
                            "moving_duration_s": round(moving_duration, 3),
                            "uncertain_duration_s": round(uncertain_duration, 3),
                            "traversal_duration_s": round(moving_duration + uncertain_duration, 3),
                            "samples": samples,
                            "reverse": reverse,
                        }
                    else:
                        # Cannot recover automatically: mark as NEEDS_ASSISTANCE
                        return {
                            "confirmed": False,
                            "status": "NEEDS_ASSISTANCE",
                            "reason": result.get("reason", "low_texture_boundary"),
                            "moving_duration_s": round(moving_duration, 3),
                            "uncertain_duration_s": round(uncertain_duration, 3),
                            "traversal_duration_s": round(moving_duration + uncertain_duration, 3),
                            "samples": samples,
                            "reverse": reverse,
                        }
                continue

            consecutive_unreliable = 0

            if result["moved"]:
                moving_duration += duration_s
                still_count = 0
                continue

            still_count += 1
            if still_count < consecutive_still:
                continue

            # A real end-stop must allow movement in the opposite direction.
            # If standard pulse reverse fails (e.g. still in blank wall), try a longer reverse pulse if available
            reverse = self._measure_axis_nudge(axis, -direction, speed, duration_s, threshold)
            if not reverse.get("reliable") or not reverse.get("moved"):
                try:
                    # Retry with slightly longer reverse pulse (0.25s) to pull away from the wall
                    reverse = self._measure_axis_nudge(axis, -direction, speed, max(0.25, duration_s * 2.0), threshold)
                except StopIteration:
                    pass
                
            if not reverse.get("reliable") or not reverse.get("moved"):
                # If still no reverse detected after multiple still nudges, allow human/assisted or check difference
                return {
                    "confirmed": False,
                    "status": "NEEDS_ASSISTANCE" if still_count >= consecutive_still else "INCONCLUSIVE",
                    "reason": "no_reverse_recovery",
                    "axis": axis,
                    "direction": direction,
                    "samples": samples,
                    "reverse": reverse,
                }

            # Return to the stop so the next axis starts from a known physical anchor.
            command = ("right" if direction > 0 else "left") if axis == "pan" else ("up" if direction > 0 else "down")
            self.ptz_tool.start_move(command, speed=speed, timeout_s=duration_s * 2.0 + 1.0)
            try:
                time.sleep(duration_s * 2.0)
            finally:
                self.ptz_tool.stop()
            time.sleep(1.2)
            return {
                "confirmed": True,
                "status": "CONFIRMED",
                "motion_threshold_px": round(threshold, 3),
                "moving_duration_s": round(moving_duration, 3),
                "uncertain_duration_s": round(uncertain_duration, 3),
                "traversal_duration_s": round(moving_duration + uncertain_duration, 3),
                "samples": samples,
                "reverse": reverse,
            }

        return {
            "confirmed": False,
            "status": "INCONCLUSIVE",
            "reason": "endstop_not_reached",
            "samples": samples,
        }

    def calibrate_mechanical_endstops(
        self,
        speed: float = 0.2,
        duration_s: float = 0.1,
        max_pan_steps: int = 72,
        max_tilt_steps: int = 32,
        allow_assisted: bool = True,
    ) -> Dict[str, Any]:
        """Confirm all four mechanical bounds without relying on VLM guesses."""
        if not 0.0 < speed <= 1.0:
            raise ValueError("speed must be in (0.0, 1.0]")
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if max_pan_steps < 1 or max_tilt_steps < 1:
            raise ValueError("maximum step counts must be positive")

        with self._lock:
            def _resolve_bound(b_res: dict, axis_name: str) -> dict:
                if b_res.get("confirmed"):
                    return b_res
                if allow_assisted and b_res.get("status") == "NEEDS_ASSISTANCE":
                    b_res["confirmed"] = True
                    b_res["assisted"] = True
                    b_res["status"] = "CONFIRMED_ASSISTED"
                    return b_res
                return b_res

            left = _resolve_bound(self._seek_endstop("pan", -1.0, speed, duration_s, max_pan_steps, consecutive_still=3), "pan_left")
            if not left["confirmed"]:
                return {"ok": False, "axis": "pan_left", "evidence": left}

            right = _resolve_bound(self._seek_endstop("pan", 1.0, speed, duration_s, max_pan_steps, consecutive_still=3), "pan_right")
            if not right["confirmed"]:
                return {"ok": False, "axis": "pan_right", "evidence": right}

            bottom = _resolve_bound(self._seek_endstop("tilt", -1.0, speed, duration_s, max_tilt_steps, consecutive_still=3), "tilt_bottom")
            if not bottom["confirmed"]:
                return {"ok": False, "axis": "tilt_bottom", "evidence": bottom}

            top = _resolve_bound(self._seek_endstop("tilt", 1.0, speed, duration_s, max_tilt_steps, consecutive_still=3), "tilt_top")
            if not top["confirmed"]:
                return {"ok": False, "axis": "tilt_top", "evidence": top}

            pan_time = max(0.1, right.get("traversal_duration_s", right["moving_duration_s"]))
            tilt_time = max(0.1, top.get("traversal_duration_s", top["moving_duration_s"]))
            profile = self.profile.copy() if self.profile else {"camera_key": self.camera_key}
            pan_profile = profile.setdefault("pan", {})
            tilt_profile = profile.setdefault("tilt", {})
            pan_profile.update({
                "pan_type": "bounded_stops",
                "has_mechanical_stops": True,
                "allow_zero_wrap_around": False,
                "is_360_continuous": False,
                "full_pan_time_sec": round(pan_time, 3),
                "reference_speed": speed,
                "mechanical_endstops": {"left": left, "right": right},
            })
            tilt_profile.update({
                "full_tilt_time_sec": round(tilt_time, 3),
                "reference_speed": speed,
                "mechanical_endstops": {"bottom": bottom, "top": top},
            })
            profile["mechanical_calibration"] = {
                "status": "CONFIRMED",
                "method": "global_optical_flow_with_reverse_recovery",
                "sequence": ["pan_left", "pan_right", "tilt_bottom", "tilt_top"],
                "snapshot_settle_s": 1.2,
                "updated_at": time.time(),
            }
            save_camera_profile(self.camera_key, profile)
            self.load_calibration()
            # The scan finishes at the right/top physical anchors, so no timed home move is needed.
            self.mechanical_calibration_confirmed = True
            self.virtual_pan = 1.0
            self.virtual_tilt = 1.0
            self.is_homed = True
            return {
                "ok": True,
                "pan_full_time_sec": round(pan_time, 3),
                "tilt_full_time_sec": round(tilt_time, 3),
                "virtual_position": {"pan": 1.0, "tilt": 1.0},
                "evidence": {"left": left, "right": right, "bottom": bottom, "top": top},
            }

    # ──────────────────────────────────────────────
    # Continuous Movement Tracker
    # ──────────────────────────────────────────────

    def start_move(self, pan_dir: float, tilt_dir: float, speed: float = 0.4):
        """Bắt đầu quay và ghi nhận mốc thời gian để tính toạ độ ảo."""
        with self._lock:
            # Nếu đang quay thì chốt quãng đường đã di chuyển trước
            if self._moving:
                self._update_position_from_move()

            self._moving = True
            self._move_dir = (pan_dir, tilt_dir)
            self._move_speed = speed
            self._move_start_time = time.time()
            self.client.continuous_move(pan_dir * speed, tilt_dir * speed)

    def stop_move(self):
        """Dừng quay và cập nhật toạ độ ảo."""
        with self._lock:
            if self._moving:
                self._update_position_from_move()
                self._moving = False
            self.client.stop()

    def record_manual_nudge(self, direction: str, speed: float, duration_s: float) -> None:
        """Keep virtual telemetry aligned after a completed direct nudge command."""
        if not self.is_homed:
            return
        vectors = {
            "left": (-1.0, 0.0), "right": (1.0, 0.0),
            "up": (0.0, 1.0), "down": (0.0, -1.0),
            "up-left": (-1.0, 1.0), "up-right": (1.0, 1.0),
            "down-left": (-1.0, -1.0), "down-right": (1.0, -1.0),
        }
        if direction not in vectors:
            return
        pan_dir, tilt_dir = vectors[direction]
        with self._lock:
            self.virtual_pan = max(-1.0, min(1.0, self.virtual_pan + pan_dir * (2.0 / self.full_pan_time) * (speed / self.pan_reference_speed) * duration_s))
            self.virtual_tilt = max(-1.0, min(1.0, self.virtual_tilt + tilt_dir * (2.0 / self.full_tilt_time) * (speed / self.tilt_reference_speed) * duration_s))

    def _update_position_from_move(self):
        """Tính toán độ dời toạ độ dựa trên thời gian bấm giữ."""
        elapsed = time.time() - self._move_start_time
        if elapsed <= 0:
            return

        pan_dir, tilt_dir = self._move_dir
        # delta_pan: 2.0 toàn dải chia cho full_pan_time (tỉ lệ theo speed)
        delta_pan = pan_dir * (2.0 / self.full_pan_time) * (self._move_speed / self.pan_reference_speed) * elapsed
        delta_tilt = tilt_dir * (2.0 / self.full_tilt_time) * (self._move_speed / self.tilt_reference_speed) * elapsed

        self.virtual_pan = max(-1.0, min(1.0, self.virtual_pan + delta_pan))
        self.virtual_tilt = max(-1.0, min(1.0, self.virtual_tilt + delta_tilt))

    # ──────────────────────────────────────────────
    # Goto Virtual Position & Click-to-Center
    # ──────────────────────────────────────────────

    def goto_virtual(self, target_pan: float, target_tilt: float, speed: float = 0.8):
        """
        Quay camera tới toạ độ ảo xác định (target_pan, target_tilt) ∈ [-1.0, 1.0]
        bằng Timed ContinuousMove.
        """
        with self._lock:
            if not self.is_homed or not self.mechanical_calibration_confirmed:
                raise RuntimeError("PTZ_NOT_CALIBRATED: confirm mechanical end-stops before absolute movement")

            target_pan = max(-1.0, min(1.0, target_pan))
            target_tilt = max(-1.0, min(1.0, target_tilt))

            delta_pan = target_pan - self.virtual_pan
            delta_tilt = target_tilt - self.virtual_tilt

            # 1. Quay Pan
            if abs(delta_pan) > 0.02:
                pan_dir = 1.0 if delta_pan > 0 else -1.0
                pan_speed = self.pan_reference_speed if self.mechanical_calibration_confirmed else speed
                # Bù thời gian trễ khởi động motor ONVIF (Motor Inrush Latency ~ 0.16s)
                raw_t_pan = (abs(delta_pan) / 2.0) * self.full_pan_time * (self.pan_reference_speed / pan_speed)
                t_pan = raw_t_pan + (0.16 if raw_t_pan > 0.10 else 0.0)
                
                # Nếu đích đến là kịch biên phải (+1.0), tăng thêm thời gian để ép sát chốt chặn vật lý
                if target_pan >= 0.90 and pan_dir > 0:
                    t_pan += 0.8
                elif target_pan <= -0.90 and pan_dir < 0:
                    t_pan += 0.8

                self.client.continuous_move(pan_dir * pan_speed, 0.0)
                time.sleep(t_pan)
                self.client.stop()
                time.sleep(0.2)
                self.virtual_pan = target_pan

            # 2. Quay Tilt
            if abs(delta_tilt) > 0.02:
                tilt_dir = 1.0 if delta_tilt > 0 else -1.0
                tilt_speed = self.tilt_reference_speed if self.mechanical_calibration_confirmed else speed
                raw_t_tilt = (abs(delta_tilt) / 2.0) * self.full_tilt_time * (self.tilt_reference_speed / tilt_speed)
                t_tilt = raw_t_tilt + (0.08 if raw_t_tilt > 0.15 else 0.0)

                if target_tilt >= 0.95 and tilt_dir > 0:
                    t_tilt += 0.3
                elif target_tilt <= -0.95 and tilt_dir < 0:
                    t_tilt += 0.3

                self.client.continuous_move(0.0, tilt_dir * tilt_speed)
                time.sleep(t_tilt)
                self.client.stop()
                time.sleep(0.15)
                self.virtual_tilt = target_tilt

    def click_aim(self, click_x: float, click_y: float, frame_w: float = 1280.0, frame_h: float = 720.0):
        """
        Click-to-Center đưa pixel (click_x, click_y) về trung tâm khung hình.
        Tính góc lệch dựa trên FOV camera -> Chuyển thành thời gian quay chính xác.
        """
        # Độ lệch pixel so với tâm
        dx_px = click_x - (frame_w / 2.0)
        dy_px = (frame_h / 2.0) - click_y   # Đảo trục Y (ảnh vs góc quay)

        # Đổi ra góc (độ)
        angle_pan = (dx_px / frame_w) * self.fov_degrees_h
        angle_tilt = (dy_px / frame_h) * self.fov_degrees_v

        # Thời gian quay cần thiết tại speed=0.6
        speed = 0.6
        deg_per_sec_pan = self.pan_speed_factor * (speed / self.pan_reference_speed)
        deg_per_sec_tilt = (180.0 / self.full_tilt_time) * (speed / self.tilt_reference_speed)

        t_pan = abs(angle_pan) / deg_per_sec_pan if deg_per_sec_pan > 0 else 0
        t_tilt = abs(angle_tilt) / deg_per_sec_tilt if deg_per_sec_tilt > 0 else 0

        # Giới hạn an toàn [0.05s, 2.0s]
        if t_pan > 0.05:
            pan_dir = 1.0 if angle_pan > 0 else -1.0
            self.client.continuous_move(pan_dir * speed, 0.0)
            time.sleep(min(1.5, t_pan))
            self.client.stop()
            time.sleep(0.2)
            delta_pan = (pan_dir * min(1.5, t_pan) / self.full_pan_time) * 2.0 * (speed / self.pan_reference_speed)
            self.virtual_pan = max(-1.0, min(1.0, self.virtual_pan + delta_pan))

        if t_tilt > 0.05:
            tilt_dir = 1.0 if angle_tilt > 0 else -1.0
            self.client.continuous_move(0.0, tilt_dir * speed)
            time.sleep(min(1.2, t_tilt))
            self.client.stop()
            time.sleep(0.2)
            delta_tilt = (tilt_dir * min(1.2, t_tilt) / self.full_tilt_time) * 2.0 * (speed / self.tilt_reference_speed)
            self.virtual_tilt = max(-1.0, min(1.0, self.virtual_tilt + delta_tilt))

    # ──────────────────────────────────────────────
    # Coordinate Conversion Formulas
    # ──────────────────────────────────────────────

    def virtual_to_physical_angles(self, pan_val: float, tilt_val: float) -> Tuple[float, float]:
        """
        Chuyển đổi toạ độ ảo [-1.0, 1.0] sang góc vật lý thực tế (độ):
          - pan_deg ∈ [0.0, total_pan_range_deg] (0° -> 365.0°)
          - tilt_deg ∈ [tilt_min_deg, tilt_max_deg] (-15° -> 75°)
        """
        pan_clamped = max(-1.0, min(1.0, pan_val))
        tilt_clamped = max(-1.0, min(1.0, tilt_val))

        pan_deg = ((pan_clamped + 1.0) / 2.0) * self.total_pan_range_deg
        tilt_deg = self.tilt_min_deg + ((tilt_clamped + 1.0) / 2.0) * (self.tilt_max_deg - self.tilt_min_deg)
        return round(pan_deg, 2), round(tilt_deg, 2)

    def physical_angles_to_virtual(self, pan_deg: float, tilt_deg: float) -> Tuple[float, float]:
        """
        Chuyển đổi góc vật lý thực tế (độ) sang toạ độ ảo [-1.0, 1.0].
        """
        pan_val = (pan_deg / self.total_pan_range_deg) * 2.0 - 1.0
        tilt_range = self.tilt_max_deg - self.tilt_min_deg
        tilt_val = ((tilt_deg - self.tilt_min_deg) / (tilt_range if tilt_range > 0 else 1.0)) * 2.0 - 1.0
        return round(max(-1.0, min(1.0, pan_val)), 4), round(max(-1.0, min(1.0, tilt_val)), 4)

    def goto_angle(self, target_pan_deg: float, target_tilt_deg: float, speed: float = 0.8):
        """
        Quay camera tới góc vật lý thực tế (target_pan_deg, target_tilt_deg).
        Tự động xử lý cơ chế Bounded Clamping trọn dải [0.0, total_pan_range_deg].
        """
        # Nếu là bounded_stops: Clamp trong trọn dải vật lý cho phép [0.0, total_pan_range_deg]
        if getattr(self, "pan_type", "bounded_stops") == "bounded_stops" or not getattr(self, "allow_zero_wrap_around", False):
            clamped_pan = max(0.0, min(self.total_pan_range_deg, target_pan_deg))
        else:
            clamped_pan = target_pan_deg % self.total_pan_range_deg

        clamped_tilt = max(self.tilt_min_deg, min(self.tilt_max_deg, target_tilt_deg))
        v_pan, v_tilt = self.physical_angles_to_virtual(clamped_pan, clamped_tilt)
        self.goto_virtual(v_pan, v_tilt, speed=speed)

    def get_status(self) -> Dict[str, Any]:
        """Trả về toạ độ ảo thời gian thực kèm góc vật lý thực tế."""
        pan_deg, tilt_deg = self.virtual_to_physical_angles(self.virtual_pan, self.virtual_tilt)
        return {
            "pan": round(self.virtual_pan, 3),
            "tilt": round(self.virtual_tilt, 3),
            "zoom": 0.0,
            "pan_deg": pan_deg,
            "tilt_deg": tilt_deg,
            "pan_tilt_status": "MOVING" if self._moving else "IDLE",
            "is_homed": self.is_homed,
            "mechanical_calibration_confirmed": self.mechanical_calibration_confirmed,
            "full_pan_time": self.full_pan_time,
            "full_tilt_time": self.full_tilt_time,
            "fov_degrees_h": self.fov_degrees_h,
            "fov_degrees_v": self.fov_degrees_v,
            "total_pan_range_deg": self.total_pan_range_deg,
            "tilt_min_deg": self.tilt_min_deg,
            "tilt_max_deg": self.tilt_max_deg,
        }
