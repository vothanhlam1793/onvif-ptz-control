"""
RTSP -> MJPEG bridge và Continuous RTSP Stream Worker.
- StreamRelay: cung cấp HTTP multipart MJPEG stream cho Web UI.
- RTSPStreamWorker: duy trì kết nối RTSP liên tục bằng background thread,
  gắn timestamp cho từng frame để snapshot tức thì mà không bao giờ đọc phải stale frame.
- snapshot(): hàm chụp frame đơn fallback khi worker chưa chạy.
"""

import subprocess
import threading
import time
from typing import Generator, Optional, Tuple
import cv2
import numpy as np


class RTSPStreamWorker:
    """
    Worker duy trì kết nối RTSP liên tục trong background thread.
    - Không phải reconnect mỗi lần chụp ảnh (tiết kiệm 0.5s - 0.7s handshake).
    - Lưu frame mới nhất kèm timestamp để đảm bảo tính thời gian thực.
    - Hỗ trợ get_settled_frame(min_timestamp) chống hiện tượng stale frame.
    """

    def __init__(self, rtsp_url: str, width: int = 1280, height: int = 720):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._latest_timestamp: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._reconnect_count = 0

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False

    def _worker_loop(self):
        while self._running:
            cap = cv2.VideoCapture(self.rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                time.sleep(1.0)
                continue

            while self._running:
                # grab() liên tục để dọn sạch buffer của RTSP socket
                grabbed = cap.grab()
                if not grabbed:
                    break

                t_now = time.time()
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if ok:
                        with self._lock:
                            self._latest_frame = buf.tobytes()
                            self._latest_timestamp = t_now

                # Điều tiết nhẹ ~25-30 fps để không ngốn CPU vô ích
                time.sleep(0.02)

            cap.release()
            if self._running:
                time.sleep(0.5)

    def get_latest_frame(self) -> Tuple[Optional[bytes], float]:
        """Trả về (frame_bytes, timestamp) của frame mới nhất hiện có."""
        with self._lock:
            return self._latest_frame, self._latest_timestamp

    def get_settled_frame(self, min_timestamp: float, timeout_s: float = 3.0) -> Optional[bytes]:
        """
        Lấy frame được đọc SAU mốc min_timestamp (ví dụ: t_stop + settle_time).
        Nếu frame hiện tại chưa qua mốc này, chờ cho tới khi có frame mới.
        """
        t_start = time.time()
        while time.time() - t_start < timeout_s:
            with self._lock:
                if self._latest_frame and self._latest_timestamp >= min_timestamp:
                    return self._latest_frame
            time.sleep(0.04)

        # Hết timeout mà chưa kịp có frame mới hơn, trả frame mới nhất hiện có
        with self._lock:
            return self._latest_frame


# Singleton quản lý worker theo URL
_WORKERS: dict[str, RTSPStreamWorker] = {}
_WORKERS_LOCK = threading.Lock()


def get_stream_worker(rtsp_url: str, width: int = 1280, height: int = 720) -> RTSPStreamWorker:
    """Lấy hoặc khởi tạo một RTSPStreamWorker duy nhất cho URL cụ thể."""
    with _WORKERS_LOCK:
        if rtsp_url not in _WORKERS:
            worker = RTSPStreamWorker(rtsp_url, width, height)
            worker.start()
            _WORKERS[rtsp_url] = worker
        return _WORKERS[rtsp_url]


class StreamRelay:
    """
    Dùng ffmpeg để decode RTSP stream -> JPEG frames cho Web UI.
    """

    def __init__(self, rtsp_url: str, width: int = 1280, height: int = 720):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _capture_loop(self):
        cmd = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-vf", f"scale={self.width}:{self.height}",
            "-q:v", "5",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "pipe:1",
        ]
        proc = None
        while self._running:
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                buf = b""
                while self._running:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        start = buf.find(b"\xff\xd8")
                        end   = buf.find(b"\xff\xd9", start + 2)
                        if start == -1 or end == -1:
                            break
                        frame = buf[start:end + 2]
                        with self._lock:
                            self._latest_frame = frame
                        buf = buf[end + 2:]
            except Exception:
                pass
            finally:
                if proc:
                    proc.kill()
            if self._running:
                time.sleep(1)

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    def mjpeg_generator(self) -> Generator[bytes, None, None]:
        while True:
            frame = self.get_frame()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame +
                    b"\r\n"
                )
            time.sleep(0.033)


def snapshot(rtsp_url: str, width: int = 1280, height: int = 720) -> Optional[bytes]:
    """
    Chụp frame từ RTSP stream:
    - Ưu tiên bốc ngay từ RTSPStreamWorker đang chạy (0ms latency, không reconnect).
    - Fallback mở kết nối cv2.VideoCapture on-demand nếu worker chưa chạy.
    """
    with _WORKERS_LOCK:
        if rtsp_url in _WORKERS:
            frame, _ = _WORKERS[rtsp_url].get_latest_frame()
            if frame:
                return frame

    # Fallback on-demand
    try:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                return buf.tobytes()
    except Exception:
        pass

    return None
