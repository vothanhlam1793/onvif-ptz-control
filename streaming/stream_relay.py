"""
RTSP -> MJPEG bridge dùng ffmpeg subprocess.
Cấp frame JPEG liên tục qua generator cho FastAPI StreamingResponse.
Snapshot chụp frame đơn bằng ffmpeg một lần.
"""

import subprocess
import threading
import time
from typing import Generator, Optional


class StreamRelay:
    """
    Dùng ffmpeg để decode RTSP stream -> JPEG frames.
    Mỗi frame được buffer vào bộ nhớ, các client HTTP đọc qua generator.
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
            "-q:v", "5",          # JPEG quality (1=best, 31=worst)
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
                    # Find JPEG boundaries: SOI=FFD8, EOI=FFD9
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
                time.sleep(1)  # Reconnect delay

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    def mjpeg_generator(self) -> Generator[bytes, None, None]:
        """Generator cho HTTP multipart/x-mixed-replace MJPEG stream."""
        while True:
            frame = self.get_frame()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame +
                    b"\r\n"
                )
            time.sleep(0.033)  # ~30fps cap


def snapshot(rtsp_url: str, width: int = 1920, height: int = 1080) -> Optional[bytes]:
    """
    Chụp 1 frame đơn từ RTSP stream.
    Dùng cho AI inspection: chất lượng cao, không cần stream liên tục.
    """
    cmd = [
        "ffmpeg",
        "-loglevel", "quiet",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-vf", f"scale={width}:{height}",
        "-vframes", "1",
        "-q:v", "2",
        "-f", "image2",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=5)
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except Exception:
        pass
    return None
