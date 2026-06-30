import threading
import time
from datetime import datetime, timezone

import cv2

from config import CAPTURE_INTERVAL_SECONDS


class CaptureBuffer:
    """Lock-protected storage for the most recently captured frame."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.timestamp = None


def _capture_forever(source, buffer):
    capture = None
    backoff_seconds = 1.0

    while True:
        try:
            if capture is None or not capture.isOpened():
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(source)
                if not capture.isOpened():
                    capture.release()
                    capture = None
                    time.sleep(backoff_seconds)
                    backoff_seconds = min(backoff_seconds * 2.0, 30.0)
                    continue

            succeeded, frame = capture.read()
            if not succeeded or frame is None:
                capture.release()
                capture = None
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2.0, 30.0)
                continue

            captured_at = datetime.now(timezone.utc)
            with buffer.lock:
                buffer.frame = frame.copy()
                buffer.timestamp = captured_at

            backoff_seconds = 1.0
            time.sleep(CAPTURE_INTERVAL_SECONDS)
        except Exception as exc:
            print(f"Camera capture error: {exc}")
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
                capture = None
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2.0, 30.0)


def start_capture_thread(source):
    """Start camera capture in a daemon thread and return its shared buffer."""

    buffer = CaptureBuffer()
    thread = threading.Thread(
        target=_capture_forever,
        args=(source, buffer),
        name="camera-capture",
        daemon=True,
    )
    thread.start()
    return buffer


def get_latest_frame(buffer):
    """Return a safe copy of the latest frame and its capture timestamp."""

    with buffer.lock:
        frame = None if buffer.frame is None else buffer.frame.copy()
        return frame, buffer.timestamp
