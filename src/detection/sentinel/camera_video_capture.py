from asyncio import timeout
from requests.packages import target
import numpy
import threading
from typing import Any
import cv2


class CameraSignalLostError(RuntimeError):
    def __init__(self):
        super().__init__("Sinal da câmera perdido.")


class CameraVideoCapture():
    video_source: Any
    capture: cv2.VideoCapture
    latest_frame: numpy.ndarray | None
    lock: threading.Lock
    is_stopped: threading.Event
    is_ready: threading.Event
    is_connected: bool
    thread: threading.Thread

    def __init__(self, video_source: Any, connection_timeout_seconds: float = 5):
        self.video_source = video_source
        self.capture = cv2.VideoCapture(self.video_source)

        self.lock = threading.Lock()
        self.is_stopped = threading.Event()
        self.is_ready = threading.Event()
        self.is_connected = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

        is_ready = self.is_ready.wait(
            timeout=connection_timeout_seconds
        )

        if not is_ready:
            raise CameraSignalLostError()

    def _reader(self):
        if not self.capture.isOpened():
            raise CameraSignalLostError()

        self.is_connected = True

        while not self.is_stopped.is_set():
            if not self.is_connected:
                raise CameraSignalLostError()

            success, frame = self.capture.read()

            if not success:
                self.is_connected = False

                raise CameraSignalLostError()
            
            with self.lock:
                self.latest_frame = frame

                if not self.is_ready.is_set():
                    self.is_ready.set()

    def read_latest(self):
        return self.latest_frame

    def is_available(self):
        return self.latest_frame is not None

    def release(self):
        self.is_stopped.set()
        self.thread.join(timeout=1.0)
        self.capture.release()