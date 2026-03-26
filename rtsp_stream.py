"""
RTSP phone camera stream connector.

Phone setup:
  Android - Install 'IP Webcam' (by Pavel Khlebovich), start server.
            Default RTSP URL: rtsp://<phone-ip>:8080/h264_ulaw.sdp
  iOS     - Install 'IP Camera Lite', start stream.
            Default RTSP URL: rtsp://<phone-ip>:554/live

Usage:
    stream = RTSPStream("rtsp://192.168.1.100:8080/h264_ulaw.sdp")
    stream.start()
    while True:
        frame = stream.read()
        if frame is None:
            break
        cv2.imshow("Phone Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    stream.stop()
"""

import threading
import cv2


class RTSPStream:
    def __init__(self, url: str, reconnect: bool = True):
        self.url = url
        self.reconnect = reconnect
        self._cap: cv2.VideoCapture | None = None
        self._frame = None
        self._running = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self):
        self._cap = self._open_capture()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ConnectionError(f"Could not connect to RTSP stream: {self.url}")
        return cap

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if self.reconnect and self._running:
                    print("Stream lost, reconnecting...")
                    self._cap.release()
                    try:
                        self._cap = self._open_capture()
                    except ConnectionError:
                        pass
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


def main():
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "rtsp://10.43.41.58:8554/stream"
    print(f"Connecting to: {url}")
    print("Press 'q' to quit.")

    with RTSPStream(url) as stream:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            max_width = 1280
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (max_width, int(h * scale)))

            cv2.imshow("Phone Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()