import cv2
import numpy as np
import threading
import time
import sys

# =========================
# CONFIGURACIÓN
# =========================
RTSP_URL = "rtsp://10.43.41.58:8554/stream"
DESIRED_ARUCO_DICTIONARY = "DICT_ARUCO_ORIGINAL"

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}


# =========================
# STREAM RTSP
# =========================
class RTSPStream:
    def __init__(self, url: str, reconnect: bool = True):
        self.url = url
        self.reconnect = reconnect
        self._cap = None
        self._frame = None
        self._running = False
        self._lock = threading.Lock()
        self._thread = None

    def _open_capture(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ConnectionError(f"No se pudo conectar al stream RTSP: {self.url}")
        return cap

    def start(self):
        self._cap = self._open_capture()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()

            if not ret:
                if self.reconnect and self._running:
                    print("[INFO] Stream perdido, reconectando...")
                    try:
                        self._cap.release()
                    except:
                        pass

                    time.sleep(1)

                    try:
                        self._cap = self._open_capture()
                    except ConnectionError:
                        continue
                continue

            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# =========================
# DETECCIÓN ARUCO
# =========================
def main():
    if ARUCO_DICT.get(DESIRED_ARUCO_DICTIONARY, None) is None:
        print(f"[ERROR] El diccionario '{DESIRED_ARUCO_DICTIONARY}' no es válido.")
        sys.exit(0)

    print(f"[INFO] Conectando a: {RTSP_URL}")
    print(f"[INFO] Detectando marcadores del tipo: {DESIRED_ARUCO_DICTIONARY}")
    print("[INFO] Presiona 'q' para salir.")

    aruco_dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICT[DESIRED_ARUCO_DICTIONARY]
    )
    aruco_parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)

    cv2.namedWindow("Deteccion ArUco", cv2.WINDOW_NORMAL)

    try:
        with RTSPStream(RTSP_URL) as stream:
            while True:
                frame = stream.read()

                if frame is None:
                    continue

                corners, ids, rejected = detector.detectMarkers(frame)

                if ids is not None and len(corners) > 0:
                    ids = ids.flatten()

                    for marker_corner, marker_id in zip(corners, ids):
                        corners_reshaped = marker_corner.reshape((4, 2))
                        (top_left, top_right, bottom_right, bottom_left) = corners_reshaped

                        top_left = (int(top_left[0]), int(top_left[1]))
                        top_right = (int(top_right[0]), int(top_right[1]))
                        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

                        # Dibujar contorno
                        cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                        cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                        cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                        cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                        # Centro
                        center_x = int((top_left[0] + bottom_right[0]) / 2.0)
                        center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                        # ID
                        cv2.putText(
                            frame,
                            f"ID: {marker_id}",
                            (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                cv2.imshow("Deteccion ArUco", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except ConnectionError as e:
        print(f"[ERROR] {e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
