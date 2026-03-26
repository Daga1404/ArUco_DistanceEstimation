import cv2
import numpy as np
import threading
import time

# =========================================================
# CONFIGURACIÓN
# =========================================================
RTSP_URL = "rtsp://10.43.41.58:8554/stream"

# Debe coincidir con el diccionario usado para generar el marcador
DESIRED_ARUCO_DICTIONARY = "DICT_ARUCO_ORIGINAL"

# ID objetivo
TARGET_ID = 2

# Tamaño real del marcador (lado completo) en metros
# Ejemplo: 5 cm = 0.05 m
MARKER_SIZE_M = 0.08

# Distancia real conocida para comparar error
# Ejemplo: marcador colocado a 100 cm de la cámara
KNOWN_DISTANCE_M = 1

# iPhone 15 Pro:
# 24.0 -> 1x
# 28.0 -> 1.2x
# 35.0 -> 1.5x
IPHONE_EQUIV_FOCAL_MM = 24.0

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


# =========================================================
# STREAM RTSP
# =========================================================
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
                    except Exception:
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


# =========================================================
# FUNCIONES
# =========================================================
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def marker_width_in_pixels(corners_4x2):
    top_left, top_right, bottom_right, bottom_left = corners_4x2
    width_top = euclidean_distance(top_left, top_right)
    width_bottom = euclidean_distance(bottom_left, bottom_right)
    return (width_top + width_bottom) / 2.0


def focal_px_from_equiv_mm(frame_width_px, equiv_focal_mm):
    # Aproximación usando ancho full-frame = 36 mm
    return (equiv_focal_mm / 36.0) * frame_width_px


def estimate_distance_pinhole(marker_real_size_m, focal_length_px, marker_width_px):
    if marker_width_px <= 0:
        return None
    return (focal_length_px * marker_real_size_m) / marker_width_px


def compute_errors(z_est_m, z_real_m):
    abs_error_m = abs(z_est_m - z_real_m)
    pct_error = (abs_error_m / z_real_m) * 100 if z_real_m > 0 else 0.0
    return abs_error_m, pct_error


# =========================================================
# MAIN
# =========================================================
def main():
    if DESIRED_ARUCO_DICTIONARY not in ARUCO_DICT:
        print("[ERROR] Diccionario no válido.")
        return

    aruco_dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICT[DESIRED_ARUCO_DICTIONARY]
    )
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dictionary, detector_params)

    print(f"[INFO] Conectando a {RTSP_URL}")
    print(f"[INFO] Detectando marcador ID {TARGET_ID}")
    print(f"[INFO] Focal equivalente seleccionada: {IPHONE_EQUIV_FOCAL_MM} mm")
    print("[INFO] Presiona q para salir")

    cv2.namedWindow("Aruco + Distance + Error", cv2.WINDOW_NORMAL)

    try:
        with RTSPStream(RTSP_URL) as stream:
            while True:
                frame = stream.read()
                if frame is None:
                    continue

                frame_h, frame_w = frame.shape[:2]
                focal_px = focal_px_from_equiv_mm(frame_w, IPHONE_EQUIV_FOCAL_MM)

                corners, ids, rejected = detector.detectMarkers(frame)

                detected = False

                if ids is not None and len(corners) > 0:
                    ids = ids.flatten()

                    for marker_corner, marker_id in zip(corners, ids):
                        if int(marker_id) != TARGET_ID:
                            continue

                        detected = True

                        pts = marker_corner.reshape((4, 2))
                        pts_int = pts.astype(int)

                        top_left = tuple(pts_int[0])
                        top_right = tuple(pts_int[1])
                        bottom_right = tuple(pts_int[2])
                        bottom_left = tuple(pts_int[3])

                        width_px = marker_width_in_pixels(pts)
                        z_est_m = estimate_distance_pinhole(
                            MARKER_SIZE_M,
                            focal_px,
                            width_px
                        )

                        abs_error_m, pct_error = compute_errors(z_est_m, KNOWN_DISTANCE_M)

                        center_x = int(np.mean(pts[:, 0]))
                        center_y = int(np.mean(pts[:, 1]))

                        cv2.line(frame, top_left, top_right, (0, 255, 0), 3)
                        cv2.line(frame, top_right, bottom_right, (0, 255, 0), 3)
                        cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 3)
                        cv2.line(frame, bottom_left, top_left, (0, 255, 0), 3)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                        cv2.putText(
                            frame,
                            f"ID: {marker_id}",
                            (top_left[0], top_left[1] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 255, 0),
                            3
                        )

                        cv2.putText(
                            frame,
                            f"w = {width_px:.1f} px",
                            (top_left[0], top_left[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 0),
                            3
                        )

                        cv2.putText(
                            frame,
                            f"Z_est = {z_est_m * 100:.2f} cm",
                            (top_left[0], bottom_left[1] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 255, 255),
                            3
                        )

                        cv2.putText(
                            frame,
                            f"Error abs = {abs_error_m * 100:.2f} cm",
                            (top_left[0], bottom_left[1] + 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 165, 255),
                            3
                        )

                        cv2.putText(
                            frame,
                            f"Error % = {pct_error:.2f} %",
                            (top_left[0], bottom_left[1] + 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 165, 255),
                            3
                        )

                mode_text = {
                    24.0: "iPhone 1x (24mm)",
                    28.0: "iPhone 1.2x (28mm)",
                    35.0: "iPhone 1.5x (35mm)"
                }.get(IPHONE_EQUIV_FOCAL_MM, f"{IPHONE_EQUIV_FOCAL_MM}mm")

                overlay_lines = [
                    mode_text,
                    f"f_aprox = {focal_px:.1f} px",
                    f"W = {MARKER_SIZE_M * 100:.1f} cm",
                    f"Z_real = {KNOWN_DISTANCE_M * 100:.1f} cm",
                    "Marker 2 detected" if detected else "Marker 2 not detected",
                ]
                overlay_colors = [
                    (255, 255, 255),
                    (255, 255, 255),
                    (255, 255, 255),
                    (255, 255, 255),
                    (0, 255, 0) if detected else (0, 0, 255),
                ]
                line_h = 45
                panel_h = len(overlay_lines) * line_h + 15
                cv2.rectangle(frame, (10, 10), (500, 10 + panel_h), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (500, 10 + panel_h), (80, 80, 80), 1)
                for i, (text, color) in enumerate(zip(overlay_lines, overlay_colors)):
                    cv2.putText(
                        frame,
                        text,
                        (20, 10 + line_h * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        color,
                        2
                    )

                cv2.imshow("Aruco + Distance + Error", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except ConnectionError as e:
        print(f"[ERROR] {e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()