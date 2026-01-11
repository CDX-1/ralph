import json
import os
import socket
import struct
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import (
    CONF_THRESH,
    LEFT_MAX,
    RIGHT_MIN,
    RELEVANT_CLASSES,
)
from spatial import decide_action, draw_overlays
from midas_depth import MidasDepth


HOST = "0.0.0.0"
PORT = 5001
JPEG_QUALITY = 80
SHOW_WINDOW = os.environ.get("SHOW_STREAM", "").lower() in ("1", "true", "yes")
SWERVE_DISTANCE_M = float(os.environ.get("SWERVE_DISTANCE_M", "1.5"))
SWERVE_RISK_CENTER_MIN = float(os.environ.get("SWERVE_RISK_CENTER_MIN", "0.06"))
SWERVE_RISK_DIFF_MIN = float(os.environ.get("SWERVE_RISK_DIFF_MIN", "0.02"))
SHOW_MIDAS = os.environ.get("SHOW_MIDAS", "").lower() not in ("0", "false", "no")
STOP_DISTANCE_M = float(os.environ.get("STOP_DISTANCE_M", "0.05"))
MIDAS_CLOSE_THRESHOLD = float(os.environ.get("MIDAS_CLOSE_THRESHOLD", "0.65"))


def init_midas():
    if not SHOW_MIDAS:
        print("MiDaS display disabled (SHOW_MIDAS=0)")
        return None
    try:
        midas = MidasDepth(
            device="cpu",
            input_size=(384, 216),
            close_threshold=MIDAS_CLOSE_THRESHOLD,
            run_every_n_frames=4,
        )
        print("MiDaS display enabled")
        return midas
    except Exception as exc:
        print(f"MiDaS init failed: {exc}")
        return None


def recvall(sock: socket.socket, length: int) -> bytes:
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


def handle_client(conn: socket.socket, addr, model: YOLO, midas):
    print(f"Client connected: {addr}")
    prev_area_by_key = {}

    try:
        while True:
            header = recvall(conn, 4)
            if not header:
                break
            (length,) = struct.unpack("!I", header)
            if length <= 0:
                continue

            payload = recvall(conn, length)
            if not payload:
                break

            frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            yolo_t0 = time.time()
            results = model(
                frame,
                verbose=False,
                classes=list(RELEVANT_CLASSES.keys()),
            )[0]
            yolo_fps = 1.0 / max(time.time() - yolo_t0, 1e-6)

            overlays = []
            action = "GO"
            reason = "No detections yet"
            risks = None

            if results is not None and results.boxes is not None:
                dets = []
                for b in results.boxes:
                    conf = float(b.conf[0])
                    if conf < CONF_THRESH:
                        continue
                    cls = int(b.cls[0])
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    dets.append({
                        "cls": cls,
                        "conf": conf,
                        "xyxy": (x1, y1, x2, y2),
                    })

                action, reason, debug, prev_area_by_key, overlays = decide_action(
                    frame.shape[1], frame.shape[0], dets, prev_area_by_key
                )
                risks = debug.get("risk")

            distance_ahead = None
            distance_critical = False
            if overlays:
                center_objects = [o for o in overlays if o["region"] == "CENTER"]
                if center_objects:
                    main_object = max(center_objects, key=lambda x: x["area_norm"])
                    area_norm = main_object["area_norm"]
                    distance_ahead = max(0.3, 10.0 * (0.05 / max(area_norm, 0.001))) / 2
                    if distance_ahead <= 1.0:
                        distance_critical = True

            if distance_ahead is not None and distance_ahead <= STOP_DISTANCE_M:
                action = "STOP"
                reason = f"Obstacle within {STOP_DISTANCE_M:.2f}m"

            response = {
                "action": action,
                "reason": reason,
                "yolo_fps": yolo_fps,
                "distance_ahead": distance_ahead,
                "distance_critical": distance_critical,
                "timestamp": time.time(),
            }
            payload = json.dumps(response).encode("utf-8")
            conn.sendall(struct.pack("!I", len(payload)) + payload)

            if SHOW_WINDOW:
                frame = draw_overlays(frame, overlays, action, reason, yolo_fps)
                cv2.imshow("RPI Stream (YOLO on Mac)", frame)

                if midas:
                    depth = midas.update(frame)
                    if depth["valid"] and depth.get("close_ahead"):
                        action = "STOP"
                        reason = "MiDaS: close obstacle ahead"
                    if depth["valid"] and depth["depth_vis"] is not None:
                        cv2.imshow("MiDaS Depth Map (Mac)", depth["depth_vis"])

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except ConnectionError:
        pass
    finally:
        conn.close()
        print(f"Client disconnected: {addr}")
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


def main():
    model = YOLO("models/yolov8n.pt")
    print(f"Mac server listening on {HOST}:{PORT}")
    midas = init_midas()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        while True:
            conn, addr = server.accept()
            handle_client(conn, addr, model, midas)


if __name__ == "__main__":
    main()
