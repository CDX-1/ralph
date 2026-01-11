import argparse
import json
import socket
import struct
import time

import cv2
import numpy as np
from ultralytics import YOLO

from vision import (
    compute_turn_and_confidence,
    decide_action,
    detect_floor_path_line,
    load_floor_calibration,
    pixel_to_world,
)

def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def handle_client(conn, addr, model, H, far_left, far_right, preview):
    print(f"Client connected: {addr}")
    last_turn = "LEFT"
    last_time = time.time()
    try:
        while True:
            header = recv_exact(conn, 4)
            if not header:
                break
            (frame_len,) = struct.unpack("!I", header)
            if frame_len == 0:
                break
            payload = recv_exact(conn, frame_len)
            if payload is None:
                break

            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w, _ = frame.shape
            fifth = w // 5

            floor_y, floor_origin, floor_target = detect_floor_path_line(frame)
            desired_turn = None
            path_bias = 0.0
            if floor_y is not None:
                dx = floor_target[0] - floor_origin[0]
                path_bias = float(np.clip(dx / max(1.0, w * 0.3), -1.0, 1.0))
                if dx < -w * 0.05:
                    desired_turn = "LEFT"
                elif dx > w * 0.05:
                    desired_turn = "RIGHT"

            results = model(frame, verbose=False)[0]

            ll_objects = []
            l_objects = []
            middle_objects = []
            r_objects = []
            rr_objects = []

            frame_area = float(w * h)

            col_boundaries = [
                (0, fifth, "LL", ll_objects),
                (fifth, 2 * fifth, "L", l_objects),
                (2 * fifth, 3 * fifth, "M", middle_objects),
                (3 * fifth, 4 * fifth, "R", r_objects),
                (4 * fifth, w, "RR", rr_objects),
            ]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2

                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                box_area_normalized = box_area / frame_area

                bx = max(0, min(cx, w - 1))
                by = max(0, min(y2, h - 1))

                X, Y = pixel_to_world(H, bx, by)
                dist_m = (X**2 + Y**2) ** 0.5

                obj_data = (dist_m, box_area_normalized)

                overlaps = []
                for col_start, col_end, region_name, region_list in col_boundaries:
                    if x1 < col_end and x2 > col_start:
                        overlaps.append(region_name)
                        region_list.append(obj_data)

                if preview:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (bx, by), 5, (0, 0, 255), -1)
                    overlap_str = "+".join(overlaps) if overlaps else "--"
                    cv2.putText(
                        frame,
                        f"{overlap_str} {dist_m:.2f}m",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            action, last_turn, risk_meta = decide_action(
                ll_objects,
                l_objects,
                middle_objects,
                r_objects,
                rr_objects,
                last_turn,
                desired_turn,
                verbose=False,
            )
            turn, confidence = compute_turn_and_confidence(action, risk_meta, path_bias)

            now = time.time()
            fps = 1.0 / max(1e-6, now - last_time)
            last_time = now

            response = {
                "action": action,
                "turn": float(turn),
                "confidence": float(confidence),
                "fps": float(fps),
            }
            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

            if preview:
                cv2.line(frame, (fifth, 0), (fifth, h), (255, 255, 255), 2)
                cv2.line(frame, (2 * fifth, 0), (2 * fifth, h), (255, 255, 255), 2)
                cv2.line(frame, (3 * fifth, 0), (3 * fifth, h), (255, 255, 255), 2)
                cv2.line(frame, (4 * fifth, 0), (4 * fifth, h), (255, 255, 255), 2)
                if far_left is not None and far_right is not None:
                    cv2.line(frame, far_left, far_right, (255, 128, 0), 2)
                if floor_y is not None:
                    cv2.line(frame, floor_origin, floor_target, (0, 200, 255), 2)

                LL_dist = min([obj[0] for obj in ll_objects]) if ll_objects else None
                L_dist = min([obj[0] for obj in l_objects]) if l_objects else None
                M_dist = min([obj[0] for obj in middle_objects]) if middle_objects else None
                R_dist = min([obj[0] for obj in r_objects]) if r_objects else None
                RR_dist = min([obj[0] for obj in rr_objects]) if rr_objects else None

                dist_info = f"LL:{LL_dist:.1f}" if LL_dist else "LL:--"
                dist_info += f" L:{L_dist:.1f}" if L_dist else " L:--"
                dist_info += f" M:{M_dist:.1f}" if M_dist else " M:--"
                dist_info += f" R:{R_dist:.1f}" if R_dist else " R:--"
                dist_info += f" RR:{RR_dist:.1f}" if RR_dist else " RR:--"
                cv2.putText(
                    frame,
                    f"{dist_info} FPS:{fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{action} turn={turn:+.2f} conf={confidence:.2f} fps={fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Stream Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if preview:
            cv2.destroyAllWindows()
        conn.close()
        print(f"Client disconnected: {addr}")

def main():
    parser = argparse.ArgumentParser(description="Frame stream server for robot decisions.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--calib", default="floor_calib.json")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    H, far_left, far_right = load_floor_calibration(args.calib)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print(f"Listening on {args.host}:{args.port}")
    while True:
        conn, addr = server.accept()
        handle_client(conn, addr, model, H, far_left, far_right, args.preview)

if __name__ == "__main__":
    main()
