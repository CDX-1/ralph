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
    middle_blocked,
    middle_clear,
    load_floor_calibration,
    pixel_to_world,
    detect_floor_path_line,
)

OCCLUSION_AREA_RATIO = 0.5
OCCLUSION_CLEAR_RATIO = 0.35
OCCLUSION_MAX_BOX_RATIO = 0.4

def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def handle_client(conn, addr, model, H, preview, simple_mode):
    print(f"Client connected: {addr}")
    last_turn = "RIGHT"
    last_time = time.time()
    occluded = False
    
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

            # Detect floor path line for desired direction
            _, origin, target = detect_floor_path_line(frame)
            desired_turn = None
            if origin and target:
                dx = target[0] - origin[0]
                if abs(dx) > w * 0.05:  # 5% threshold
                    desired_turn = "LEFT" if dx < 0 else "RIGHT"

            results = model(frame, verbose=False)[0]

            ll_objects = []
            l_objects = []
            middle_objects = []
            r_objects = []
            rr_objects = []

            frame_area = float(w * h)
            heatmap_boxes = []
            total_box_area = 0.0
            max_box_area = 0.0

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
                total_box_area += box_area_normalized
                if box_area_normalized > max_box_area:
                    max_box_area = box_area_normalized

                # Use bottom center of box for distance measurement
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
                    if simple_mode:
                        heatmap_boxes.append((x1, y1, x2, y2, dist_m))

            # Use the full decision logic instead of simplified version
            if simple_mode:
                if middle_blocked(middle_objects):
                    action = "STOP"
                elif middle_clear(middle_objects):
                    action = "FORWARD"
                else:
                    action = "STOP"
                risk_meta = {"left_risk": 0.0, "right_risk": 0.0}
                turn, confidence = compute_turn_and_confidence(action, risk_meta, 0.0)
            else:
                action, last_turn, risk_meta = decide_action(
                    ll_objects,
                    l_objects,
                    middle_objects,
                    r_objects,
                    rr_objects,
                    last_turn,
                    desired_turn,
                    verbose=preview  # Print debug info when preview is on
                )

                turn, confidence = compute_turn_and_confidence(action, risk_meta, 0.0)

            if occluded:
                if total_box_area < OCCLUSION_CLEAR_RATIO and max_box_area < (OCCLUSION_MAX_BOX_RATIO * 0.8):
                    occluded = False
            else:
                if total_box_area >= OCCLUSION_AREA_RATIO or max_box_area >= OCCLUSION_MAX_BOX_RATIO:
                    occluded = True

            if occluded:
                action = "STOP"
                turn = 0.0
                confidence = 1.0

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
                # Draw region boundaries
                cv2.line(frame, (fifth, 0), (fifth, h), (255, 255, 255), 2)
                cv2.line(frame, (2 * fifth, 0), (2 * fifth, h), (255, 255, 255), 2)
                cv2.line(frame, (3 * fifth, 0), (3 * fifth, h), (255, 255, 255), 2)
                cv2.line(frame, (4 * fifth, 0), (4 * fifth, h), (255, 255, 255), 2)

                # Draw path line
                if origin and target:
                    cv2.line(frame, origin, target, (255, 0, 255), 3)
                    cv2.circle(frame, target, 8, (255, 0, 255), -1)

                # Show distances per region
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
                    f"{dist_info}",
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
                if simple_mode and heatmap_boxes:
                    heat = np.zeros((h, w), dtype=np.float32)
                    max_dist = 3.0
                    for x1, y1, x2, y2, dist_m in heatmap_boxes:
                        intensity = max(0.0, 1.0 - min(dist_m, max_dist) / max_dist)
                        if x2 > x1 and y2 > y1:
                            heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], intensity)
                    heat_u8 = (heat * 255).astype(np.uint8)
                    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(frame, 0.6, heat_color, 0.4, 0)
                    cv2.imshow("Depth Heatmap", overlay)
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
    parser.add_argument("--simple-forward", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    H, _, _ = load_floor_calibration(args.calib)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print(f"Listening on {args.host}:{args.port}")
    while True:
        conn, addr = server.accept()
        handle_client(conn, addr, model, H, args.preview, args.simple_forward)

if __name__ == "__main__":
    main()
