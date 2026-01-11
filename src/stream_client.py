import argparse
import json
import socket
import struct
import sys
import time

import cv2

from motors import MotorController


def send_frame(sock, frame, jpeg_quality):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        return False
    payload = buf.tobytes()
    header = struct.pack("!I", len(payload))
    sock.sendall(header + payload)
    return True


def main():
    parser = argparse.ArgumentParser(description="Stream camera frames to server and print actions.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Failed to open camera", file=sys.stderr)
        sys.exit(1)

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    sock_file = sock.makefile("r", encoding="utf-8", newline="\n")

    try:
        motors = MotorController()
    except Exception as exc:
        print(f"Motor controller unavailable: {exc}", file=sys.stderr)
        motors = None

    next_frame_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if not send_frame(sock, frame, args.quality):
                break

            line = sock_file.readline()
            if not line:
                break
            raw = line.strip()
            print(raw)
            action_data = None
            try:
                action_data = json.loads(raw)
            except json.JSONDecodeError:
                action_data = None

            if action_data and motors is not None:
                action = action_data.get("action")
                turn = float(action_data.get("turn", 0.0))
                confidence = float(action_data.get("confidence", 0.0))
                speed = max(0.0, min(1.0, abs(turn)))
                print(f"action={action} turn={turn:+.2f} conf={confidence:.2f} speed={speed:.2f}")

                if action == "STOP":
                    motors.stop()
                elif action == "FORWARD":
                    motors.forward(speed=max(0.3, speed))
                elif action == "STEER_LEFT":
                    motors.turn_left(speed=max(0.3, speed))
                elif action == "STEER_RIGHT":
                    motors.turn_right(speed=max(0.3, speed))

            if args.fps > 0:
                next_frame_time += 1.0 / args.fps
                sleep_for = next_frame_time - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_frame_time = time.time()
    finally:
        try:
            sock.sendall(struct.pack("!I", 0))
        except Exception:
            pass
        sock.close()
        if motors is not None:
            motors.cleanup()
        cap.release()


if __name__ == "__main__":
    main()
