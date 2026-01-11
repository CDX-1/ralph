import argparse
import socket
import struct
import sys
import time

import cv2


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
            print(line.strip())

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
        cap.release()


if __name__ == "__main__":
    main()
