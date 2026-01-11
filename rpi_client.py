import json
import os
import socket
import struct
import sys
import time
import subprocess

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from camera import get_camera_index
from config import FRAME_W, FRAME_H


SERVER_HOST = os.environ.get("MAC_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("MAC_SERVER_PORT", "5001"))
JPEG_QUALITY = 80
PRINT_EVERY_N_FRAMES = 10


def _which(cmd: str):
    from shutil import which as _which_inner
    return _which_inner(cmd)


class SafeAudioManager:
    def __init__(self):
        self.mode = "none"
        self.enabled = False
        self.audio_files = {}
        self._player = None

        audio_dir = os.path.join("voice", "commands")
        self.audio_files = {
            "STOP": os.path.join(audio_dir, "stop.mp3"),
            "OBSTACLE": os.path.join(audio_dir, "obstacle_detected.mp3"),
            "GO": os.path.join(audio_dir, "go.mp3"),
            "TURN_LEFT": os.path.join(audio_dir, "turn_left.mp3"),
            "TURN_RIGHT": os.path.join(audio_dir, "turn_right.mp3"),
        }

        if not all(os.path.exists(p) for p in self.audio_files.values()):
            print("Audio disabled: missing files in voice/commands")
            return

        try:
            import pygame
            import pygame.mixer
            pygame.mixer.init()
            self.mode = "pygame"
            self.enabled = True

            def _play(path: str):
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.play()

            self._player = _play
            print("Audio enabled (pygame)")
            return
        except Exception as exc:
            print(f"Audio pygame unavailable: {exc}")

        if _which("mpg123"):
            self.mode = "system"
            self.enabled = True
            self._player = lambda p: subprocess.Popen(
                ["mpg123", "-q", p],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("Audio enabled (mpg123)")
            return

        if _which("aplay"):
            self.mode = "system"
            self.enabled = True
            self._player = lambda p: subprocess.Popen(
                ["aplay", p],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("Audio enabled (aplay)")
            return

        print("Audio disabled: no backend available")

    def play(self, command: str):
        if not self.enabled or command not in self.audio_files:
            return
        if self._player:
            self._player(self.audio_files[command])


def recvall(sock: socket.socket, length: int) -> bytes:
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


def main():
    audio = SafeAudioManager()

    cam_index = get_camera_index()
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    last_action = None
    last_audio_time = 0.0
    audio_cooldown = 3.0
    frame_count = 0

    print(f"Connecting to Mac server {SERVER_HOST}:{SERVER_PORT}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((SERVER_HOST, SERVER_PORT))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read from camera.")
                break

            frame_count += 1
            loop_t0 = time.time()

            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                continue

            payload = buffer.tobytes()
            sock.sendall(struct.pack("!I", len(payload)) + payload)

            header = recvall(sock, 4)
            if not header:
                print("Server disconnected.")
                break

            (length,) = struct.unpack("!I", header)
            response = recvall(sock, length)
            if not response:
                print("Server disconnected.")
                break

            data = json.loads(response.decode("utf-8"))
            action = data.get("action", "GO")
            reason = data.get("reason", "")
            yolo_fps = float(data.get("yolo_fps", 0.0))
            distance_critical = bool(data.get("distance_critical", False))

            current_time = time.time()
            if audio.enabled:
                if distance_critical and "STOP" in audio.audio_files:
                    audio.play("STOP")
                    print("STOP - Object within 1 meter")
                elif action == "STOP":
                    audio.play("STOP")
                    print("STOP")
                elif action == "GO":
                    audio.play("GO")
                    print("GO - Safe to proceed")
                elif action == "STEER_LEFT":
                    audio.play("TURN_LEFT")
                    print("TURN LEFT")
                elif action == "STEER_RIGHT":
                    audio.play("TURN_RIGHT")
                    print("TURN RIGHT")
                elif action == "WARN":
                    if action != last_action or (current_time - last_audio_time) > audio_cooldown:
                        audio.play("OBSTACLE")
                        last_audio_time = current_time
                        print("OBSTACLE DETECTED")

            last_action = action

            loop_fps = 1.0 / max(time.time() - loop_t0, 1e-6)
            if frame_count % PRINT_EVERY_N_FRAMES == 0:
                print(
                    f"[{frame_count}] action={action} reason={reason} "
                    f"loop_fps={loop_fps:.1f} yolo_fps={yolo_fps:.1f}"
                )

            if os.path.exists("STOP"):
                print("STOP file found, exiting.")
                break

    cap.release()


if __name__ == "__main__":
    main()
