import os
import time
import subprocess
import cv2
from ultralytics import YOLO

from config import (
    FRAME_W,
    FRAME_H,
    DETECT_EVERY_N_FRAMES,
    CONF_THRESH,
    RELEVANT_CLASSES,
)
from camera import get_camera_index
from spatial import decide_action


PRINT_EVERY_N_FRAMES = 10
DEBUG_SAVE_EVERY_N_FRAMES = 0


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

        # Try pygame first if available.
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

        # System players as fallback.
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


def init_midas():
    if os.environ.get("DISABLE_MIDAS", "").lower() in ("1", "true", "yes"):
        print("MiDaS disabled by DISABLE_MIDAS")
        return None

    try:
        from midas_depth import MidasDepth
    except Exception as exc:
        print(f"MiDaS unavailable: {exc}")
        return None

    try:
        print("Initializing MiDaS depth estimation...")
        midas = MidasDepth(
            device="cpu",
            input_size=(384, 216),
            close_threshold=0.5,
            run_every_n_frames=4,
        )
        print("MiDaS enabled")
        return midas
    except Exception as exc:
        print(f"MiDaS init failed: {exc}")
        return None


def main():
    audio = SafeAudioManager()
    midas = init_midas()
    midas_enabled = midas is not None

    model = YOLO("models/yolov8n.pt")

    cam_index = get_camera_index()
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    prev_area_by_key = {}
    frame_count = 0
    last_action = None
    last_audio_time = 0.0
    audio_cooldown = 3.0

    depth_close_count = 0
    next_turn_allowed = 0.0
    midas_override_active = False

    last_results = None
    last_yolo_fps = 0.0
    last_midas_fps = 0.0

    print("Starting headless loop")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from camera.")
            break

        frame_count += 1
        t0 = time.time()
        current_time = t0

        do_detect = (frame_count % DETECT_EVERY_N_FRAMES == 0)

        if do_detect:
            yolo_t0 = time.time()
            results = model(frame, verbose=False, classes=list(RELEVANT_CLASSES.keys()))[0]
            yolo_dt = max(time.time() - yolo_t0, 1e-6)
            last_yolo_fps = 1.0 / yolo_dt
            last_results = results
        else:
            results = last_results

        overlays = []
        action = "GO"
        reason = "No detections yet"

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

            action, reason, _debug, prev_area_by_key, overlays = decide_action(
                frame.shape[1], frame.shape[0], dets, prev_area_by_key
            )

        fps = 1.0 / max(time.time() - t0, 1e-6)

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

                if frame_count % 30 == 0:
                    status = "CRITICAL" if distance_critical else "OK"
                    print(
                        f"YOLO distance: {distance_ahead:.2f}m [{status}] area={area_norm:.4f}"
                    )

        if midas_enabled:
            midas_t0 = time.time()
            depth = midas.update(frame)
            midas_dt = max(time.time() - midas_t0, 1e-6)
            last_midas_fps = 1.0 / midas_dt
            if depth["valid"]:
                if frame_count % 30 == 0:
                    print(
                        "MiDaS fps={:.1f} roi={:.3f} left={:.3f} right={:.3f} close={}".format(
                            last_midas_fps,
                            depth["roi_v"],
                            depth["left_v"],
                            depth["right_v"],
                            depth["close_ahead"],
                        )
                    )

                if depth["close_ahead"]:
                    depth_close_count += 1
                    if depth_close_count >= 2 and current_time > next_turn_allowed:
                        midas_override_active = True
                        print(
                            "MiDaS override: wall ahead - STOP and TURN {}".format(
                                depth["turn_dir"].upper()
                            )
                        )
                        if audio.enabled and "STOP" in audio.audio_files:
                            audio.play("STOP")
                        next_turn_allowed = current_time + 0.4
                        depth_close_count = 0
                        action = "STOP"
                        reason = f"MiDaS: Wall ahead - turn {depth['turn_dir']}"
                else:
                    depth_close_count = 0
                    midas_override_active = False

        if audio.enabled and not midas_override_active:
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

        if frame_count % PRINT_EVERY_N_FRAMES == 0:
            print(
                f"[{frame_count}] action={action} reason={reason} "
                f"loop_fps={fps:.1f} yolo_fps={last_yolo_fps:.1f} midas_fps={last_midas_fps:.1f}"
            )

        if DEBUG_SAVE_EVERY_N_FRAMES and (frame_count % DEBUG_SAVE_EVERY_N_FRAMES == 0):
            out_path = f"debug_{frame_count}.jpg"
            cv2.imwrite(out_path, frame)
            print(f"Saved {out_path}")

        if os.path.exists("STOP"):
            print("STOP file found, exiting.")
            break

    cap.release()


if __name__ == "__main__":
    main()
