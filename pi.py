# detect_spatial_pi5.py
# Pi 5 (SSH/headless) optimized: YOLOv8n + spatial analysis + distance/approach filtering
# + optional audio (never crashes) + performance stats (loop_fps + detect_fps)
#
# Run (inside venv):
#   pip install ultralytics opencv-python-headless numpy
#   python detect_spatial_pi5.py
#
# Notes:
# - Designed for SSH/headless: NO cv2.imshow()
# - Uses corridor crop + imgsz=320 + class filtering + detect-every-N-frames cache

import os
import time
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO

# ----------------------------
# Config (Pi 5 performance)
# ----------------------------
CAM_INDEX = 0

# Faster capture size (tune)
FRAME_W, FRAME_H = 416, 240  # try 320x240 for max speed

# Run inference every N frames (2 is a good Pi 5 default)
DETECT_EVERY_N_FRAMES = 2

# YOLO input size (smaller = faster)
YOLO_IMGSZ = 320

# Confidence threshold
CONF_THRESH = 0.45

# Corridor crop (analyze middle slice only)
CROP_LEFT_FRAC = 0.20   # crop starts at 20% width
CROP_RIGHT_FRAC = 0.80  # crop ends at 80% width (middle 60%)

# COCO classes to consider as obstacles
# Common: 0=person, 2=car, 1=bicycle, 3=motorcycle, 5=bus, 7=truck, 56=chair
# For speed: use [0] (person only) or [0,2] (person+car)
MODEL_CLASSES = [0, 2]
OBSTACLE_CLASSES = set(MODEL_CLASSES)

# Spatial zones
LEFT_MAX = 1 / 3
RIGHT_MIN = 2 / 3

# Distance proxy thresholds (normalized box area)
AREA_WARN = 0.030
AREA_STOP = 0.070
AREA_GROWTH_APPROACH = 0.008

# Output behavior
PRINT_EVERY_N_FRAMES = 10
DEBUG_SAVE_EVERY_N_FRAMES = 0  # set e.g. 120 to save occasional debug frames

# Audio alerts (optional; won’t crash)
AUDIO_COOLDOWN = 3.0
AUDIO_REPEAT_STOP = 1.5

# ----------------------------
# Safe audio (never crash)
# ----------------------------

def which(cmd: str):
    from shutil import which as _which
    return _which(cmd)


def init_audio():
    """
    Try pygame mixer; if unavailable, fallback to system players.
    Returns dict:
      mode: "pygame" | "system" | "none"
      play_fn(path)->None
      stop_fn()->None
    """
    # Try pygame first
    try:
        import pygame
        try:
            import pygame.mixer
            pygame.mixer.init()
            def play_pygame(path):
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
            def stop_pygame():
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
            return {"mode": "pygame", "play_fn": play_pygame, "stop_fn": stop_pygame}
        except Exception as e:
            print("⚠ pygame.mixer unavailable:", e)
    except Exception:
        pass

    # System players
    if which("mpg123"):
        def play_sys(path):
            subprocess.Popen(["mpg123", "-q", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"mode": "system", "play_fn": play_sys, "stop_fn": lambda: None}

    if which("aplay"):
        def play_sys(path):
            subprocess.Popen(["aplay", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"mode": "system", "play_fn": play_sys, "stop_fn": lambda: None}

    # No audio available
    return {"mode": "none", "play_fn": lambda p: None, "stop_fn": lambda: None}


# --------------------------------
# Spatial decision
# --------------------------------

def decide_action(frame_w: int, frame_h: int, dets: list, prev_area_by_key: dict):
    frame_area = float(frame_w * frame_h)

    risk_left = 0.0
    risk_center = 0.0
    risk_right = 0.0

    stop_triggered = False
    warn_triggered = False
    approach_triggered = False

    for d in dets:
        cls = d["cls"]
        x1, y1, x2, y2 = d["xyxy"]

        cx = (x1 + x2) / 2.0
        box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        area_norm = box_area / frame_area

        if cx < frame_w * LEFT_MAX:
            region = "LEFT"
        elif cx > frame_w * RIGHT_MIN:
            region = "RIGHT"
        else:
            region = "CENTER"

        key = (cls, region)
        prev_area = prev_area_by_key.get(key, 0.0)
        delta_area = area_norm - prev_area
        prev_area_by_key[key] = area_norm

        is_approaching = delta_area > AREA_GROWTH_APPROACH
        if is_approaching:
            approach_triggered = True

        base_risk = area_norm * (1.2 if region == "CENTER" else 0.9)
        if is_approaching:
            base_risk *= 1.5

        if region == "LEFT":
            risk_left += base_risk
        elif region == "CENTER":
            risk_center += base_risk
        else:
            risk_right += base_risk

        if cls in OBSTACLE_CLASSES:
            if area_norm >= AREA_STOP and region == "CENTER":
                stop_triggered = True
            elif area_norm >= AREA_WARN:
                warn_triggered = True

    if stop_triggered:
        return "STOP", "Close obstacle in path", prev_area_by_key

    if approach_triggered and risk_center > max(risk_left, risk_right) and risk_center > 0.05:
        return "WARN", "Approaching object ahead", prev_area_by_key

    risks = {"LEFT": risk_left, "CENTER": risk_center, "RIGHT": risk_right}
    best_zone = min(risks, key=risks.get)

    if best_zone == "CENTER":
        if warn_triggered and risk_center > 0.03:
            return "WARN", "Obstacle(s) nearby", prev_area_by_key
        return "GO", "Clear path", prev_area_by_key

    if best_zone == "LEFT":
        return "STEER_LEFT", "Safer corridor left", prev_area_by_key

    return "STEER_RIGHT", "Safer corridor right", prev_area_by_key


# ----------------------------
# Main
# ----------------------------

def main():
    # Audio files
    audio = init_audio()
    audio_dir = os.path.join("voice", "commands")
    audio_files = {
        "STOP": os.path.join(audio_dir, "stop.mp3"),
        "OBSTACLE": os.path.join(audio_dir, "obstacle_detected.mp3"),
    }
    audio_enabled = (audio["mode"] != "none") and all(os.path.exists(p) for p in audio_files.values())
    print(f"Audio mode={audio['mode']} enabled={audio_enabled}")

    # Load model (prefer your local file)
    local_model = os.path.join("models", "yolov8n.pt")
    model_path = local_model if os.path.exists(local_model) else "yolov8n.pt"
    print("Using model:", model_path)
    model = YOLO(model_path)

    # Limit classes for speed
    model.overrides["classes"] = MODEL_CLASSES

    # Camera
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    prev_area_by_key = {}
    frame_count = 0

    last_results = None
    last_action = None
    last_audio_time = 0.0

    # Performance tracking
    loop_t0 = time.time()
    detect_t0 = time.time()
    loop_frames = 0
    detect_frames = 0

    print("Starting loop... (SSH/headless)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from camera.")
            break

        frame_count += 1
        loop_frames += 1

        # Corridor crop for inference
        h, w = frame.shape[:2]
        crop_x1 = int(w * CROP_LEFT_FRAC)
        crop_x2 = int(w * CROP_RIGHT_FRAC)
        roi = frame[:, crop_x1:crop_x2]

        # Inference scheduling
        do_detect = (frame_count % DETECT_EVERY_N_FRAMES == 0)

        if do_detect:
            # Run YOLO on ROI
            results = model(roi, imgsz=YOLO_IMGSZ, verbose=False)[0]
            last_results = results
            detect_frames += 1
        else:
            results = last_results

        # Build detections (convert ROI coords -> full-frame coords)
        dets = []
        if results is not None and results.boxes is not None:
            for b in results.boxes:
                conf = float(b.conf[0])
                if conf < CONF_THRESH:
                    continue
                cls = int(b.cls[0])

                bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = bx1 + crop_x1, by1, bx2 + crop_x1, by2

                dets.append({"cls": cls, "conf": conf, "xyxy": (x1, y1, x2, y2)})

        action, reason, prev_area_by_key = decide_action(w, h, dets, prev_area_by_key)

        # Audio alerts (never crash)
        now = time.time()
        if audio_enabled:
            if action == "STOP":
                if (action != last_action) or (now - last_audio_time > AUDIO_REPEAT_STOP):
                    audio["play_fn"](audio_files["STOP"])
                    last_audio_time = now
            elif action in ("WARN", "STEER_LEFT", "STEER_RIGHT"):
                if (action != last_action) or (now - last_audio_time > AUDIO_COOLDOWN):
                    audio["play_fn"](audio_files["OBSTACLE"])
                    last_audio_time = now

        last_action = action

        # Compute performance every second
        t = time.time()
        if t - loop_t0 >= 1.0:
            loop_fps = loop_frames / (t - loop_t0)
            loop_frames = 0
            loop_t0 = t

            # detect fps is how often inference runs
            detect_fps = detect_frames / max(t - detect_t0, 1e-6)
            detect_frames = 0
            detect_t0 = t

            print(f"loop_fps={loop_fps:.1f} detect_fps={detect_fps:.1f} action={action} reason={reason}")

        # Optional: periodic debug frame dump (for quick checking via scp)
        if DEBUG_SAVE_EVERY_N_FRAMES and (frame_count % DEBUG_SAVE_EVERY_N_FRAMES == 0):
            out_path = f"debug_{frame_count}.jpg"
            cv2.imwrite(out_path, frame)
            print("Saved", out_path)

        # Optional: exit file flag (touch STOP to quit)
        if os.path.exists("STOP"):
            print("STOP file found, exiting.")
            break

    cap.release()


if __name__ == "__main__":
    main()
