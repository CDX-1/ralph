# detect_spatial.py
# Works on Mac (with GUI) AND Raspberry Pi (headless over SSH).
# YOLOv8 + spatial analysis + optional audio alerts (won't crash if audio unavailable).

import os
import time
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO

# ----------------------------
# Config (tune for your demo)
# ----------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
DETECT_EVERY_N_FRAMES = 1
CONF_THRESH = 0.45

OBSTACLE_CLASSES = {0, 1, 2, 3, 5, 7, 56}  # person, bicycle, car, motorcycle, bus, truck, chair

LEFT_MAX = 1 / 3
RIGHT_MIN = 2 / 3

AREA_WARN = 0.030
AREA_STOP = 0.070
AREA_GROWTH_APPROACH = 0.008

AUDIO_COOLDOWN = 3.0  # seconds between warning voice lines
DEBUG_SAVE_EVERY_N_FRAMES = 0  # set e.g. 60 to save frame every 60 frames in headless mode

# ----------------------------
# Environment detection
# ----------------------------
def is_headless() -> bool:
    # On Linux, DISPLAY is usually required for cv2.imshow
    if os.name != "posix":
        return False
    if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
        return False
    # macOS typically isn't headless in the same way; but this is fine.
    return True

HEADLESS = is_headless()

# ----------------------------
# Safe audio (never crash)
# ----------------------------
def init_audio():
    """
    Try pygame mixer; if unavailable, fallback to system players.
    Returns a dict with:
      mode: "pygame" | "system" | "none"
      play_fn: function(path)->None
      stop_fn: function()->None
    """
    # 1) Try pygame
    try:
        import pygame
        try:
            import pygame.mixer  # may not exist in some builds
            pygame.mixer.init()
            def play_pygame(path):
                # Stop current and play new
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
            def stop_pygame():
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
            return {"mode": "pygame", "play_fn": play_pygame, "stop_fn": stop_pygame}
        except Exception as e:
            print("⚠ pygame.mixer unavailable, falling back to system audio:", e)
    except Exception:
        pass

    # 2) System audio fallback (Linux: aplay/mpg123; macOS: afplay)
    # We'll try in order.
    def which(cmd):
        from shutil import which as _which
        return _which(cmd)

    if which("afplay"):
        def play_sys(path):
            subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        def stop_sys():
            pass
        return {"mode": "system", "play_fn": play_sys, "stop_fn": stop_sys}

    if which("mpg123"):
        def play_sys(path):
            subprocess.Popen(["mpg123", "-q", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        def stop_sys():
            pass
        return {"mode": "system", "play_fn": play_sys, "stop_fn": stop_sys}

    if which("aplay"):
        def play_sys(path):
            subprocess.Popen(["aplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        def stop_sys():
            pass
        return {"mode": "system", "play_fn": play_sys, "stop_fn": stop_sys}

    return {"mode": "none", "play_fn": lambda p: None, "stop_fn": lambda: None}


# --------------------------------
# Helper: compute spatial decision
# --------------------------------
def decide_action(frame_w: int, frame_h: int, dets: list, prev_area_by_key: dict):
    frame_area = float(frame_w * frame_h)

    risk_left = 0.0
    risk_center = 0.0
    risk_right = 0.0

    stop_triggered = False
    warn_triggered = False
    approach_triggered = False

    overlays = []

    for d in dets:
        cls = d["cls"]
        conf = d["conf"]
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
        prev_area = prev_area_by_key.get(ke# detect_spatial_pi5.py
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
    main()y, 0.0)
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

        overlays.append({
            "cls": cls,
            "conf": conf,
            "xyxy": (x1, y1, x2, y2),
            "region": region,
            "area_norm": area_norm,
            "delta_area": delta_area,
            "approaching": is_approaching,
        })

    if stop_triggered:
        return "STOP", "Close obstacle in path", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays

    if approach_triggered and risk_center > max(risk_left, risk_right) and risk_center > 0.05:
        return "WARN", "Approaching object ahead", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays

    risks = {"LEFT": risk_left, "CENTER": risk_center, "RIGHT": risk_right}
    best_zone = min(risks, key=risks.get)

    if best_zone == "CENTER":
        if warn_triggered and risk_center > 0.03:
            return "WARN", "Obstacle(s) nearby", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays
        return "GO", "Clear path", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays

    if best_zone == "LEFT":
        return "STEER_LEFT", "Safer corridor left", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays
    return "STEER_RIGHT", "Safer corridor right", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays


def draw_overlays(frame, overlays, action, reason, fps):
    h, w = frame.shape[:2]
    x_left = int(w * LEFT_MAX)
    x_right = int(w * RIGHT_MIN)
    cv2.line(frame, (x_left, 0), (x_left, h), (0, 0, 0), 2)
    cv2.line(frame, (x_right, 0), (x_right, h), (0, 0, 0), 2)

    cv2.putText(frame, f"ACTION: {action}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(frame, f"Reason: {reason}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    for o in overlays:
        x1, y1, x2, y2 = map(int, o["xyxy"])
        cls = o["cls"]
        conf = o["conf"]
        region = o["region"]
        area_norm = o["area_norm"]
        approaching = o["approaching"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        label = f"cls={cls} {conf:.2f} {region} area={area_norm:.3f}"
        if approaching:
            label += " APPROACH"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return frame


def main():
    print(f"HEADLESS={HEADLESS}")

    # Audio setup
    audio = init_audio()

    audio_dir = os.path.join("voice", "commands")
    audio_files = {
        "STOP": os.path.join(audio_dir, "stop.mp3"),
        "OBSTACLE": os.path.join(audio_dir, "obstacle_detected.mp3"),
    }

    audio_enabled = (audio["mode"] != "none") and all(os.path.exists(p) for p in audio_files.values())
    if audio_enabled:
        print(f"✓ Audio enabled ({audio['mode']})")
    else:
        print(f"⚠ Audio disabled (mode={audio['mode']}). Missing files or no audio backend.")

    # Model path: prefer local models/yolov8n.pt if present, else auto-download yolov8n.pt
    local_model = os.path.join("models", "yolov8n.pt")
    model_path = local_model if os.path.exists(local_model) else "yolov8n.pt"
    print("Using model:", model_path)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    prev_area_by_key = {}
    frame_count = 0
    last_action = None
    last_audio_time = 0.0
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from camera.")
            break

        frame_count += 1
        t0 = time.time()

        do_detect = (frame_count % DETECT_EVERY_N_FRAMES == 0)
        if do_detect:
            results = model(frame, verbose=False)[0]
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
                dets.append({"cls": cls, "conf": conf, "xyxy": (x1, y1, x2, y2)})

            action, reason, debug, prev_area_by_key, overlays = decide_action(
                frame.shape[1], frame.shape[0], dets, prev_area_by_key
            )

        fps = 1.0 / max(time.time() - t0, 1e-6)

        # ---- Audio behavior ----
        now = time.time()
        if audio_enabled:
            if action == "STOP":
                # stop line can repeat, but don't spam
                if (action != last_action) or (now - last_audio_time > 1.5):
                    audio["play_fn"](audio_files["STOP"])
                    last_audio_time = now
                    print("🔊 STOP")
            elif action in ("WARN", "STEER_LEFT", "STEER_RIGHT"):
                if (action != last_action) or (now - last_audio_time > AUDIO_COOLDOWN):
                    audio["play_fn"](audio_files["OBSTACLE"])
                    last_audio_time = now
                    print("🔊 OBSTACLE")

        last_action = action

        # ---- Output / visualization ----
        if not HEADLESS:
            frame = draw_overlays(frame, overlays, action, reason, fps)
            cv2.imshow("Spatial Analysis (YOLOv8)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        else:
            # Headless: print every ~10 frames
            if frame_count % 10 == 0:
                print(f"[{frame_count}] ACTION={action} reason={reason} fps={fps:.1f}")

            # Optional debug frame saving
            if DEBUG_SAVE_EVERY_N_FRAMES and (frame_count % DEBUG_SAVE_EVERY_N_FRAMES == 0):
                out_path = f"debug_{frame_count}.jpg"
                cv2.imwrite(out_path, frame)
                print("Saved", out_path)

    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()