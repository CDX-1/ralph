# detect_spatial_mac.py
# Mac webcam demo: YOLOv8 object detection + spatial analysis (left/center/right),
# distance proxy (box area), temporal approach (area growth), and a simple action decision.
#
# Keys:
#   q = quit
#
# Install:
#   python3 -m venv ai-test
#   source ai-test/bin/activate
#   pip install --upgrade pip
#   pip install ultralytics opencv-python numpy

import time
import cv2
import numpy as np
import pygame
import os
from ultralytics import YOLO
from colorama import Fore

# ----------------------------
# Config (tune for your demo)
# ----------------------------
# Auto-detect available cameras and use the second one if available
def find_cameras():
    available_cameras = []
    # Suppress OpenCV errors during camera detection
    cv2.setLogLevel(0)  # Suppress error messages
    
    for i in range(3):  # Check first 3 camera indices (reduced to avoid errors)
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows for faster detection
            if cap.isOpened():
                # Verify camera works by reading a frame
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
            cap.release()
        except:
            pass
    
    # Re-enable logging
    cv2.setLogLevel(3)
    return available_cameras

# Find and use the second available camera (or first if only one exists)
print("Detecting cameras...")
available_cams = find_cameras()
if len(available_cams) > 1:
    CAM_INDEX = available_cams[1]  # Use second camera
    print(f"✓ Using camera index {CAM_INDEX} (found cameras: {available_cams})")
elif len(available_cams) > 0:
    CAM_INDEX = available_cams[0]
    print(Fore.GREEN + f" Only one camera found at index {CAM_INDEX}")
else:
    CAM_INDEX = 0
    print(Fore.RED + "No cameras detected, defaulting to index 0")

# Lower resolution = faster
FRAME_W, FRAME_H = 640, 480

# Run detection every N frames (2 or 3 helps FPS a lot)
DETECT_EVERY_N_FRAMES = 1

# Only consider detections above this confidence
CONF_THRESH = 0.45

# COCO dataset class IDs - only include relevant outdoor/navigation obstacles
# Excluded: animals, food, kitchen items, sports equipment, electronics, etc.
RELEVANT_CLASSES = {
    # People
    0: 'person',
    
    # Vehicles
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    
    # Street furniture & infrastructure
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    
    # Common outdoor obstacles
    14: 'bird',  # birds can be obstacles
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    60: 'dining table',
    62: 'tv',
    63: 'laptop',
    65: 'bed',
    66: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
}

# Classes to treat as critical "obstacles" requiring immediate action
# Subset of RELEVANT_CLASSES that are high priority
OBSTACLE_CLASSES = {0, 1, 2, 3, 5, 7, 9, 11, 13, 56, 57, 58, 60}  # people, vehicles, furniture

# Corridor definitions (fractions of width)
LEFT_MAX = 1/3
RIGHT_MIN = 2/3

# "Distance" proxy using normalized box area (box_area / frame_area)
# Tune these based on your camera distance + environment.
AREA_WARN = 0.030     # medium distance -> warn/slow/steer
AREA_STOP = 0.070     # close -> stop
# Temporal approach detection: if area increases by this much between frames, treat as approaching
AREA_GROWTH_APPROACH = 0.008

# --------------------------------
# Helper: compute spatial decision
# --------------------------------
def decide_action(frame_w: int, frame_h: int, dets: list, prev_area_by_key: dict):
    """
    dets: list of dicts:
      {
        "cls": int,
        "conf": float,
        "xyxy": (x1,y1,x2,y2)
      }

    prev_area_by_key: stores last area_norm per "key" for crude approach detection.
      key is (cls, region_bucket) for simplicity (hackathon-level tracking)

    Returns:
      action: str in {"GO", "STEER_LEFT", "STEER_RIGHT", "WARN", "STOP"}
      reason: short string
      debug: dict
      updated_prev_area_by_key
      overlays: list of per-object info to draw
    """
    frame_area = float(frame_w * frame_h)

    # Risk scores per zone (lower is better)
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

        # Basic geometry
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        area_norm = box_area / frame_area

        # Determine region
        if cx < frame_w * LEFT_MAX:
            region = "LEFT"
        elif cx > frame_w * RIGHT_MIN:
            region = "RIGHT"
        else:
            region = "CENTER"

        # Very simple "tracking key": class + region bucket
        key = (cls, region)
        prev_area = prev_area_by_key.get(key, 0.0)
        delta_area = area_norm - prev_area
        prev_area_by_key[key] = area_norm

        is_approaching = delta_area > AREA_GROWTH_APPROACH
        if is_approaching:
            approach_triggered = True

        # Weight risk by how "big" it is and whether it's centered
        # (this is hackathon-level but works)
        base_risk = area_norm * (1.2 if region == "CENTER" else 0.9)

        # Add extra risk if approaching
        if is_approaching:
            base_risk *= 1.5

        # Add to region risk
        if region == "LEFT":
            risk_left += base_risk
        elif region == "CENTER":
            risk_center += base_risk
        else:
            risk_right += base_risk

        # Decide warn/stop triggers (only if the object is in obstacle classes)
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

    # Primary decision logic:
    # 1) If close obstacle in center -> STOP
    if stop_triggered:
        return "STOP", "Close obstacle in path", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays

    # 2) If approaching strongly + in center risk high -> WARN (or STOP if you want)
    if approach_triggered and risk_center > max(risk_left, risk_right) and risk_center > 0.05:
        return "WARN", "Approaching object ahead", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays

    # 3) Choose lowest-risk corridor for steering (if center is risky)
    risks = {"LEFT": risk_left, "CENTER": risk_center, "RIGHT": risk_right}
    best_zone = min(risks, key=risks.get)

    # If center is best and no warn -> GO
    if best_zone == "CENTER":
        if warn_triggered and risk_center > 0.03:
            return "WARN", "Obstacle(s) nearby", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays
        return "GO", "Clear path", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays

    # If center is worse than a side, steer
    if best_zone == "LEFT":
        return "STEER_LEFT", "Safer corridor left", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays
    else:
        return "STEER_RIGHT", "Safer corridor right", {"risk": (risk_left, risk_center, risk_right)}, prev_area_by_key, overlays


def draw_overlays(frame, overlays, action, reason, fps):
    h, w = frame.shape[:2]

    # Draw corridor lines
    x_left = int(w * LEFT_MAX)
    x_right = int(w * RIGHT_MIN)
    cv2.line(frame, (x_left, 0), (x_left, h), (0, 0, 0), 2)
    cv2.line(frame, (x_right, 0), (x_right, h), (0, 0, 0), 2)

    # Draw action banner
    cv2.putText(frame, f"ACTION: {action}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(frame, f"Reason: {reason}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Draw each detection
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
    # Initialize pygame mixer for audio
    pygame.mixer.init()
    
    # Audio paths (relative to project root)
    audio_dir = os.path.join('voice', 'commands')
    audio_files = {
        'STOP': os.path.join(audio_dir, 'stop.mp3'),
        'OBSTACLE': os.path.join(audio_dir, 'obstacle_detected.mp3'),
        'GO': os.path.join(audio_dir, 'go.mp3'),
        'TURN_LEFT': os.path.join(audio_dir, 'turn_left.mp3'),
        'TURN_RIGHT': os.path.join(audio_dir, 'turn_right.mp3'),
    }
    
    # Check audio files
    audio_enabled = all(os.path.exists(f) for f in audio_files.values())
    if audio_enabled:
        print("✓ Audio enabled")
    else:
        print("⚠ Some audio files missing - running without audio")
    
    model = YOLO("models/yolov8n.pt")  # use models directory

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    prev_area_by_key = {}
    frame_count = 0
    last_action = None
    last_audio_time = 0
    audio_cooldown = 3.0  # seconds between audio alerts

    last_results = None  # cache results when skipping frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from camera.")
            break

        frame_count += 1

        t0 = time.time()

        # Optionally skip detections to speed up
        do_detect = (frame_count % DETECT_EVERY_N_FRAMES == 0)

        if do_detect:
            # Only detect relevant classes - more efficient than filtering after
            results = model(frame, verbose=False, classes=list(RELEVANT_CLASSES.keys()))[0]
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
                
                # Classes already filtered at detection time via model(classes=...)
                # No need to filter again here
                
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()

                dets.append({
                    "cls": cls,
                    "conf": conf,
                    "xyxy": (x1, y1, x2, y2),
                })

            action, reason, debug, prev_area_by_key, overlays = decide_action(
                frame.shape[1], frame.shape[0], dets, prev_area_by_key
            )

        fps = 1.0 / max(time.time() - t0, 1e-6)
        
        # Calculate distance to object directly ahead
        distance_ahead = None  # Distance in meters
        distance_critical = False  # Flag for critical distance
        
        if overlays:
            # Find object in center (directly ahead)
            center_objects = [o for o in overlays if o['region'] == 'CENTER']
            if center_objects:
                # Use the largest object in center as the main obstacle
                main_object = max(center_objects, key=lambda x: x['area_norm'])
                area_norm = main_object['area_norm']
                
                # Estimate distance: larger area = closer (inverse relationship)
                # Calibration: area 0.1 ≈ 1m, area 0.05 ≈ 2m, area 0.01 ≈ 10m
                distance_ahead = max((0.3, 10.0 * (0.05 / max(area_norm, 0.001))) / 2)
                
                # Check if within critical distance (1 meter)
                if distance_ahead <= 1.0:
                    distance_critical = True
                
                # Print distance info every 30 frames (reduce console spam)
                if frame_count % 30 == 0:
                    status = "CRITICAL!" if distance_critical else "OK"
                    print(f"Distance ahead: {distance_ahead:.2f}m [{status}] (area: {area_norm:.4f})")
        

        # Play audio based on action
        current_time = time.time()
        
        if audio_enabled:
            # Priority 1: If distance is within 1 meter, override with STOP
            if distance_critical and 'STOP' in audio_files:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load(audio_files['STOP'])
                    pygame.mixer.music.play()
                    print(f"STOP - Object within 1 meter!")
            # For STOP action, play continuously (repeat when finished)
            elif action == "STOP" and 'STOP' in audio_files:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load(audio_files['STOP'])
                    pygame.mixer.music.play()
                    print(f"STOP")
            # For GO action, play continuously (repeat when finished)
            elif action == "GO" and 'GO' in audio_files:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load(audio_files['GO'])
                    pygame.mixer.music.play()
                    print(f"GO - Safe to proceed")
            # For STEER_LEFT action, play turn_left continuously
            elif action == "STEER_LEFT" and 'TURN_LEFT' in audio_files:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load(audio_files['TURN_LEFT'])
                    pygame.mixer.music.play()
                    print(f"TURN LEFT")
            # For STEER_RIGHT action, play turn_right continuously
            elif action == "STEER_RIGHT" and 'TURN_RIGHT' in audio_files:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load(audio_files['TURN_RIGHT'])
                    pygame.mixer.music.play()
                    print(f"TURN RIGHT")
            # For WARN action, use obstacle detected with cooldown
            elif action == "WARN" and 'OBSTACLE' in audio_files:
                if action != last_action or (current_time - last_audio_time) > audio_cooldown:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load(audio_files['OBSTACLE'])
                        pygame.mixer.music.play()
                        last_audio_time = current_time
                        print(f"OBSTACLE DETECTED")
        
        last_action = action

        frame = draw_overlays(frame, overlays, action, reason, fps)
        cv2.imshow("Spatial Analysis (YOLOv8)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
