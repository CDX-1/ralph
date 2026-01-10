import time
import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from camera import get_camera_index
from audio import AudioManager
from spatial import decide_action, draw_overlays
from midas_depth import MidasDepth


def main():
    audio = AudioManager()
    
    print("\nInitializing MiDaS depth estimation...")
    print("This may take a few minutes on first run (downloading ~25MB model)")
    print("Press 'c' during runtime to calibrate close threshold")
    
    midas = None
    midas_enabled = False
    
    try:
        midas = MidasDepth(
            device="cpu",
            input_size=(384, 216),
            close_threshold=0.65,
            run_every_n_frames=4
        )
        midas_enabled = True
        print("✓ MiDaS depth estimation enabled")
    except Exception as e:
        print(f"\n⚠ MiDaS initialization failed: {e}")
        print("⚠ Continuing with YOLO-only mode...")
        print("Troubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try running again (download will resume)")
        print("  3. Disable VPN/firewall temporarily")
        print("  4. Model downloads to: C:\\Users\\<user>\\.cache\\torch\\hub\\")
        midas_enabled = False
    
    model = YOLO("models/yolov8n.pt")
    
    cam_index = get_camera_index()
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    prev_area_by_key = {}
    frame_count = 0
    last_action = None
    last_audio_time = 0
    audio_cooldown = 3.0
    
    depth_close_count = 0
    next_turn_allowed = 0.0
    midas_override_active = False

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
        
        distance_ahead = None
        distance_critical = False
        depth = {"valid": False, "depth_vis": None}
        
        if overlays:
            center_objects = [o for o in overlays if o['region'] == 'CENTER']
            if center_objects:
                main_object = max(center_objects, key=lambda x: x['area_norm'])
                area_norm = main_object['area_norm']
                
                distance_ahead = max(0.3, 10.0 * (0.05 / max(area_norm, 0.001))) / 2
                
                if distance_ahead <= 1.0:
                    distance_critical = True
                
                if frame_count % 30 == 0:
                    status = "CRITICAL!" if distance_critical else "OK"
                    print(f"YOLO Distance ahead: {distance_ahead:.2f}m [{status}] (area: {area_norm:.4f})")
        
        if midas_enabled:
            depth = midas.update(frame)
            
            if depth["valid"]:
                if frame_count % 30 == 0:
                    print(f"MiDaS: roi={depth['roi_v']:.3f}, left={depth['left_v']:.3f}, right={depth['right_v']:.3f}, close={depth['close_ahead']}")
                
                if depth["close_ahead"]:
                    depth_close_count += 1
                    
                    if depth_close_count >= 2 and current_time > next_turn_allowed:
                        midas_override_active = True
                        print(f"\n!!! MiDaS OVERRIDE: Wall detected ahead - STOP and TURN {depth['turn_dir'].upper()} !!!\n")
                        
                        if audio.enabled and 'STOP' in audio.audio_files:
                            audio.play('STOP')
                        
                        next_turn_allowed = current_time + 0.4
                        depth_close_count = 0
                        
                        action = "STOP"
                        reason = f"MiDaS: Wall ahead - turn {depth['turn_dir']}"
                else:
                    depth_close_count = 0
                    midas_override_active = False

        current_time = time.time()
        
        if audio.enabled and not midas_override_active:
            if distance_critical and 'STOP' in audio.audio_files:
                audio.play('STOP')
                print(f"STOP - Object within 1 meter!")
            elif action == "STOP":
                audio.play('STOP')
                print(f"STOP")
            elif action == "GO":
                audio.play('GO')
                print(f"GO - Safe to proceed")
            elif action == "STEER_LEFT":
                audio.play('TURN_LEFT')
                print(f"TURN LEFT")
            elif action == "STEER_RIGHT":
                audio.play('TURN_RIGHT')
                print(f"TURN RIGHT")
            elif action == "WARN":
                if action != last_action or (current_time - last_audio_time) > audio_cooldown:
                    audio.play('OBSTACLE')
                    last_audio_time = current_time
                    print(f"OBSTACLE DETECTED")
        
        last_action = action

        frame = draw_overlays(frame, overlays, action, reason, fps)
        
        if midas_override_active:
            cv2.putText(frame, "MIDAS OVERRIDE ACTIVE", (10, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        cv2.imshow("Spatial Analysis (YOLOv8)", frame)
        
        if midas_enabled and depth["valid"] and depth["depth_vis"] is not None:
            cv2.imshow('MiDaS Depth Map', depth["depth_vis"])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and midas_enabled:
            print("\n" + "="*50)
            print("CALIBRATION MODE")
            print("="*50)
            print("Position at desired stop distance...")
            print("Sampling for 2 seconds...\n")
            
            samples = []
            cal_start = time.time()
            while time.time() - cal_start < 2.0:
                ret, cal_frame = cap.read()
                if ret:
                    val = midas.get_calibration_value(cal_frame)
                    samples.append(val)
                    print(f"Sample: {val:.4f}")
                    time.sleep(0.1)
            
            if samples:
                avg = np.mean(samples)
                recommended = avg * 0.95
                print(f"\nAverage: {avg:.4f}")
                print(f"Recommended: {recommended:.4f}")
                print(f"Current: {midas.close_threshold:.4f}")
                print(f"\nUpdate: MidasDepth(close_threshold={recommended:.4f})")
                print("="*50 + "\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
