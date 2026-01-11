import cv2
import numpy as np
import time
from ultralytics import YOLO

from vision import (
    compute_turn_and_confidence,
    decide_action,
    detect_floor_path_line,
    load_floor_calibration,
    pixel_to_world,
)

last_turn = "LEFT"

model = YOLO("models/yolov8n.pt")
H, far_left, far_right = load_floor_calibration("floor_calib.json")

cap = cv2.VideoCapture(0)
prev_time = 0
paused = False
paused_frame = None

while True:
    # Handle keyboard input (check before reading frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("p") or key == ord(" "):  # 'p' or spacebar to toggle pause
        paused = not paused
        if paused:
            print("\n=== PAUSED ===")
        else:
            print("=== RESUMED ===")
    
    # If paused, use the last frame, otherwise read a new one
    if paused:
        if paused_frame is not None:
            frame = paused_frame.copy()
        else:
            # If we haven't captured a frame yet, read one to pause on
            ret, frame = cap.read()
            if not ret:
                break
            paused_frame = frame.copy()
    else:
        ret, frame = cap.read()
        if not ret:
            break
        paused_frame = frame.copy()  # Update paused frame

    h, w, _ = frame.shape
    fifth = w // 5

    # Draw 4 divider lines for 5 columns
    cv2.line(frame, (fifth, 0), (fifth, h), (255, 255, 255), 2)
    cv2.line(frame, (2 * fifth, 0), (2 * fifth, h), (255, 255, 255), 2)
    cv2.line(frame, (3 * fifth, 0), (3 * fifth, h), (255, 255, 255), 2)
    cv2.line(frame, (4 * fifth, 0), (4 * fifth, h), (255, 255, 255), 2)
    cv2.line(frame, far_left, far_right, (255, 128, 0), 2)
    floor_y, floor_origin, floor_target = detect_floor_path_line(frame)
    desired_turn = None
    path_bias = 0.0
    if floor_y is not None:
        cv2.line(frame, floor_origin, floor_target, (0, 200, 255), 2)
        dx = floor_target[0] - floor_origin[0]
        path_bias = float(np.clip(dx / max(1.0, w * 0.3), -1.0, 1.0))
        if dx < -w * 0.05:
            desired_turn = "LEFT"
        elif dx > w * 0.05:
            desired_turn = "RIGHT"

    # Process frame (even when paused, so we can see bounding boxes and distances)
    results = model(frame, verbose=False)[0]

    # Track objects (distance, box_area_normalized) in each region (5 columns)
    ll_objects = []
    l_objects = []
    middle_objects = []
    r_objects = []
    rr_objects = []
    
    frame_area = float(w * h)  # Total frame area for normalization

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

        # Calculate bounding box area (normalized 0-1)
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        box_area_normalized = box_area / frame_area

        # bottom-center = assumed floor contact point
        bx = cx
        by = y2

        # clamp
        bx = max(0, min(bx, w - 1))
        by = max(0, min(by, h - 1))

        X, Y = pixel_to_world(H, bx, by)      # meters on floor plane
        dist_m = (X**2 + Y**2) ** 0.5         # distance from near-left origin

        # Store as (distance, box_area_normalized) tuple
        obj_data = (dist_m, box_area_normalized)

        # Determine which columns this box overlaps (not just based on center)
        # A box overlaps a column if [x1, x2] intersects the column's x-range
        # Standard interval intersection: [a1, a2] intersects [b1, b2] if a1 < b2 and a2 > b1
        overlaps = []
        for col_start, col_end, region_name, region_list in col_boundaries:
            if x1 < col_end and x2 > col_start:  # Interval intersection
                overlaps.append((region_name, region_list))

        # Add this object to all columns it overlaps
        for region_name, region_list in overlaps:
            region_list.append(obj_data)

        # Determine primary region for display (based on center)
        if cx < fifth:
            primary_region = "LL"
        elif cx < 2 * fifth:
            primary_region = "L"
        elif cx < 3 * fifth:
            primary_region = "M"
        elif cx < 4 * fifth:
            primary_region = "R"
        else:
            primary_region = "RR"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (bx, by), 5, (0, 0, 255), -1)

        # Show all regions this object overlaps
        overlap_str = "+".join([r for r, _ in overlaps])
        cv2.putText(frame, f"{overlap_str} {dist_m:.2f}m", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Get minimum distance in each region (None if no objects) for display
    LL_dist = min([obj[0] for obj in ll_objects]) if ll_objects else None
    L_dist = min([obj[0] for obj in l_objects]) if l_objects else None
    M_dist = min([obj[0] for obj in middle_objects]) if middle_objects else None
    R_dist = min([obj[0] for obj in r_objects]) if r_objects else None
    RR_dist = min([obj[0] for obj in rr_objects]) if rr_objects else None

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Pass all objects (distance, box_area) tuples for weighted risk calculation
    action, last_turn, risk_meta = decide_action(
        ll_objects,
        l_objects,
        middle_objects,
        r_objects,
        rr_objects,
        last_turn,
        desired_turn,
    )
    turn, confidence = compute_turn_and_confidence(action, risk_meta, path_bias)
    cv2.putText(frame, f"ACTION: {action}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"TURN: {turn:+.2f} CONF: {confidence:.2f}", (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display distance info (5 columns)
    dist_info = f"LL:{LL_dist:.1f}" if LL_dist else "LL:--"
    dist_info += f" L:{L_dist:.1f}" if L_dist else " L:--"
    dist_info += f" M:{M_dist:.1f}" if M_dist else " M:--"
    dist_info += f" R:{R_dist:.1f}" if R_dist else " R:--"
    dist_info += f" RR:{RR_dist:.1f}" if RR_dist else " RR:--"
    cv2.putText(frame, f"{dist_info} FPS:{fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display pause status if paused
    if paused:
        cv2.putText(frame, "PAUSED - Press SPACE or 'p' to resume", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YOLO + Floor Distance", frame)

cap.release()
cv2.destroyAllWindows()
