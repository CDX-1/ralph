import cv2
import json
import numpy as np

# Distance thresholds (in meters)
CRITICAL_DISTANCE = 0.5   # Emergency stop - object extremely close
STOP_DISTANCE = 0.8       # Stop if object closer than this
WARN_DISTANCE = 1.5       # Start steering if object closer than this
SAFE_DISTANCE = 2.5       # Objects beyond this are low priority

# Bounding box size thresholds (normalized 0-1)
LARGE_OBJECT = 0.15       # Box area > 15% of frame
MEDIUM_OBJECT = 0.08      # Box area > 8% of frame
SMALL_OBJECT = 0.03       # Box area > 3% of frame


def calculate_threat_score(objects):
    """
    Calculate threat score combining distance and object size.
    Closer objects and larger boxes = higher threat.
    Returns: (threat_score, min_distance, max_area)
    """
    if not objects:
        return 0.0, None, None

    distances = [obj[0] for obj in objects]
    areas = [obj[1] for obj in objects]
    min_dist = min(distances)
    max_area = max(areas)

    threat = 0.0
    for dist, area in objects:
        # Distance component: inverse square (closer = much higher weight)
        dist_threat = 1.0 / max(dist, 0.1) ** 2
        
        # Size component: larger objects get exponential boost
        if area >= LARGE_OBJECT:
            size_multiplier = 3.0
        elif area >= MEDIUM_OBJECT:
            size_multiplier = 2.0
        elif area >= SMALL_OBJECT:
            size_multiplier = 1.5
        else:
            size_multiplier = 1.0
        
        # Combined threat for this object
        threat += dist_threat * size_multiplier * (1.0 + area)

    return threat, min_dist, max_area


def has_critical_obstacle(objects):
    """Check if any object is critically close AND large enough to matter."""
    if not objects:
        return False
    
    for dist, area in objects:
        # Critical if very close regardless of size, or close with medium+ size
        if dist < CRITICAL_DISTANCE:
            return True
        if dist < STOP_DISTANCE and area >= SMALL_OBJECT:
            return True
    
    return False


def has_blocking_obstacle(objects):
    """Check if region has obstacles that should block forward movement."""
    if not objects:
        return False
    
    for dist, area in objects:
        # Block if close with meaningful size
        if dist < STOP_DISTANCE and area >= SMALL_OBJECT:
            return True
        # Block if medium distance but very large object
        if dist < WARN_DISTANCE and area >= MEDIUM_OBJECT:
            return True
    
    return False


def get_weighted_steer_distance(objects):
    """
    Get effective steering distance weighted by object size.
    Larger objects influence steering from further away.
    """
    if not objects:
        return None
    
    weighted_dist = 0.0
    total_weight = 0.0
    
    for dist, area in objects:
        # Weight by size - larger objects count more
        weight = 1.0 + area * 5.0  # area is 0-1, so weight is 1-6
        weighted_dist += dist * weight
        total_weight += weight
    
    return weighted_dist / total_weight if total_weight > 0 else None


def decide_action(LL_objects, L_objects, M_objects, R_objects, RR_objects, last_turn, desired_turn, verbose=True):
    """
    Decide action using BOTH distance and bounding box size.
    
    Key principles:
    1. Larger objects trigger avoidance from further away
    2. Small distant objects are mostly ignored
    3. Close large objects = emergency stop
    4. Medium-range large objects = steer around
    5. Considers object size when choosing turn direction
    """
    
    # Calculate threat scores for each region
    LL_threat, LL_min, LL_max_area = calculate_threat_score(LL_objects)
    L_threat, L_min, L_max_area = calculate_threat_score(L_objects)
    M_threat, M_min, M_max_area = calculate_threat_score(M_objects)
    R_threat, R_min, R_max_area = calculate_threat_score(R_objects)
    RR_threat, RR_min, RR_max_area = calculate_threat_score(RR_objects)

    # Aggregate left and right
    left_threat = LL_threat + L_threat
    left_min = min([d for d in [LL_min, L_min] if d is not None], default=None)
    left_max_area = max([a for a in [LL_max_area, L_max_area] if a is not None], default=None)

    right_threat = R_threat + RR_threat
    right_min = min([d for d in [R_min, RR_min] if d is not None], default=None)
    right_max_area = max([a for a in [R_max_area, RR_max_area] if a is not None], default=None)

    # Weighted steering distances (accounts for object size)
    left_steer_dist = get_weighted_steer_distance(LL_objects + L_objects)
    right_steer_dist = get_weighted_steer_distance(R_objects + RR_objects)
    middle_steer_dist = get_weighted_steer_distance(M_objects)

    risk_meta = {
        "LL_threat": LL_threat,
        "L_threat": L_threat,
        "M_threat": M_threat,
        "R_threat": R_threat,
        "RR_threat": RR_threat,
        "left_threat": left_threat,
        "right_threat": right_threat,
        "left_min": left_min,
        "right_min": right_min,
        "M_min": M_min,
        "left_max_area": left_max_area,
        "right_max_area": right_max_area,
        "M_max_area": M_max_area,
    }

    if verbose:
        print("\n=== Decision Debug (Distance + Size) ===")
        print(f"Threats: LL={LL_threat:.2f} L={L_threat:.2f} M={M_threat:.2f} R={R_threat:.2f} RR={RR_threat:.2f}")
        
        def fmt_dist(d): return f"{d:.2f}m" if d else "None"
        def fmt_area(a): return f"{a*100:.1f}%" if a else "None"
        
        print(f"Distances: LL={fmt_dist(LL_min)} L={fmt_dist(L_min)} M={fmt_dist(M_min)} R={fmt_dist(R_min)} RR={fmt_dist(RR_min)}")
        print(f"Max Areas: LL={fmt_area(LL_max_area)} L={fmt_area(L_max_area)} M={fmt_area(M_max_area)} R={fmt_area(R_max_area)} RR={fmt_area(RR_max_area)}")
        print(f"Aggregated: left_threat={left_threat:.2f} (min={fmt_dist(left_min)}, area={fmt_area(left_max_area)})")
        print(f"            right_threat={right_threat:.2f} (min={fmt_dist(right_min)}, area={fmt_area(right_max_area)})")

    # CRITICAL STOP: Object extremely close in middle
    if has_critical_obstacle(M_objects):
        if verbose:
            print(f"→ STOP: Critical obstacle in middle (dist={M_min:.2f}m, area={M_max_area*100:.1f}%)")
        return "STOP", last_turn, risk_meta

    # EMERGENCY STOP: All directions blocked by close obstacles
    left_critical = has_critical_obstacle(LL_objects + L_objects)
    right_critical = has_critical_obstacle(R_objects + RR_objects)
    middle_critical = has_critical_obstacle(M_objects)
    
    if (left_critical and right_critical) or (middle_critical and left_critical and right_critical):
        if verbose:
            print(f"→ STOP: Surrounded by obstacles")
        return "STOP", last_turn, risk_meta

    # BLOCKING CHECK: Middle has obstacle that should block forward movement
    middle_blocked = has_blocking_obstacle(M_objects)
    
    if middle_blocked:
        # Need to steer around middle obstacle
        left_clear = not has_blocking_obstacle(LL_objects + L_objects)
        right_clear = not has_blocking_obstacle(R_objects + RR_objects)
        
        if left_clear and right_clear:
            # Both sides clear - choose based on threat scores and desired turn
            if desired_turn == "LEFT" and left_threat <= right_threat * 1.3:
                if verbose:
                    print(f"→ STEER_LEFT: Middle blocked, both sides clear, following path (left_threat={left_threat:.2f} vs right={right_threat:.2f})")
                return "STEER_LEFT", "LEFT", risk_meta
            elif desired_turn == "RIGHT" and right_threat <= left_threat * 1.3:
                if verbose:
                    print(f"→ STEER_RIGHT: Middle blocked, both sides clear, following path (right_threat={right_threat:.2f} vs left={left_threat:.2f})")
                return "STEER_RIGHT", "RIGHT", risk_meta
            elif left_threat < right_threat * 0.7:
                if verbose:
                    print(f"→ STEER_LEFT: Middle blocked, left significantly safer (threat: {left_threat:.2f} < {right_threat:.2f})")
                return "STEER_LEFT", "LEFT", risk_meta
            else:
                if verbose:
                    print(f"→ STEER_RIGHT: Middle blocked, right safer or equal (threat: {right_threat:.2f} <= {left_threat:.2f})")
                return "STEER_RIGHT", "RIGHT", risk_meta
        
        elif left_clear:
            if verbose:
                print(f"→ STEER_LEFT: Middle blocked (M: {M_min:.2f}m, {M_max_area*100:.1f}%), only left clear")
            return "STEER_LEFT", "LEFT", risk_meta
        
        elif right_clear:
            if verbose:
                print(f"→ STEER_RIGHT: Middle blocked (M: {M_min:.2f}m, {M_max_area*100:.1f}%), only right clear")
            return "STEER_RIGHT", "RIGHT", risk_meta
        
        else:
            # Both sides blocked too - choose lesser evil
            if left_threat < right_threat:
                if verbose:
                    print(f"→ STEER_LEFT: All blocked, left less threatening ({left_threat:.2f} < {right_threat:.2f})")
                return "STEER_LEFT", "LEFT", risk_meta
            else:
                if verbose:
                    print(f"→ STEER_RIGHT: All blocked, right less threatening ({right_threat:.2f} <= {left_threat:.2f})")
                return "STEER_RIGHT", "RIGHT", risk_meta

    # Middle clear - check sides for steering adjustments
    # Use weighted distances that account for object size
    left_needs_avoid = left_steer_dist and left_steer_dist < WARN_DISTANCE
    right_needs_avoid = right_steer_dist and right_steer_dist < WARN_DISTANCE

    if left_needs_avoid and not right_needs_avoid:
        if verbose:
            print(f"→ STEER_RIGHT: Left has close objects (weighted_dist={left_steer_dist:.2f}m, area={left_max_area*100:.1f}%)")
        return "STEER_RIGHT", "RIGHT", risk_meta
    
    if right_needs_avoid and not left_needs_avoid:
        if verbose:
            print(f"→ STEER_LEFT: Right has close objects (weighted_dist={right_steer_dist:.2f}m, area={right_max_area*100:.1f}%)")
        return "STEER_LEFT", "LEFT", risk_meta

    # Both sides have objects - use threat scores (which include size)
    if left_needs_avoid and right_needs_avoid:
        if left_threat > right_threat * 1.5:
            if verbose:
                print(f"→ STEER_RIGHT: Left significantly more threatening ({left_threat:.2f} > {right_threat:.2f})")
            return "STEER_RIGHT", "RIGHT", risk_meta
        elif right_threat > left_threat * 1.5:
            if verbose:
                print(f"→ STEER_LEFT: Right significantly more threatening ({right_threat:.2f} > {left_threat:.2f})")
            return "STEER_LEFT", "LEFT", risk_meta

    # Follow desired path if safe
    if desired_turn == "LEFT":
        if not left_needs_avoid or (left_threat <= right_threat * 1.3):
            if verbose:
                print(f"→ STEER_LEFT: Following path line (safe or acceptable risk)")
            return "STEER_LEFT", "LEFT", risk_meta
    elif desired_turn == "RIGHT":
        if not right_needs_avoid or (right_threat <= left_threat * 1.3):
            if verbose:
                print(f"→ STEER_RIGHT: Following path line (safe or acceptable risk)")
            return "STEER_RIGHT", "RIGHT", risk_meta

    # Path is clear
    if verbose:
        print(f"→ FORWARD: Clear path ahead (left_threat={left_threat:.2f}, right_threat={right_threat:.2f}, M_threat={M_threat:.2f})")
    return "FORWARD", last_turn, risk_meta


def compute_turn_and_confidence(action, risk_meta, path_bias):
    """Compute turn angle and confidence based on action and risk."""
    left_threat = float(risk_meta.get("left_threat", 0.0))
    right_threat = float(risk_meta.get("right_threat", 0.0))
    threat_sum = left_threat + right_threat

    if threat_sum > 0:
        threat_bias = (right_threat - left_threat) / threat_sum
    else:
        threat_bias = 0.0

    base_strength = max(abs(path_bias), abs(threat_bias))

    if action == "STEER_LEFT":
        strength = max(0.3, min(0.9, base_strength + 0.2))
        return -strength, strength
    if action == "STEER_RIGHT":
        strength = max(0.3, min(0.9, base_strength + 0.2))
        return strength, strength
    if action == "STOP":
        return 0.0, 1.0
    # FORWARD
    return 0.0, 0.5 if threat_sum == 0 else 0.6


def load_floor_calibration(calib_path="floor_calib.json"):
    """Load floor calibration and compute homography matrix."""
    with open(calib_path, "r") as f:
        data = json.load(f)

    img_pts = np.array(data["img_pts"], dtype=np.float32)
    width_m = float(data["width_m"])
    length_m = float(data["length_m"])

    world_pts = np.array([
        [0.0,     0.0],
        [width_m, 0.0],
        [width_m, length_m],
        [0.0,     length_m]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(img_pts, world_pts)
    if H is None:
        raise RuntimeError("Homography failed. Recalibrate with cleaner points.")
    
    far_right = tuple(map(int, data["img_pts"][2]))
    far_left = tuple(map(int, data["img_pts"][3]))
    return H, far_left, far_right


def pixel_to_world(H, px, py):
    """Convert pixel coordinates to world coordinates in meters."""
    p = np.array([[[px, py]]], dtype=np.float32)
    w = cv2.perspectiveTransform(p, H)[0][0]
    return float(w[0]), float(w[1])


def detect_floor_path_line(frame):
    """
    Detect the path line on the floor using color-based segmentation.
    Returns: (top_y, origin_point, target_point)
    """
    h, w, _ = frame.shape
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    blur = cv2.GaussianBlur(lab, (5, 5), 0)

    # Sample floor reference color from bottom-center
    sample_w = max(10, int(w * 0.06))
    sample_h = max(10, int(h * 0.06))
    x_center = w // 2
    x0 = max(0, x_center - sample_w // 2)
    x1 = min(w, x0 + sample_w)
    y0 = max(0, h - sample_h)
    y1 = h
    floor_ref = blur[y0:y1, x0:x1].mean(axis=(0, 1))

    # Compute color distance from floor reference
    diff = np.linalg.norm(blur.astype(np.float32) - floor_ref.astype(np.float32), axis=2)

    # Adaptive threshold
    sample_diff = diff[y0:y1, x0:x1]
    T = max(12.0, float(sample_diff.mean() + 2.5 * sample_diff.std()))

    floor_mask = (diff < T).astype(np.uint8) * 255

    # Morphological operations to clean mask
    k = max(3, (min(w, h) // 200) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find seed point at bottom-center
    seed_x = x_center
    seed_y = h - 2

    if floor_mask[seed_y, seed_x] == 0:
        found = False
        for r in range(1, 25):
            for dx in range(-r, r + 1):
                xx = np.clip(seed_x + dx, 0, w - 1)
                if floor_mask[seed_y, xx] != 0:
                    seed_x = int(xx)
                    found = True
                    break
            if found:
                break
        if not found:
            return None, (x_center, h - 1), (x_center, h - 1)

    # Flood fill to get connected floor component
    ff = floor_mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, (seed_x, seed_y), 128)

    component = (ff == 128).astype(np.uint8)
    ys, xs = np.where(component > 0)
    if len(xs) == 0:
        return None, (x_center, h - 1), (x_center, h - 1)

    # Find topmost point of floor region
    top_y = int(ys.min())
    band = (ys <= top_y + max(3, h // 80))
    top_x = int(xs[band].mean()) if band.any() else int(xs[ys.argmin()])

    origin = (x_center, h - 1)
    target = (top_x, top_y)
    return top_y, origin, target