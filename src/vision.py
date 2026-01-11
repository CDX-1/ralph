import cv2
import json
import numpy as np

# Distance thresholds (in meters)
STOP_DISTANCE = 1.0  # Stop if object closer than this
WARN_DISTANCE = 2.0  # Warn/steer if object closer than this
FRONT_BLOCK_DISTANCE = 1.0
FRONT_CLEAR_DISTANCE = 1.6
FRONT_FAR_IGNORE_DISTANCE = 2.5


def calculate_risk_score(objects):
    """
    Calculate weighted risk score for a list of objects.
    Objects are tuples of (distance, box_area_normalized).
    Closer objects and larger boxes contribute more to risk.
    Returns: risk score (higher = more risky), minimum distance
    """
    if not objects:
        return 0.0, None

    distances = [obj[0] for obj in objects]
    min_dist = min(distances)

    # Use inverse distance squared - closer objects have much higher weight
    # Multiply by box area (normalized) - larger boxes contribute more
    # Cap distances at 0.1m to prevent division issues
    risk = sum((1.0 / max(d, 0.1) ** 2) * (1.0 + area) for d, area in objects)
    # The (1.0 + area) multiplier: area is normalized 0-1, so this gives 1x-2x multiplier
    # Larger boxes get proportionally more weight

    return risk, min_dist


def filter_objects_by_distance(objects, max_distance):
    return [obj for obj in objects if obj[0] <= max_distance]


def middle_blocked(middle_objects, threshold=FRONT_BLOCK_DISTANCE):
    if not middle_objects:
        return False
    return min(obj[0] for obj in middle_objects) <= threshold


def middle_clear(middle_objects, threshold=FRONT_CLEAR_DISTANCE):
    if not middle_objects:
        return True
    return min(obj[0] for obj in middle_objects) > threshold


def choose_avoid_turn(LL_objects, L_objects, M_objects, R_objects, RR_objects, last_turn):
    if not middle_blocked(M_objects):
        return None

    left_objects = filter_objects_by_distance(LL_objects + L_objects, FRONT_FAR_IGNORE_DISTANCE)
    right_objects = filter_objects_by_distance(R_objects + RR_objects, FRONT_FAR_IGNORE_DISTANCE)

    left_risk, _ = calculate_risk_score(left_objects)
    right_risk, _ = calculate_risk_score(right_objects)

    if left_risk == right_risk:
        return last_turn if last_turn in ("LEFT", "RIGHT") else "RIGHT"
    return "LEFT" if left_risk < right_risk else "RIGHT"


def get_side_risk_meta(LL_objects, L_objects, R_objects, RR_objects):
    left_risk, _ = calculate_risk_score(LL_objects + L_objects)
    right_risk, _ = calculate_risk_score(R_objects + RR_objects)
    return {
        "left_risk": left_risk,
        "right_risk": right_risk,
    }


def decide_action(LL_objects, L_objects, M_objects, R_objects, RR_objects, last_turn, desired_turn, verbose=True):
    """
    Decide action based on distance and bounding box size of objects in each region.
    Uses weighted risk scoring - closer objects and larger boxes have more impact.
    Args:
        LL_objects: list of (distance, box_area_normalized) tuples in left-left region
        L_objects: list of (distance, box_area_normalized) tuples in left region
        M_objects: list of (distance, box_area_normalized) tuples in middle region
        R_objects: list of (distance, box_area_normalized) tuples in right region
        RR_objects: list of (distance, box_area_normalized) tuples in right-right region
        last_turn: last turn direction for consistency
    Returns:
        (action, last_turn, risk_meta)
    """
    # Calculate risk scores and minimum distances for each region
    LL_risk, LL_min = calculate_risk_score(LL_objects)
    L_risk, L_min = calculate_risk_score(L_objects)
    M_risk, M_min = calculate_risk_score(M_objects)
    R_risk, R_min = calculate_risk_score(R_objects)
    RR_risk, RR_min = calculate_risk_score(RR_objects)

    # Aggregate left side: combine LL and L (risk is additive, min is minimum)
    left_risk = LL_risk + L_risk
    left_min = min([d for d in [LL_min, L_min] if d is not None]) if (LL_min is not None or L_min is not None) else None

    # Aggregate right side: combine R and RR
    right_risk = R_risk + RR_risk
    right_min = min([d for d in [R_min, RR_min] if d is not None]) if (R_min is not None or RR_min is not None) else None

    risk_meta = {
        "LL_risk": LL_risk,
        "L_risk": L_risk,
        "M_risk": M_risk,
        "R_risk": R_risk,
        "RR_risk": RR_risk,
        "left_risk": left_risk,
        "right_risk": right_risk,
        "left_min": left_min,
        "right_min": right_min,
        "M_min": M_min,
    }

    if verbose:
        # Print detailed debug info
        print("\n=== Decision Debug ===")
        print(f"Region risks: LL={LL_risk:.3f} L={L_risk:.3f} M={M_risk:.3f} R={R_risk:.3f} RR={RR_risk:.3f}")
        LL_str = f"{LL_min:.2f}" if LL_min is not None else "None"
        L_str = f"{L_min:.2f}" if L_min is not None else "None"
        M_str = f"{M_min:.2f}" if M_min is not None else "None"
        R_str = f"{R_min:.2f}" if R_min is not None else "None"
        RR_str = f"{RR_min:.2f}" if RR_min is not None else "None"
        print(f"Region mins: LL={LL_str:>6} L={L_str:>6} M={M_str:>6} R={R_str:>6} RR={RR_str:>6}")
        left_min_str = f"{left_min:.2f}" if left_min is not None else "None"
        right_min_str = f"{right_min:.2f}" if right_min is not None else "None"
        print(f"Aggregated: left_risk={left_risk:.3f} left_min={left_min_str} | right_risk={right_risk:.3f} right_min={right_min_str}")

    # Critical: Stop if anything is very close in middle
    if M_min is not None and M_min < STOP_DISTANCE:
        if verbose:
            print(f"→ STOP: Middle has close object (M_min={M_min:.2f}m < {STOP_DISTANCE}m)")
        return "STOP", last_turn, risk_meta

    # Stop if middle and both sides have close objects
    if (M_min is not None and M_min < STOP_DISTANCE and
            left_min is not None and left_min < STOP_DISTANCE and
            right_min is not None and right_min < STOP_DISTANCE):
        if verbose:
            print(f"→ STOP: All sides blocked (M={M_min:.2f}, L={left_min:.2f}, R={right_min:.2f} all < {STOP_DISTANCE}m)")
        return "STOP", last_turn, risk_meta

    # If middle has close objects, steer to safer side (lower risk)
    if M_min is not None and M_min < WARN_DISTANCE:
        # Compare left and right risk scores
        if left_risk == 0 and right_risk == 0:
            # Both sides clear - use last_turn for consistency
            if desired_turn in ("LEFT", "RIGHT"):
                action = "STEER_LEFT" if desired_turn == "LEFT" else "STEER_RIGHT"
                if verbose:
                    print(f"→ {action}: Middle blocked (M={M_min:.2f}m), both sides clear, using desired_turn={desired_turn}")
            else:
                action = "STEER_LEFT" if last_turn == "LEFT" else "STEER_RIGHT"
                if verbose:
                    print(f"→ {action}: Middle blocked (M={M_min:.2f}m), both sides clear, using last_turn={last_turn}")
            return action, last_turn, risk_meta
        if left_risk == 0:
            # Only left is clear
            if verbose:
                print(f"→ STEER_LEFT: Middle blocked (M={M_min:.2f}m), only left is clear (left_risk=0, right_risk={right_risk:.3f})")
            return "STEER_LEFT", "LEFT", risk_meta
        if right_risk == 0:
            # Only right is clear
            if verbose:
                print(f"→ STEER_RIGHT: Middle blocked (M={M_min:.2f}m), only right is clear (left_risk={left_risk:.3f}, right_risk=0)")
            return "STEER_RIGHT", "RIGHT", risk_meta
        # Both sides have objects - choose the side with lower risk
        if left_risk < right_risk:
            if verbose:
                print(f"→ STEER_LEFT: Middle blocked (M={M_min:.2f}m), left safer (left_risk={left_risk:.3f} < right_risk={right_risk:.3f})")
            return "STEER_LEFT", "LEFT", risk_meta
        if right_risk < left_risk:
            if verbose:
                print(f"→ STEER_RIGHT: Middle blocked (M={M_min:.2f}m), right safer (left_risk={left_risk:.3f} > right_risk={right_risk:.3f})")
            return "STEER_RIGHT", "RIGHT", risk_meta
        if desired_turn in ("LEFT", "RIGHT"):
            action = "STEER_LEFT" if desired_turn == "LEFT" else "STEER_RIGHT"
            if verbose:
                print(f"→ {action}: Middle blocked (M={M_min:.2f}m), risks tied, using desired_turn={desired_turn}")
            return action, desired_turn, risk_meta
        if verbose:
            print(f"→ STEER_RIGHT: Middle blocked (M={M_min:.2f}m), risks tied, defaulting right")
        return "STEER_RIGHT", "RIGHT", risk_meta

    # Middle is clear, check sides using risk scores
    # If one side has significantly higher risk, steer to the other
    left_close = left_min is not None and left_min < STOP_DISTANCE
    right_close = right_min is not None and right_min < STOP_DISTANCE

    if left_close and not right_close:
        if verbose:
            print(f"→ STEER_RIGHT: Left has close object (left_min={left_min:.2f}m < {STOP_DISTANCE}m), right clear")
        return "STEER_RIGHT", "RIGHT", risk_meta
    if right_close and not left_close:
        if verbose:
            print(f"→ STEER_LEFT: Right has close object (right_min={right_min:.2f}m < {STOP_DISTANCE}m), left clear")
        return "STEER_LEFT", "LEFT", risk_meta

    # Use risk score comparison for steering decisions
    # If one side has much higher risk (1.5x threshold), prefer the other
    if left_risk > 0 and right_risk > 0:
        if left_risk > right_risk * 1.5:  # Left is significantly riskier
            if verbose:
                print(f"→ STEER_RIGHT: Left significantly riskier (left_risk={left_risk:.3f} > {1.5}*right_risk={right_risk:.3f}*1.5={right_risk*1.5:.3f})")
            return "STEER_RIGHT", "RIGHT", risk_meta
        if right_risk > left_risk * 1.5:  # Right is significantly riskier
            if verbose:
                print(f"→ STEER_LEFT: Right significantly riskier (right_risk={right_risk:.3f} > {1.5}*left_risk={left_risk:.3f}*1.5={left_risk*1.5:.3f})")
            return "STEER_LEFT", "LEFT", risk_meta
    elif left_risk > 0 and right_risk == 0:
        # Only left has objects
        if verbose:
            print(f"→ STEER_RIGHT: Only left has objects (left_risk={left_risk:.3f}, right_risk=0)")
        return "STEER_RIGHT", "RIGHT", risk_meta
    elif right_risk > 0 and left_risk == 0:
        # Only right has objects
        if verbose:
            print(f"→ STEER_LEFT: Only right has objects (left_risk=0, right_risk={right_risk:.3f})")
        return "STEER_LEFT", "LEFT", risk_meta

    # Follow desired turn if it is safe
    if desired_turn == "LEFT":
        if left_risk == 0 or left_risk <= right_risk * 1.5:
            if verbose:
                print(f"→ STEER_LEFT: Following path line (left_risk={left_risk:.3f}, right_risk={right_risk:.3f})")
            return "STEER_LEFT", "LEFT", risk_meta
    elif desired_turn == "RIGHT":
        if right_risk == 0 or right_risk <= left_risk * 1.5:
            if verbose:
                print(f"→ STEER_RIGHT: Following path line (left_risk={left_risk:.3f}, right_risk={right_risk:.3f})")
            return "STEER_RIGHT", "RIGHT", risk_meta

    # All clear or objects are far away with low risk
    if verbose:
        print(f"→ FORWARD: All clear or low risk (left_risk={left_risk:.3f}, right_risk={right_risk:.3f}, M_risk={M_risk:.3f})")
    return "FORWARD", last_turn, risk_meta


def compute_turn_and_confidence(action, risk_meta, path_bias):
    left_risk = float(risk_meta.get("left_risk", 0.0))
    right_risk = float(risk_meta.get("right_risk", 0.0))
    risk_sum = left_risk + right_risk

    if risk_sum > 0:
        risk_bias = (right_risk - left_risk) / risk_sum
    else:
        risk_bias = 0.0

    base_strength = max(abs(path_bias), abs(risk_bias))

    if action == "STEER_LEFT":
        strength = max(0.2, base_strength)
        return -strength, strength
    if action == "STEER_RIGHT":
        strength = max(0.2, base_strength)
        return strength, strength
    if action == "STOP":
        return 0.0, 1.0
    # FORWARD
    return 0.0, 0.5 if risk_sum == 0 else 0.6


def load_floor_calibration(calib_path="floor_calib.json"):
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
    p = np.array([[[px, py]]], dtype=np.float32)
    w = cv2.perspectiveTransform(p, H)[0][0]
    return float(w[0]), float(w[1])


def detect_floor_path_line(frame):
    h, w, _ = frame.shape
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    blur = cv2.GaussianBlur(lab, (5, 5), 0)

    # Sample "floor reference" colour from a small patch at bottom-center
    sample_w = max(10, int(w * 0.06))
    sample_h = max(10, int(h * 0.06))
    x_center = w // 2
    x0 = max(0, x_center - sample_w // 2)
    x1 = min(w, x0 + sample_w)
    y0 = max(0, h - sample_h)
    y1 = h
    floor_ref = blur[y0:y1, x0:x1].mean(axis=(0, 1))

    # Compute per-pixel colour distance from floor_ref (in LAB space)
    diff = np.linalg.norm(blur.astype(np.float32) - floor_ref.astype(np.float32), axis=2)

    # Adaptive-ish threshold: base + some texture tolerance from the sample region
    sample_diff = diff[y0:y1, x0:x1]
    T = max(12.0, float(sample_diff.mean() + 2.5 * sample_diff.std()))

    # Floor mask = pixels that look like the floor colour
    floor_mask = (diff < T).astype(np.uint8) * 255

    # Clean mask a bit (reduces speckles and small gaps)
    k = max(3, (min(w, h) // 200) | 1)  # odd kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep ONLY the connected component that touches the bottom-center seed
    seed_x = x_center
    seed_y = h - 2

    if floor_mask[seed_y, seed_x] == 0:
        # If bottom-center isn't classified as floor (lighting change etc.),
        # fall back to a tiny search around it to find a nearby floor pixel.
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

    # Flood fill into a temporary mask (OpenCV needs a padded mask)
    ff = floor_mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, (seed_x, seed_y), 128)  # filled component becomes 128

    component = (ff == 128).astype(np.uint8)
    ys, xs = np.where(component > 0)
    if len(xs) == 0:
        return None, (x_center, h - 1), (x_center, h - 1)

    # Topmost point of THAT connected floor region
    top_y = int(ys.min())

    # Pick a stable x at that top row: mean x of component pixels near top_y
    band = (ys <= top_y + max(3, h // 80))  # small band near the top
    top_x = int(xs[band].mean()) if band.any() else int(xs[ys.argmin()])

    origin = (x_center, h - 1)
    target = (top_x, top_y)
    return top_y, origin, target
