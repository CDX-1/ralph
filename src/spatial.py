import cv2
try:
    from .config import LEFT_MAX, RIGHT_MIN, AREA_WARN, AREA_STOP, AREA_GROWTH_APPROACH, OBSTACLE_CLASSES
except ImportError:
    from config import LEFT_MAX, RIGHT_MIN, AREA_WARN, AREA_STOP, AREA_GROWTH_APPROACH, OBSTACLE_CLASSES

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
        cy = (y1 + y2) / 2.0
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
    else:
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
        conf = o["conf"]
        region = o["region"]
        area_norm = o["area_norm"]
        approaching = o["approaching"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        label = f"{conf:.2f} {region} {area_norm:.3f}"
        if approaching:
            label += " APPR"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame
