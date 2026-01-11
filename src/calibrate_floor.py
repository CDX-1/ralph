import cv2
import json
import numpy as np

WINDOW = "Floor Calibration"
POINTS = []

# You will type these in the OpenCV window
width_str = ""
length_str = ""
mode = "width"  # or "length"

def on_mouse(event, x, y, flags, param):
    global POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(POINTS) < 4:
            POINTS.append([x, y])
            print(f"Point {len(POINTS)}: ({x}, {y})")
        else:
            print("Already have 4 points. Press R to reset, S to save.")

def draw_ui(frame):
    global width_str, length_str, mode

    h, w, _ = frame.shape

    # draw clicked points
    for i, (x, y) in enumerate(POINTS, start=1):
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(frame, str(i), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if len(POINTS) == 4:
        p = np.array(POINTS, dtype=np.int32)
        cv2.polylines(frame, [p], isClosed=True, color=(0, 255, 0), thickness=2)

    # instructions
    lines = [
        "Click 4 floor corners in order: near-left, near-right, far-right, far-left",
        "Type WIDTH (meters), press TAB to switch to LENGTH. Backspace to edit.",
        "Keys: TAB switch field | R reset points | S save | Q quit",
        f"WIDTH (m): {width_str}" + ("  <" if mode == "width" else ""),
        f"LENGTH(m): {length_str}" + ("  <" if mode == "length" else ""),
        f"Points: {len(POINTS)}/4",
    ]

    y = 30
    for t in lines:
        cv2.putText(frame, t, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        y += 28

def parse_float(s):
    try:
        return float(s)
    except:
        return None

def main():
    global width_str, length_str, mode, POINTS

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0. Try a different index (1,2) or check permissions.")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        draw_ui(frame)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            POINTS = []
            print("Reset points.")

        elif key == 9:  # TAB
            mode = "length" if mode == "width" else "width"

        elif key == 8:  # Backspace
            if mode == "width":
                width_str = width_str[:-1]
            else:
                length_str = length_str[:-1]

        elif key == ord("s"):
            if len(POINTS) != 4:
                print("Need exactly 4 points before saving.")
                continue

            w_m = parse_float(width_str)
            l_m = parse_float(length_str)
            if w_m is None or l_m is None or w_m <= 0 or l_m <= 0:
                print("Enter valid WIDTH and LENGTH in meters before saving.")
                continue

            data = {
                "img_pts": POINTS,
                "width_m": w_m,
                "length_m": l_m,
                "order": ["near-left", "near-right", "far-right", "far-left"]
            }
            with open("floor_calib.json", "w") as f:
                json.dump(data, f, indent=2)
            print("Saved to floor_calib.json")

        # digits and dot
        elif (48 <= key <= 57) or key == ord("."):
            ch = chr(key)
            if mode == "width":
                width_str += ch
            else:
                length_str += ch

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()