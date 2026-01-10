import cv2

def find_cameras():
    available_cameras = []
    cv2.setLogLevel(0)
    
    for i in range(3):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
            cap.release()
        except:
            pass
    
    cv2.setLogLevel(3)
    return available_cameras

def get_camera_index():
    print("Detecting cameras...")
    available_cams = find_cameras()
    
    if len(available_cams) > 1:
        cam_index = available_cams[1]
        print(f"✓ Using camera index {cam_index} (found cameras: {available_cams})")
    elif len(available_cams) > 0:
        cam_index = available_cams[0]
        print(f"✓ Only one camera found at index {cam_index}")
    else:
        cam_index = 0
        print("⚠ No cameras detected, defaulting to index 0")
    
    return cam_index
