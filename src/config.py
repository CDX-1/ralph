FRAME_W, FRAME_H = 640, 480
DETECT_EVERY_N_FRAMES = 1
CONF_THRESH = 0.45

RELEVANT_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    60: 'dining table',
}

OBSTACLE_CLASSES = {0, 1, 2, 3, 5, 7, 13, 56, 57, 58, 60}

LEFT_MAX = 1/3
RIGHT_MIN = 2/3

AREA_WARN = 0.030
AREA_STOP = 0.070
AREA_GROWTH_APPROACH = 0.008
