<img src="https://i.imgur.com/M6R0kj1.png" width="800" height="400"> 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/-Raspberry_Pi-C51A4A?style=for-the-badge&logo=Raspberry-Pi)

# Ralph
Real-time AI-powered autonomous navigation robot with obstacle detection, depth estimation, and motor control for Raspberry Pi-based 2-wheel robots.

## Overview
Ralph is an intelligent autonomous navigation system that combines YOLOv8-nano object detection with MiDaS depth estimation to enable safe robot navigation. The system detects obstacles, analyzes their position and depth, provides voice-guided feedback, and autonomously controls a 2-wheel differential drive robot to avoid collisions.

## Features

- **Real-time Object Detection**: Uses YOLOv8-nano model for fast and accurate obstacle detection
- **Depth Estimation**: MiDaS depth estimation for precise distance measurement and wall detection
- **Spatial Analysis**: Analyzes obstacle positions (left, center, right) and distances in 3D space
- **Temporal Tracking**: Monitors approaching objects based on area growth over time
- **Autonomous Motor Control**: Controls 2-wheel differential drive motors with PWM speed control
- **Voice Feedback**: Text-to-speech navigation commands powered by ElevenLabs API
- **Multi-camera Support**: Automatic detection and selection of available cameras
- **Raspberry Pi Integration**: Designed for deployment on Raspberry Pi with GPIO motor control
- **Simulation Mode**: Test and develop without physical hardware

## Technologies

- **PyTorch**: Deep learning framework for model inference
- **YOLOv8-nano**: Lightweight object detection model from Ultralytics
- **MiDaS**: Intel's monocular depth estimation for 3D scene understanding
- **OpenCV**: Computer vision library for camera input and image processing
- **pygame**: Audio playback for voice commands
- **ElevenLabs API**: High-quality text-to-speech voice generation
- **RPi.GPIO**: Raspberry Pi GPIO control for motor drivers (L298N compatible)
- **Python 3.8+**: Core programming language

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CDX-1/ralph.git
cd ralph
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your ElevenLabs API key in `.env`:
```
ELEVENLABS_KEY=your_api_key_here
```

## Project Structure

```
ralph/
├── src/
│   ├── main.py              # Main application with motor control
│   ├── config.py            # Configuration constants
│   ├── camera.py            # Camera detection and setup
│   ├── audio.py             # Audio playback management
│   ├── spatial.py           # Spatial analysis and decision logic
│   ├── midas_depth.py       # MiDaS depth estimation
│   ├── motor_controller.py  # 2-wheel motor control (L298N driver)
│   └── __init__.py          # Package initialization
├── voice/
│   ├── voice.py             # ElevenLabs TTS generator
│   └── commands/            # Generated audio files
├── models/
│   └── yolov8n.pt           # YOLOv8-nano model
├── runs/                    # YOLO detection results
├── run_rpi.py               # Raspberry Pi launcher
├── requirements.txt         # Python dependencies
├── MIDAS_GUIDE.md           # MiDaS setup and calibration guide
└── README.md
```
on Development Machine (Simulation Mode)

```bash
python src/main.py
```

In simulation mode (without RPi.GPIO), the system will:
- Display motor commands in the console
- Show live video feed with detections
- Provide voice navigation commands
- Display depth map visualization

### Running on Raspberry Pi (Autonomous Robot Mode)

```bash
python run_rpi.py
```

In robot mode, the system will:
- Automatically detect available cameras
- Load YOLOv8-nano and MiDaS models
- Control 2-wheel motors via GPIO pins
- Navigate autonomously with obstacle avoidance
- Provide voice navigation feedback:
  - "Stop" - when close obstacles detected or wall ahead
  - "Turn Left/Right" - for steering around obstacles
  - "Go" - when path is clear
  - "Obstacle detected" - when obstacles in warning range

**Controls:**
- Press `q` to quit and stop motors
- Press `c` to calibrate MiDaS depth threshold
```

This trains a specialized model on the crosswalk dataset located in `data/crosswalk/`.

###Motor Control Configuration

### Default GPIO Pin Mapping (L298N Motor Driver)

**Left Motor:**
- GPIO 17: Forward direction
- GPIO 27: Backward direction
- GPIO 12: PWM speed control (Enable A)

**Right Motor:**
- GPIO 22: Forward direction
- GPIO 23: Backward direction
- GPIO 13: PWM speed control (Enable B)

### Motor Commands
- `forward()`: Move straight ahead
- `backward()`: Move in reverse
- `turn_left()`: Pivot left (left backward, right forward)
- `turn_right()`: Pivot right (left forward, right backward)
- `stop()`: Stop all motors

To customize pins, modify the `MotorController` initialization in `src/main.py`.

## Configuration

Key parameters can be adjusted in `src/config.py`:
Hardware Requirements

### For Development/Testing:
- Python 3.8 or higher
- Webcam or USB camera
- GPU recommended for faster inference (optional)

### For Autonomous Robot:
- Raspberry Pi 3B+ or 4 (recommended)
- USB camera or Raspberry Pi Camera Module
- L298N motor driver or compatible H-bridge
- 2x DC motors (differential drive)
- Power supply (separate for motors and Pi)
- Chassis with wheels

### Optional:
- ElevenLabs API key for voice generation (pre-generated audio files included)

## How It Works

1. **Vision System**: YOLOv8 detects objects in real-time while MiDaS estimates depth
2. **Spatial Analysis**: Objects are categorized by position (left/center/right) and distance
3. **Decision Logic**: System determines action (GO/STOP/TURN) based on obstacles and depth
4. **Motor Control**: Autonomous navigation commands are sent to motor controller
5. **Audio Feedback**: Voice commands provide human-readable status updates

## Acknowledgments

- YOLOv8 by Ultralytics
- MiDaS by Intel ISL
Use the calibration mode (press `c`) to find optimal threshold for your setup.

Key parameters can be adjusted in `main.py`:

- `CONF_THRESH`: Detection confidence threshold (default: 0.45)
- `OBSTACLE_CLASSES`: Object classes to treat as obstacles
- `AREA_WARN`: Distance threshold for warnings (default: 0.030)
- `AREA_STOP`: Distance threshold for stop command (default: 0.070)
- `DETECT_EVERY_N_FRAMES`: Frame skip for performance (default: 1)

## Requirements

- Python 3.8 or higher
- Webcam or external camera
- GPU recommended for faster inference (optional)
- ElevenLabs API key for voice generation

## License

This project uses the COCO Crosswalk Detection dataset under CC BY 4.0 license.

## Acknowledgments

- YOLOv8 by Ultralytics
- ElevenLabs for text-to-speech API
