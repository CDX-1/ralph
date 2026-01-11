<img src="https://i.imgur.com/M6R0kj1.png" width="800" height="400"> 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/-Raspberry_Pi-C51A4A?style=for-the-badge&logo=Raspberry-Pi)

# Ralph
Real-time AI-powered obstacle detection and navigation assistance system using computer vision and voice feedback.

## Overview
Ralph is an intelligent navigation assistant that uses YOLOv8-nano for real-time object detection and spatial analysis. The system detects obstacles, analyzes their position and proximity, and provides voice-guided navigation commands to help users safely navigate their environment.

## Features

- **Real-time Object Detection**: Uses YOLOv8-nano model for fast and accurate obstacle detection
- **Spatial Analysis**: Analyzes obstacle positions (left, center, right) and distances
- **Temporal Tracking**: Monitors approaching objects based on area growth
- **Voice Feedback**: Text-to-speech navigation commands powered by ElevenLabs API
- **Custom Dataset Training**: Includes crosswalk detection dataset for specialized training
- **Multi-camera Support**: Automatic detection and selection of available cameras

## Technologies

- **PyTorch**: Deep learning framework for model inference
- **YOLOv8-nano**: Lightweight object detection model from Ultralytics
- **OpenCV**: Computer vision library for camera input and image processing
- **pygame**: Audio playback for voice commands
- **ElevenLabs API**: High-quality text-to-speech voice generation
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
│   ├── main.py           # Main application entry point
│   ├── config.py         # Configuration constants
│   ├── camera.py         # Camera detection and setup
│   ├── audio.py          # Audio playback management
│   ├── spatial.py        # Spatial analysis and decision logic
│   ├── midas_depth.py    # MiDaS depth estimation
│   └── __init__.py       # Package initialization
├── voice/
│   ├── voice.py          # ElevenLabs TTS generator
│   └── commands/         # Generated audio files
├── models/               # YOLO model files
├── data/                 # Training datasets
├── run.py                # Application launcher
├── requirements.txt      # Python dependencies
└── README.md
```

## Usage

### Running the Main Detection System

```bash
python run.py
```

The system will:
- Automatically detect available cameras
- Load the YOLOv8-nano model
- Start real-time obstacle detection
- Provide voice navigation commands:
  - "Stop" - when close obstacles detected ahead
  - "Obstacle detected" - when obstacles in warning range
  - "Go" - when path is clear

Press `q` to quit.

### Training Custom Crosswalk Detection Model

```bash
python train_crosswalk.py
```

This trains a specialized model on the crosswalk dataset located in `data/crosswalk/`.

### Generating Voice Commands

```bash
python voice/voice.py
```

Generates custom voice command audio files using ElevenLabs API.

## Project Structure

```
ralph/
├── main.py                  # Main detection and navigation system
├── train_crosswalk.py       # Training script for crosswalk detection
├── voice/
│   ├── voice.py            # Voice command generation
│   └── commands/           # Generated audio files
├── data/
│   └── crosswalk/          # Crosswalk detection dataset
├── models/                 # Trained model weights
├── requirements.txt        # Python dependencies
└── README.md
```

## Configuration

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
- Crosswalk dataset from Roboflow Universe
- ElevenLabs for text-to-speech API
