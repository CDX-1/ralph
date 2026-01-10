"""
Simple Object Detection using YOLOv8-nano
This script performs object detection on images, videos, or webcam feed.
"""

from ultralytics import YOLO
import cv2

def detect_objects(source=0, save_results=True):
    """
    Perform object detection using YOLOv8-nano
    
    Args:
        source: Can be:
                - 0 for webcam
                - path to image file (e.g., 'image.jpg')
                - path to video file (e.g., 'video.mp4')
        save_results: Whether to save the detection results
    """
    # Load YOLOv8-nano model
    print("Loading YOLOv8-nano model...")
    model = YOLO('models/yolov8n.pt')  # n stands for nano
    
    # Perform detection
    print(f"Running detection on: {source}")
    results = model(source, stream=True)
    
    # Process results
    for result in results:
        # Get the annotated frame
        annotated_frame = result.plot()
        
        # Display the frame
        cv2.imshow('YOLOv8 Object Detection', annotated_frame)
        
        # Print detected objects
        if len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                print(f"Detected: {class_name} (confidence: {confidence:.2f})")
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Save results if specified
    if save_results:
        print("Saving results...")
        model.predict(source, save=True)
        print("Results saved to 'runs/detect/predict' folder")


def detect_image(image_path):
    """Quick function to detect objects in a single image"""
    model = YOLO('yolov8n.pt')
    results = model(image_path)
    
    # Display results
    for result in results:
        result.show()
        
        # Print detections
        print(f"\nDetected {len(result.boxes)} objects:")
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            print(f"  - {class_name}: {confidence:.2f}")


if __name__ == "__main__":
    # Example usage:
    
    # Option 1: Detect from webcam (press 'q' to quit)
    print("Starting webcam detection...")
    print("Press 'q' to quit")
    detect_objects(source=0)
    
    # Option 2: Detect from image file
    # detect_image('path/to/your/image.jpg')
    
    # Option 3: Detect from video file
    # detect_objects(source='path/to/your/video.mp4')
