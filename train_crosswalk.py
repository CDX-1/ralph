"""
Train YOLOv8 model on crosswalk detection dataset
"""

from ultralytics import YOLO
import os

def train_crosswalk_model():
    """
    Train YOLOv8-nano model on crosswalk dataset
    """
    print("=" * 60)
    print("Starting Crosswalk Detection Model Training")
    print("=" * 60)
    
    # Initialize YOLOv8-nano model
    print("\n1. Loading YOLOv8-nano base model...")
    model = YOLO('yolov8n.pt')  # Start with pretrained weights
    
    # Path to dataset YAML
    data_yaml = os.path.join('data', 'crosswalk', 'data.yaml')
    
    if not os.path.exists(data_yaml):
        print(f"\n❌ Error: Dataset file not found at {data_yaml}")
        return
    
    print(f"✓ Dataset config found: {data_yaml}")
    
    # Training configuration
    print("\n2. Starting training...")
    print("   - Base model: YOLOv8-nano")
    print("   - Dataset: Crosswalk Detection")
    print("   - Classes: 1 (crosswalk)")
    print("   - Epochs: 50")
    print("   - Image size: 640")
    print("   - Batch size: 16")
    print("\nThis may take several minutes depending on your hardware...")
    print("Press Ctrl+C to stop training early\n")
    
    try:
        # Train the model
        results = model.train(
            data=data_yaml,
            epochs=50,              # Number of training epochs
            imgsz=640,              # Image size
            batch=16,               # Batch size (adjust based on your GPU memory)
            name='crosswalk_model', # Name for this training run
            patience=10,            # Early stopping patience
            save=True,              # Save checkpoints
            device="cpu",               # Use GPU if available, otherwise CPU
            workers=4,              # Number of worker threads
            project='runs/train',   # Save location
            exist_ok=True,          # Overwrite existing
            plots=True,             # Generate training plots
            verbose=True            # Print verbose output
        )
        
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print("=" * 60)
        
        # Print results location
        save_dir = results.save_dir
        print(f"\n📁 Results saved to: {save_dir}")
        print(f"\n📊 Training metrics:")
        print(f"   - Best model: {save_dir}/weights/best.pt")
        print(f"   - Last model: {save_dir}/weights/last.pt")
        print(f"   - Training plots: {save_dir}/*.png")
        
        # Validate the model
        print("\n3. Validating model on test set...")
        metrics = model.val()
        
        print(f"\n📈 Validation Results:")
        print(f"   - mAP50: {metrics.box.map50:.4f}")
        print(f"   - mAP50-95: {metrics.box.map:.4f}")
        print(f"   - Precision: {metrics.box.mp:.4f}")
        print(f"   - Recall: {metrics.box.mr:.4f}")
        
        print("\n✓ Training pipeline completed!")
        print(f"\nTo use the trained model in main.py:")
        print(f"   Change: model = YOLO('models/yolov8n.pt')")
        print(f"   To:     model = YOLO('{save_dir}/weights/best.pt')")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Partial training results may be saved")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("\nTroubleshooting:")
        print("1. Check if dataset paths are correct in data.yaml")
        print("2. Ensure images and labels exist in train/valid/test folders")
        print("3. Reduce batch size if you get out-of-memory errors")


if __name__ == "__main__":
    train_crosswalk_model()
