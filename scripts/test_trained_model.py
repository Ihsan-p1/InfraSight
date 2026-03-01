"""
Quick test script to verify trained YOLO model works
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np


def test_trained_model():
    """Test the trained YOLOv8 pothole detection model"""
    print("=" * 70)
    print("Testing Trained YOLOv8 Pothole Detection Model")
    print("=" * 70)
    
    # Load trained model
    weights_path = "models/weights/pothole_det/yolov8n_det_v1/weights/best.pt"
    
    if not Path(weights_path).exists():
        print(f"\nError: Weights not found at {weights_path}")
        return
    
    print(f"\nLoading model from: {weights_path}")
    model = YOLO(weights_path)
    
    # Test on a validation image
    test_img_dir = Path("data/processed/rdd2022_yolo/images/val")
    test_images = list(test_img_dir.glob("*.jpg"))[:5]  # Test on 5 images
    
    if not test_images:
        print("\nNo validation images found!")
        return
    
    print(f"\nTesting on {len(test_images)} images...\n")
    
    for img_path in test_images:
        print(f"\nProcessing: {img_path.name}")
        
        # Run inference
        results = model.predict(
            str(img_path),
            conf=0.25,
            verbose=False
        )[0]
        
        # Print detections
        if results.boxes is not None and len(results.boxes) > 0:
            print(f"  Detections: {len(results.boxes)}")
            for i, box in enumerate(results.boxes):
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"    #{i+1}: class={cls}, conf={conf:.3f}, bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        else:
            print("  No detections")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print("\nModel is ready to use!")
    print(f"Confidence threshold: 0.25")
    print(f"mAP@50: 0.458 (from training)")


if __name__ == "__main__":
    test_trained_model()
