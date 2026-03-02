"""
Fine-Tune YOLOv8 for Multiple Pothole Detection
Optimized for higher density detection.
"""
from ultralytics import YOLO
import torch
from pathlib import Path

def train_multiple_pothole():
    print("=" * 60)
    print("FINE-TUNING FOR MULTIPLE POTHOLE DETECTION")
    print("=" * 60)
    
    # Use existing best model as starting point if available
    # Otherwise use yolov8m.pt
    base_model_path = 'models/weights/pothole_det/experiment_b_merged_yolov8m/weights/best.pt'
    if not Path(base_model_path).exists():
        print(f"Warning: Base model {base_model_path} not found. Using yolov8m.pt")
        base_model_path = 'yolov8m.pt'
    
    model = YOLO(base_model_path)
    data_yaml = 'data/processed/multiple_pothole/data.yaml'
    
    if not Path(data_yaml).exists():
        print(f"Error: Dataset YAML {data_yaml} not found. Run prepare_multiple_pothole_data.py first.")
        return

    print(f"Starting training on {data_yaml}...")
    
    results = model.train(
        data=data_yaml,
        epochs=5,
        imgsz=640,
        batch=8,
        device='0' if torch.cuda.is_available() else 'cpu',
        project='models/weights/multiple_pothole',
        name='multi_v1',
        
        # Hyperparameters for dense detection
        optimizer='AdamW',
        lr0=0.0005, # Slightly higher than basic fine-tune
        lrf=0.01,
        dropout=0.1,
        
        # Augmentation for diversity
        mosaic=1.0,
        mixup=0.1,
        
        # Performance
        workers=0,
        exist_ok=True
    )
    
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    train_multiple_pothole()
