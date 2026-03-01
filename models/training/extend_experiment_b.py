"""
Experiment B Extension: Epoch 30 -> 50
Goal: Collect more data points for valid curve fitting based on user feedback.

Original Exp B reached mAP 0.486 at epoch 30.
We need epoch 40 and 50 data points to determine true trajectory.
"""
from ultralytics import YOLO
import torch

def train_experiment_b_extension():
    print("=" * 60)
    print("EXPERIMENT B EXTENSION: EPOCH 30 -> 50")
    print("=" * 60)
    
    # Load previous checkpoint
    model_path = 'runs/detect/models/weights/pothole_det/experiment_b_merged_yolov8m2/weights/last.pt'
    print(f"Loading checkpoint: {model_path}")
    
    model = YOLO(model_path)
    
    print("\nResuming training to Epoch 50...")
    results = model.train(
        data="data/processed/rdd2022_yolo/data.yaml",
        epochs=50,  # Extend to 50
        resume=True,
        project='models/weights/pothole_det',
        name='experiment_b_merged_yolov8m',
        
        # Keep same settings
        batch=8,
        imgsz=640,
        device='0',
        close_mosaic=10,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        workers=0
    )
    
    print("\n" + "=" * 60)
    print("EXTENSION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    train_experiment_b_extension()
