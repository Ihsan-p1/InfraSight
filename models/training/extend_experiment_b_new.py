"""
Experiment B Extension: 20 Additional Epochs
Goal: Extend Exp B (30 eps) by 20 more epochs to reach 50 total.
Method: Start NEW training with Exp B weights (last.pt) as pretrained.

Note: Epoch counter will reset to 1/20, but model performance should start high.
We will stitch the result CSVs later for analysis.
"""
from ultralytics import YOLO
import torch

def train_experiment_b_extension():
    print("=" * 60)
    print("EXPERIMENT B EXTENSION: +20 EPOCHS")
    print("=" * 60)
    
    # Load weights from completed Exp B 
    # (Using '2' suffix folder as identified)
    model_path = 'runs/detect/models/weights/pothole_det/experiment_b_merged_yolov8m2/weights/last.pt'
    print(f"Loading pretrained weights: {model_path}")
    
    model = YOLO(model_path)
    
    print("\nStarting 20 additional epochs...")
    results = model.train(
        data="data/processed/rdd2022_yolo/data.yaml",
        epochs=20,  # +20 epochs
        project='models/weights/pothole_det',
        name='experiment_b_merged_yolov8m_ext',
        
        # Keep same settings
        batch=8,
        imgsz=640,
        device='0',
        close_mosaic=5, # Close earlier in extension
        optimizer='AdamW',
        lr0=0.001,      # Restart LR scheduler? Maybe lower? 
                        # Let's keep 0.001 to simulate continuation or 
                        # reduce slightly to 0.0005? 
                        # Stick to 0.001 to mimic standard scheduler behavior
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=0, # No warmup needed, already trained
        workers=0
    )
    
    print("\n" + "=" * 60)
    print("EXTENSION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    train_experiment_b_extension()
