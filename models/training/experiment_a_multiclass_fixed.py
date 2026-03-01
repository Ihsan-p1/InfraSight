"""
Experiment A: Multi-Class with Fixed Training Setup
Goal: Test if training setup (not task complexity) was the bottleneck

Changes from failed pilot:
- lr0: 0.01 → 0.001 (10x lower to prevent divergence)
- warmup_epochs: 3 → 10 (longer warmup)
- close_mosaic: 5 → 10 (later mosaic disable)

Exit Criteria (30 epochs):
- mAP@50 ≥ 0.25 → Training setup was the issue, continue multi-class
- mAP@50 < 0.25 → Task inherently too complex
"""
from ultralytics import YOLO
import torch


def verify_gpu():
    print("=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 60)


def train_experiment_a(
    data_yaml="data/processed/rdd2022_multiclass/data.yaml",
    epochs=30,
    batch_size=8,
    img_size=640,
    device="0"
):
    """Experiment A: Fixed training setup for multi-class"""
    
    if device == '0' and not torch.cuda.is_available():
        print("⚠️  GPU requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    print("\n🔧 Loading YOLOv8m pretrained weights...")
    model = YOLO('yolov8m.pt')
    
    print("\n" + "=" * 60)
    print("EXPERIMENT A: MULTI-CLASS WITH FIXED TRAINING")
    print("=" * 60)
    print("Changes from failed pilot:")
    print("  - Learning rate: 0.01 → 0.001 (10x lower)")
    print("  - Warmup: 3 → 10 epochs")
    print("  - Close mosaic: 5 → 10 epochs")
    print("\nExit Criteria (epoch 30):")
    print("  ✅ mAP ≥ 0.25 → Continue multi-class")
    print("  ❌ mAP < 0.25 → Task too complex")
    print("=" * 60 + "\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        
        # Output
        project='models/weights/pothole_det',
        name='experiment_a_multiclass_fixed',
        
        # Training settings
        patience=30,
        save=True,
        save_period=10,
        val=True,
        close_mosaic=10,  # Changed from 5
        
        # Simplified augmentation (same as pilot)
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        scale=0.9,
        flipud=0.0,
        fliplr=0.5,
        degrees=0.0,
        translate=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # FIXED: Lower LR + more warmup
        optimizer='AdamW',
        lr0=0.001,        # 10x lower!
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=10,  # Increased from 3
        
        # Performance
        workers=0,
        cache=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT A COMPLETE!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    verify_gpu()
    
    results = train_experiment_a(
        data_yaml="data/processed/rdd2022_multiclass/data.yaml",
        epochs=30,
        batch_size=8,
        img_size=640,
        device="0"
    )
    
    # Validate
    print("\n📊 Running final validation...")
    model = YOLO('models/weights/pothole_det/experiment_a_multiclass_fixed/weights/best.pt')
    metrics = model.val(data="data/processed/rdd2022_multiclass/data.yaml")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT A RESULTS")
    print("=" * 60)
    print(f"Best mAP@50:    {metrics.box.map50:.4f}")
    print(f"Best mAP@50-95: {metrics.box.map:.4f}")
    print(f"Ratio (50-95/50): {metrics.box.map / metrics.box.map50 * 100:.1f}%")
    print("=" * 60)
    
    # Evaluation
    final_map = metrics.box.map50
    print("\n" + "=" * 60)
    print("EXIT CRITERIA EVALUATION")
    print("=" * 60)
    if final_map >= 0.25:
        print("✅ PASSED! mAP ≥ 0.25")
        print("   Conclusion: Training setup WAS the bottleneck")
        print("   Recommendation: Continue multi-class to 150-200 epochs")
    else:
        print("❌ FAILED. mAP < 0.25")
        print("   Conclusion: Multi-class task inherently too complex")
        print("   Recommendation: Pivot to merged classes or simpler approach")
    print("=" * 60)
