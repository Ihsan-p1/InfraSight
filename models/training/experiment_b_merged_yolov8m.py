"""
Experiment B: YOLOv8m Merged 4-Class
Goal: Validate if YOLOv8m can scale from YOLOv8s on merged task

Baseline:
- YOLOv8s merged 4→1: mAP = 0.502 (proven)

Test:
- YOLOv8m merged 4→1: Expected mAP = 0.50 × 1.07 = 0.535

Exit Criteria (30 epochs):
- mAP@50 ≥ 0.35 → Scaling works, continue to 150 epochs
- mAP@50 < 0.35 → YOLOv8m overfits on 3k images
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


def train_experiment_b(
    data_yaml="data/processed/rdd2022_yolo/data.yaml",
    epochs=30,
    batch_size=8,
    img_size=640,
    device="0"
):
    """Experiment B: YOLOv8m on merged 4-class"""
    
    if device == '0' and not torch.cuda.is_available():
        print("⚠️  GPU requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    print("\n🔧 Loading YOLOv8m pretrained weights...")
    model = YOLO('yolov8m.pt')
    
    print("\n" + "=" * 60)
    print("EXPERIMENT B: YOLOv8m MERGED 4-CLASS")
    print("=" * 60)
    print("Testing scaling assumption:")
    print("  YOLOv8s merged: 0.502 (proven)")
    print("  YOLOv8m expected: 0.50 × 1.07 = 0.535")
    print("\nExit Criteria (epoch 30):")
    print("  ✅ mAP ≥ 0.35 → Scaling works")
    print("  ❌ mAP < 0.35 → Overfitting on 3k images")
    print("=" * 60 + "\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        
        # Output
        project='models/weights/pothole_det',
        name='experiment_b_merged_yolov8m',
        
        # Training settings (same as YOLOv8s baseline)
        patience=30,
        save=True,
        save_period=10,
        val=True,
        close_mosaic=10,
        
        # Simplified augmentation
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
        
        # Optimization (conservative LR for 3k images + 25M params)
        optimizer='AdamW',
        lr0=0.001,        # Lower LR to prevent overfit
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        
        # Performance
        workers=0,
        cache=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT B COMPLETE!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    verify_gpu()
    
    results = train_experiment_b(
        data_yaml="data/processed/rdd2022_yolo/data.yaml",
        epochs=30,
        batch_size=8,
        img_size=640,
        device="0"
    )
    
    # Validate
    print("\n📊 Running final validation...")
    model = YOLO('models/weights/pothole_det/experiment_b_merged_yolov8m/weights/best.pt')
    metrics = model.val(data="data/processed/rdd2022_yolo/data.yaml")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT B RESULTS")
    print("=" * 60)
    print(f"Best mAP@50:    {metrics.box.map50:.4f}")
    print(f"Best mAP@50-95: {metrics.box.map:.4f}")
    print("\nComparison:")
    print(f"  YOLOv8s baseline: 0.502")
    print(f"  YOLOv8m (this):   {metrics.box.map50:.3f}")
    print(f"  Scaling factor:   {metrics.box.map50 / 0.502:.2f}x")
    print("=" * 60)
    
    # Evaluation
    final_map = metrics.box.map50
    print("\n" + "=" * 60)
    print("EXIT CRITERIA EVALUATION")
    print("=" * 60)
    if final_map >= 0.35:
        print("✅ PASSED! mAP ≥ 0.35")
        print("   Conclusion: YOLOv8m scaling works on merged task")
        print("   Recommendation: Continue to 150 epochs, expect 0.60+")
    else:
        print("❌ FAILED. mAP < 0.35")
        print("   Conclusion: YOLOv8m overfits on 3k images")
        print("   Recommendation: Stick with YOLOv8s or get more data")
    print("=" * 60)
