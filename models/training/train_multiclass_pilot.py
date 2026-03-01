"""
Train YOLOv8m on Multi-Class RDD2022 - PILOT TEST (30 Epochs)
Goal: Validate if trend supports reaching mAP 0.70
"""
from ultralytics import YOLO
import torch


def verify_gpu():
    """Verify GPU availability"""
    print("=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 60)


def train_multiclass_pilot(
    data_yaml="data/processed/rdd2022_multiclass/data.yaml",
    epochs=30,
    batch_size=8,
    img_size=640,
    device="0"
):
    """
    Pilot test: 30 epochs to validate trend
    
    Exit Criteria:
    - Epoch 10: mAP ≥ 0.15 (good sign)
    - Epoch 20: mAP ≥ 0.25 (on track)
    - Epoch 30: mAP ≥ 0.30 (proceed to full training)
                 mAP < 0.30 (stop and re-evaluate)
    """
    
    # Check device
    if device == '0' and not torch.cuda.is_available():
        print("⚠️  GPU requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    # Initialize YOLOv8m
    print("\n🔧 Loading YOLOv8m pretrained weights...")
    model = YOLO('yolov8m.pt')
    
    # Training configuration - PILOT TEST
    print("\n🚀 Starting PILOT TEST (30 epochs)...")
    print("Exit Criteria:")
    print("  Epoch 10: mAP ≥ 0.15")
    print("  Epoch 20: mAP ≥ 0.25")
    print("  Epoch 30: mAP ≥ 0.30\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        
        # Output
        project='models/weights/pothole_det',
        name='yolov8m_multiclass_pilot',
        
        # Training settings
        patience=30,
        save=True,
        save_period=10,
        val=True,
        close_mosaic=5,
        
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
        
        # Optimization
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # Performance
        workers=0,
        cache=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("PILOT TEST COMPLETE!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    verify_gpu()
    
    print("\n" + "=" * 60)
    print("MULTI-CLASS PILOT TEST")
    print("=" * 60)
    print("Dataset: 9 classes (D00, D01, D10, D11, D20, D40, D43, D44, D50)")
    print("Images: ~15,745")
    print("Epochs: 30 (pilot)")
    print("Estimated time: 1.5-2 hours")
    print("=" * 60 + "\n")
    
    results = train_multiclass_pilot(
        data_yaml="data/processed/rdd2022_multiclass/data.yaml",
        epochs=30,
        batch_size=8,
        img_size=640,
        device="0"
    )
    
    # Validate best model
    print("\n📊 Running final validation...")
    model = YOLO('models/weights/pothole_det/yolov8m_multiclass_pilot/weights/best.pt')
    metrics = model.val(data="data/processed/rdd2022_multiclass/data.yaml")
    
    print("\n" + "=" * 60)
    print("PILOT TEST RESULTS")
    print("=" * 60)
    print(f"Best mAP@50:    {metrics.box.map50:.4f}")
    print(f"Best mAP@50-95: {metrics.box.map:.4f}")
    print("=" * 60)
    
    # Exit criteria evaluation
    final_map = metrics.box.map50
    print("\n" + "=" * 60)
    print("EXIT CRITERIA EVALUATION")
    print("=" * 60)
    if final_map >= 0.35:
        print("✅ EXCELLENT! mAP ≥ 0.35")
        print("   Recommendation: PROCEED with full 200 epochs")
        print("   Extrapolated mAP@100: ~0.70-0.75")
    elif final_map >= 0.30:
        print("✅ GOOD! mAP ≥ 0.30")
        print("   Recommendation: CONTINUE with caution")
        print("   Monitor closely at epoch 50")
    elif final_map >= 0.25:
        print("⚠️  BORDERLINE. mAP = 0.25-0.30")
        print("   Recommendation: Investigate hyperparameters")
        print("   Consider ablation study")
    else:
        print("❌ BELOW THRESHOLD. mAP < 0.25")
        print("   Recommendation: STOP multi-class approach")
        print("   Consider alternative strategies")
    print("=" * 60)
