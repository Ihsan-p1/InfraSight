"""
YOLO v8 Segmentation Training Script (Local GPU)
"""
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO


def verify_gpu():
    """Check GPU availability"""
    print("=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  WARNING: No GPU detected. Training will be slow on CPU.")
    print("=" * 60)
    print()


def train_pothole_segmentation(
    data_yaml: str = "data/processed/data.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0"
):
    """
    Train YOLOv8-Seg model for pothole and reference object segmentation
    
    Args:
        data_yaml: Path to data configuration YAML
        epochs: Number of training epochs
        batch_size: Batch size (reduce to 8 if GPU OOM)
        img_size: Input image size
        device: Device to use ('0' for GPU, 'cpu' for CPU)
    """
    print("\n" + "=" * 60)
    print("POTHOLE SEGMENTATION TRAINING")
    print("=" * 60)
    
    # Verify data file exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
    
    print(f"\n✓ Data config: {data_yaml}")
    
    # Check if using GPU
    if device != 'cpu' and not torch.cuda.is_available():
        print("⚠️  GPU requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    # Initialize model (YOLOv8-Medium Detection - best balance for mAP 0.7+)
    print("\n🔧 Loading YOLOv8m pretrained weights...")
    model = YOLO('yolov8m.pt')  # Medium model (25.9M params) for high accuracy
    
    # Training configuration
    print("\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Image Size: {img_size}")
    print(f"   Device: {device}")
    print()
    
    # Train with critical augmentations for small object detection
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        
        # Output configuration
        project='models/weights/pothole_det',
        name='yolov8m_d40_final',
        
        # Training settings
        patience=30,           # Early stopping patience (very patient for 200 epochs)
        save=True,             # Save checkpoints
        save_period=20,        # Save every 20 epochs
        val=True,              # Validate during training
        close_mosaic=20,       # Disable mosaic in last 20 epochs for better accuracy
        
        # Simplified augmentations (remove heavy ones that slow convergence)
        mosaic=1.0,            # Mosaic augmentation (critical for small objects)
        mixup=0.0,             # Removed - too confusing for pothole learning
        copy_paste=0.0,        # Removed - not needed for single class
        scale=0.9,             # Higher scale variation
        flipud=0.0,            # No vertical flip
        fliplr=0.5,            # 50% horizontal flip
        degrees=0.0,           # No rotation - roads are flat
        translate=0.0,         # No translation
        hsv_h=0.015,           # Light color augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Loss weights for imbalanced dataset
        cls=1.0,               # Class loss weight
        box=0.5,               # Box loss weight
        
        # Performance (Windows compatible - reduce workers)
        workers=0,             # Use 0 for Windows to avoid multiprocessing issues
        cache=True,            # Cache images in RAM for speed
        verbose=True,          # Detailed output
        
        # Optimization (back to default - faster convergence)
        optimizer='AdamW',     # Use AdamW optimizer
        lr0=0.01,             # Initial learning rate (default)
        lrf=0.01,             # Final learning rate
        momentum=0.937,        # SGD momentum/Adam beta1
        weight_decay=0.0005,   # Optimizer weight decay
        warmup_epochs=3        # Warmup epochs
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Print results
    print(f"\n✓ Best weights: {results.save_dir}/weights/best.pt")
    print(f"✓ Last weights: {results.save_dir}/weights/last.pt")
    
    # Validate on test/val set
    print("\n📊 Running validation...")
    metrics = model.val()
    
    print("\n" + "-" * 60)
    print("VALIDATION METRICS")
    print("-" * 60)
    
    if hasattr(metrics, 'box'):
        print(f"Box mAP@50:    {metrics.box.map50:.4f}")
        print(f"Box mAP@50-95: {metrics.box.map:.4f}")
    
    if hasattr(metrics, 'seg'):
        print(f"Mask mAP@50:    {metrics.seg.map50:.4f}")
        print(f"Mask mAP@50-95: {metrics.seg.map:.4f}")
    
    print("-" * 60)
    
    # Check if target achieved
    target_map = 0.60
    if hasattr(metrics, 'seg') and metrics.seg.map50 >= target_map:
        print(f"\n✅ TARGET ACHIEVED! mAP@50 ({metrics.seg.map50:.4f}) >= {target_map}")
    else:
        current = metrics.seg.map50 if hasattr(metrics, 'seg') else 0.0
        print(f"\n⚠️  Below target. mAP@50: {current:.4f} (target: {target_map})")
        print("   Consider: more training data, longer epochs, or data augmentation tuning")
    
    return results


def test_inference(
    weights_path: str = "models/weights/pothole_seg/yolov8n_seg_v1/weights/best.pt",
    test_image: str = None
):
    """
    Test model inference on a sample image
    
    Args:
        weights_path: Path to trained weights
        test_image: Path to test image (optional)
    """
    import cv2
    import numpy as np
    
    print("\n" + "=" * 60)
    print("TESTING INFERENCE")
    print("=" * 60)
    
    # Load model
    model = YOLO(weights_path)
    
    if test_image and Path(test_image).exists():
        # Test on provided image
        image = cv2.imread(test_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = model.predict(image, conf=0.25, verbose=False)[0]
        
        print(f"\n✓ Processed: {test_image}")
        print(f"   Detections: {len(results.boxes) if results.boxes is not None else 0}")
        
        if results.boxes is not None:
            for i, cls in enumerate(results.boxes.cls):
                class_name = "pothole" if int(cls) == 0 else "reference_object"
                conf = results.boxes.conf[i].item()
                print(f"   - {class_name}: {conf:.3f}")
    else:
        print("\nℹ️  No test image provided. Skipping inference test.")
    
    print("=" * 60)


if __name__ == "__main__":
    # Verify GPU
    verify_gpu()
    
    # Train model (YOLOv8m with D40 pothole dataset for mAP >= 0.7)
    results = train_pothole_segmentation(
        data_yaml="data/processed/rdd2022_d40_only/data.yaml",
        epochs=200,  # Long training for convergence
        batch_size=8,  # Windows compatible
        img_size=640,
        device="0"  # Use GPU
    )
    
    # Test inference (optional - provide test image path)
    # test_inference(
    #     weights_path="models/weights/pothole_seg/yolov8n_seg_v1/weights/best.pt",
    #     test_image="path/to/test_image.jpg"
    # )
