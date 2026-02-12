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
    
    # Initialize model (YOLOv8-Nano Segmentation)
    print("\n🔧 Loading YOLOv8n-seg pretrained weights...")
    model = YOLO('yolov8n-seg.pt')  # Start from COCO pretrained
    
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
        project='models/weights/pothole_seg',
        name='yolov8n_seg_v1',
        
        # Training settings
        patience=10,           # Early stopping patience
        save=True,             # Save checkpoints
        save_period=10,        # Save every 10 epochs
        val=True,              # Validate during training
        
        # CRITICAL: Augmentations for class imbalance & small objects
        mosaic=1.0,            # Mosaic augmentation (combines 4 images)
        scale=0.5,             # Scale variation (0.5-1.5x)
        flipud=0.0,            # No vertical flip (roads don't flip)
        fliplr=0.5,            # 50% horizontal flip
        
        # Loss weights for imbalanced dataset
        cls=1.0,               # Class loss weight
        box=0.5,               # Box loss weight
        
        # Performance
        workers=8,             # Data loader workers
        cache=True,            # Cache images in RAM for speed
        verbose=True,          # Detailed output
        
        # Optimization
        optimizer='AdamW',     # Use AdamW optimizer
        lr0=0.01,             # Initial learning rate
        lrf=0.01,             # Final learning rate (lr0 * lrf)
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
    
    # Train model
    results = train_pothole_segmentation(
        data_yaml="data/processed/data.yaml",
        epochs=50,
        batch_size=16,  # Reduce to 8 if GPU OOM
        img_size=640,
        device="0"  # Use '0' for GPU, 'cpu' for CPU
    )
    
    # Test inference (optional - provide test image path)
    # test_inference(
    #     weights_path="models/weights/pothole_seg/yolov8n_seg_v1/weights/best.pt",
    #     test_image="path/to/test_image.jpg"
    # )
