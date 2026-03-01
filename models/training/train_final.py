"""
Final Training: YOLOv8m on Merged Pothole Dataset
- 200 epochs, same proven config as Experiment B
- Starting fresh to avoid LR scheduler restart artifacts
- Dataset: rdd2022_yolo (merged 4-class -> 1-class pothole)
- Estimated: 15-18 hours on RTX 3050
"""
from ultralytics import YOLO
import torch


def train_final():
    print("=" * 60)
    print("FINAL TRAINING: YOLOv8m 200 EPOCHS")
    print("=" * 60)
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    model = YOLO('yolov8m.pt')

    results = model.train(
        data="data/processed/rdd2022_yolo/data.yaml",
        epochs=200,
        batch=8,
        imgsz=640,
        device='0',

        # Output
        project='models/weights/pothole_det',
        name='yolov8m_final_200',

        # Same settings proven in Exp B
        close_mosaic=20,
        patience=50,          # stop if no improvement for 50 epochs
        save=True,
        save_period=25,       # checkpoint every 25 epochs
        val=True,

        # Augmentation (same as Exp B)
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

        # Optimizer (same as Exp B)
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,

        workers=0,
        cache=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Final validation
    model = YOLO('models/weights/pothole_det/yolov8m_final_200/weights/best.pt')
    metrics = model.val(data="data/processed/rdd2022_yolo/data.yaml")
    
    print(f"\nFinal mAP@50:    {metrics.box.map50:.4f}")
    print(f"Final mAP@50-95: {metrics.box.map:.4f}")
    
    if metrics.box.map50 >= 0.70:
        print("\n🎯 TARGET ACHIEVED! mAP ≥ 0.70")
    else:
        print(f"\n📊 Final result: {metrics.box.map50:.4f} (target: 0.70)")


if __name__ == "__main__":
    train_final()
