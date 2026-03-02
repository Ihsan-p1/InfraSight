"""
Fine-Tune with Local Indonesian Dataset
Uses the best model from the previous fine-tuning (pothole + reference object)
and fine-tunes it on the 'deteksi-lubang-jalan' dataset from Roboflow.
"""
from ultralytics import YOLO
import torch
from pathlib import Path


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


def finetune_local_dataset(
    pretrained_path="models/weights/pothole_det/best.pt",
    data_yaml="data/raw/indonesian_pothole/jalan-berlubang-8/data.yaml",
    epochs=30,
    batch_size=8,
    img_size=640,
    device="0",
):
    """
    Fine-tune model on the local Indonesian dataset to improve adaptation.
    """
    if device == '0' and not torch.cuda.is_available():
        print("⚠️ GPU not available. Falling back to CPU.")
        device = 'cpu'
    
    pretrained = Path(pretrained_path)
    if not pretrained.exists():
        print(f"❌ Base Model not found: {pretrained_path}")
        print("   Please run initial fine-tuning first, or point to an existing weight.")
        return None
        
    print(f"\n🔧 Loading pretrained weights: {pretrained_path}")
    model = YOLO(pretrained_path)
    
    if not Path(data_yaml).exists():
        print(f"❌ Dataset not found: {data_yaml}")
        return None
        
    print(f"\n{'='*60}")
    print("FINE-TUNING: INDONESIAN POTHOLE DATASET")
    print(f"{'='*60}")
    print(f"  Base Model:    {pretrained_path}")
    print(f"  Dataset:       {data_yaml}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch Size:    {batch_size}")
    print(f"  Learning Rate: 0.0001 (low for domain adaptation)")
    print(f"  Device:        {device}")
    print(f"{'='*60}\n")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        
        # Output
        project='models/weights/pothole_seg',
        name='indonesian_finetune',
        exist_ok=True,
        
        # Training settings
        patience=10,
        save=True,
        save_period=5,
        val=True,       # We pointed val to train in data.yaml
        close_mosaic=5,
        
        # Augmentation
        mosaic=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        
        # Optimization — LOW LR so we don't forget 'reference_object' easily
        optimizer='AdamW',
        lr0=0.0001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2,
        
        workers=0,
        cache=True,
        verbose=True,
    )
    
    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    
    best_path = f"models/weights/pothole_seg/indonesian_finetune/weights/best.pt"
    if Path(best_path).exists():
        print(f"\n✅ New weights saved at: {best_path}")
        print("To use this model, update config/config.yaml -> weights_path")
        
    return results


if __name__ == "__main__":
    verify_gpu()
    # We use our previous best.pt which learned both Potholes and Reference Objects
    finetune_local_dataset()
