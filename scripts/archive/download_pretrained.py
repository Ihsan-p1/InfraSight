"""
Download Pretrained Pothole Segmentation Model
Downloads keremberke/yolov8m-pothole-segmentation from HuggingFace
and saves it locally for fine-tuning.
"""
import os
import sys
import shutil
from pathlib import Path


def download_pretrained_model():
    """Download and verify pretrained pothole segmentation model"""
    
    # Output directory
    output_dir = Path("models/weights/pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "yolov8m-pothole-seg.pt"
    
    if output_path.exists():
        print(f"✓ Model already exists at: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        print("  Delete the file to re-download.")
    else:
        print("📥 Downloading keremberke/yolov8m-pothole-segmentation...")
        print("   Source: HuggingFace Hub")
        print()
        
        # Download using huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("📦 Installing huggingface_hub...")
            os.system(f"{sys.executable} -m pip install huggingface_hub")
            from huggingface_hub import hf_hub_download
        
        # Download best.pt from the repo
        print("🔧 Downloading weights file (best.pt)...")
        downloaded_path = hf_hub_download(
            repo_id="keremberke/yolov8m-pothole-segmentation",
            filename="best.pt",
        )
        
        # Copy to our output directory
        shutil.copy2(downloaded_path, str(output_path))
        print(f"✓ Weights saved to: {output_path}")
    
    # Verify with ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("MODEL VERIFICATION")
    print(f"{'='*60}")
    
    model = YOLO(str(output_path))
    
    print(f"  Task:       {model.task}")
    print(f"  File size:  {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    if hasattr(model, 'names'):
        print(f"  Classes:    {model.names}")
    
    if model.task == 'segment':
        print(f"\n✅ Confirmed: Segmentation model")
    elif model.task == 'detect':
        print(f"\n⚠️  Model task is 'detect' (detection-only)")
        print("   Masks will use bbox fallback in PotholeSegmenter.")
    else:
        print(f"\n⚠️  Unknown task: {model.task}")
    
    # Quick inference test
    print(f"\n🧪 Running quick inference test...")
    try:
        import numpy as np
        
        # Create dummy image
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model.predict(dummy, conf=0.01, verbose=False)
        
        has_masks = results[0].masks is not None
        print(f"  Inference:  ✅ Success")
        print(f"  Has masks:  {'✅ Yes' if has_masks else '⚠️ No detections on dummy image (normal)'}")
    except Exception as e:
        print(f"  Inference:  ❌ Failed - {e}")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print(f"  1. Fine-tune: python models/training/finetune_pothole_seg.py")
    print(f"  2. Or use directly in config.yaml:")
    print(f"     weights_path: \"{output_path}\"")
    print(f"  3. Test dashboard: cd webapp && streamlit run app.py")
    print(f"{'='*60}")
    
    return str(output_path)


if __name__ == "__main__":
    download_pretrained_model()
