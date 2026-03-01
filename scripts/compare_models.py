"""Compare fine-tuned model vs pretrained model on validation images"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from src.models.yolo_segmentation import PotholeSegmenter

FINETUNED = "runs/detect/models/weights/pothole_seg/finetuned_v1/weights/best.pt"
PRETRAINED = "models/weights/pretrained/yolov8m-pothole-seg.pt"

val_dir = Path("data/processed/rdd2022_yolo/images/val")
images = sorted(val_dir.glob("*.jpg"))[:30]

print("=" * 80)
print("MODEL COMPARISON: Fine-tuned vs Pretrained")
print("=" * 80)

for label, weights, conf in [
    ("FINETUNED (best.pt)", FINETUNED, 0.25),
    ("PRETRAINED (keremberke)", PRETRAINED, 0.10),
]:
    if not Path(weights).exists():
        print(f"\n  SKIP {label}: weights not found")
        continue
        
    seg = PotholeSegmenter(weights, conf_threshold=conf)
    
    detected = 0
    total_conf = 0
    total_dets = 0
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r = seg.detect(img_rgb)
        n = len(r['detections'])
        if n > 0:
            detected += 1
            for d in r['detections']:
                total_conf += d.confidence
                total_dets += 1
    
    avg_conf = total_conf / total_dets if total_dets > 0 else 0
    
    print(f"\n  {label} (conf_threshold={conf})")
    print(f"    Images tested:     {len(images)}")
    print(f"    Images w/ detect:  {detected}/{len(images)} ({detected/len(images)*100:.0f}%)")
    print(f"    Total detections:  {total_dets}")
    print(f"    Avg confidence:    {avg_conf:.3f}")

print(f"\n{'='*80}")
