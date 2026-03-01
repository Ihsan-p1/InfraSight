"""Debug test - Check if pretrained model detects anything at very low confidence"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from src.models.yolo_segmentation import PotholeSegmenter

# Test with very low confidence
seg = PotholeSegmenter(
    "models/weights/pretrained/yolov8m-pothole-seg.pt",
    conf_threshold=0.05
)

# Test multiple images
images_dir = Path("data/processed/rdd2022_yolo/images/val")
images = list(images_dir.glob("*.jpg"))[:20]

print(f"Testing {len(images)} images with conf=0.05")
print("=" * 60)

total = 0
for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = seg.detect(img_rgb)
    n = len(r['detections'])
    total += n
    if n > 0:
        confs = [f"{d.confidence:.3f}" for d in r['detections']]
        print(f"  FOUND: {img_path.name} -> {n} det(s) conf={confs}")

print(f"\nTotal: {total} detections out of {len(images)} images")

# Also try D40 dataset if exists
d40_dir = Path("data/processed/rdd2022_d40_only/images")
if d40_dir.exists():
    for sub in ["train", "val"]:
        sub_dir = d40_dir / sub
        if sub_dir.exists():
            d40_imgs = list(sub_dir.glob("*.jpg"))[:10]
            print(f"\nD40 {sub}: testing {len(d40_imgs)} images")
            d40_found = 0
            for img_path in d40_imgs:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                r = seg.detect(img_rgb)
                n = len(r['detections'])
                d40_found += n
                if n > 0:
                    confs = [f"{d.confidence:.3f}" for d in r['detections']]
                    print(f"  FOUND: {img_path.name} -> {n} det(s) conf={confs}")
            print(f"  Total: {d40_found} detections")
else:
    print(f"\nD40 dataset not found at {d40_dir}")

# Also check RDD2022 raw if available
raw_dir = Path("data/raw/RDD2022")
if raw_dir.exists():
    print(f"\nRDD2022 raw exists at {raw_dir}")
    for sub in raw_dir.iterdir():
        if sub.is_dir():
            print(f"  {sub.name}")
