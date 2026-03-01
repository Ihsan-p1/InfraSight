"""
Test Pretrained Model on Dataset Images
Runs the downloaded pretrained model on validation images
and saves annotated results for visual inspection.
"""
import cv2
import numpy as np
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_segmentation import PotholeSegmenter


def test_on_dataset(
    weights_path="models/weights/pretrained/yolov8m-pothole-seg.pt",
    images_dir="data/processed/rdd2022_yolo/images/val",
    output_dir="runs/pretrained_test",
    num_images=10,
    conf_threshold=0.25,
):
    """Test pretrained model on dataset images"""
    
    # Setup
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(weights_path).exists():
        print(f"❌ Weights not found: {weights_path}")
        print("   Run first: python scripts/download_pretrained.py")
        return
    
    # Load model
    print(f"🔧 Loading model: {weights_path}")
    segmenter = PotholeSegmenter(weights_path, conf_threshold=conf_threshold)
    
    # Get images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return
    
    # Pick random subset
    if len(image_files) > num_images:
        selected = random.sample(image_files, num_images)
    else:
        selected = image_files
    
    print(f"\n📷 Testing on {len(selected)} images from {images_dir}")
    print(f"   Output: {output_dir}")
    print(f"{'='*60}")
    
    total_detections = 0
    images_with_detection = 0
    
    for i, img_path in enumerate(selected):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️ Cannot read: {img_path.name}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = segmenter.detect(image_rgb, visualize=True)
        
        num_potholes = len(results['pothole_masks'])
        total_detections += num_potholes
        
        if num_potholes > 0:
            images_with_detection += 1
        
        # Get detection details
        det_info = []
        for det in results['detections']:
            det_info.append(f"{det.class_name}({det.confidence:.2f})")
        
        status = "✅" if num_potholes > 0 else "❌"
        print(f"  {status} [{i+1}/{len(selected)}] {img_path.name}: "
              f"{num_potholes} pothole(s) {' '.join(det_info)}")
        
        # Save annotated image
        if 'annotated_image' in results:
            annotated_bgr = cv2.cvtColor(results['annotated_image'], cv2.COLOR_RGB2BGR)
            out_path = output_dir / f"detected_{img_path.name}"
            cv2.imwrite(str(out_path), annotated_bgr)
        
        # Also save original with just pothole masks overlay for comparison
        if num_potholes > 0:
            mask_overlay = image.copy()
            for mask in results['pothole_masks']:
                colored = np.zeros_like(mask_overlay)
                colored[mask == 1] = [0, 255, 0]  # Green mask
                mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored, 0.4, 0)
            
            compare = np.hstack([image, mask_overlay])
            compare_path = output_dir / f"compare_{img_path.name}"
            cv2.imwrite(str(compare_path), compare)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Images tested:      {len(selected)}")
    print(f"  Images w/ detection: {images_with_detection}/{len(selected)} "
          f"({images_with_detection/len(selected)*100:.0f}%)")
    print(f"  Total detections:   {total_detections}")
    print(f"  Avg per image:      {total_detections/len(selected):.1f}")
    print(f"\n  Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_on_dataset(
        num_images=10,
        conf_threshold=0.25,
    )
