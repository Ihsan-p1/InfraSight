"""
Full Pipeline Test — Direct Image Processing
Runs complete InfraSight pipeline on dataset images:
Segmentation → Depth → Volume → Severity → Repair
Saves annotated results as images.
"""
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_segmentation import PotholeSegmenter
from src.models.depth_estimation import DepthEstimator
from src.core.volumetric import VolumetricCalculator
from src.core.severity import SeverityClassifier
from src.core.repair_advisor import RepairAdvisor
from src.visualization.mesh_3d import Mesh3DVisualizer


def run_full_pipeline(image_paths, config_path="config/config.yaml"):
    """Run full pipeline on list of images"""
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Find weights (priority: finetuned > pretrained > legacy)
    yolo_cfg = config['models']['yolo']
    for key in ['weights_path', 'weights_fallback', 'weights_legacy']:
        path = yolo_cfg.get(key, '')
        if path and Path(path).exists():
            weights = path
            break
    else:
        print("❌ No YOLO weights found!")
        return
    
    # Load models
    print(f"🔧 Loading YOLO: {weights}")
    segmenter = PotholeSegmenter(weights, conf_threshold=yolo_cfg.get('conf_threshold', 0.10))
    
    print(f"🔧 Loading Depth Anything V2...")
    depth_estimator = DepthEstimator(
        model_name=config['models']['depth']['model_name'],
        device=config['models']['depth'].get('device', 'cpu')
    )
    
    # Setup other modules
    calculator = VolumetricCalculator(config['volumetric']['calibration_constant'])
    severity_cls = SeverityClassifier()
    repair_adv = RepairAdvisor()
    visualizer = Mesh3DVisualizer()
    
    output_dir = Path("runs/pipeline_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"FULL PIPELINE TEST — {len(image_paths)} images")
    print(f"{'='*70}\n")
    
    for i, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        print(f"━━━ [{i+1}/{len(image_paths)}] {img_path.name} ━━━")
        
        # Load
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️ Cannot read image\n")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 1: Segmentation
        seg_results = segmenter.detect(image_rgb, visualize=True)
        pothole_det = segmenter.get_largest_detection(seg_results['detections'], class_id=0)
        
        if pothole_det is None:
            print(f"  ❌ No pothole detected\n")
            # Still save annotated image
            if 'annotated_image' in seg_results:
                out = cv2.cvtColor(seg_results['annotated_image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_dir / f"nodet_{img_path.name}"), out)
            continue
        
        print(f"  ✅ Pothole: conf={pothole_det.confidence:.3f}, "
              f"bbox={pothole_det.bbox}, mask_pixels={np.sum(pothole_det.mask)}")
        
        # Step 2: Depth map
        depth_map = depth_estimator.predict(image_rgb)
        print(f"  📏 Depth: range=[{depth_map.min():.3f}, {depth_map.max():.3f}]")
        
        # Step 3: Volume (no reference object — estimated mode)
        try:
            volumetric = calculator.calculate_volume(
                pothole_mask=pothole_det.mask,
                reference_mask=pothole_det.mask,
                pothole_bbox=pothole_det.bbox,
                depth_map=depth_map,
                reference_real_area=45.9,  # Estimated (card area placeholder)
                reference_type='estimated'
            )
            
            print(f"  📐 Area: {volumetric.area_cm2:.1f} cm², "
                  f"Depth: {volumetric.avg_depth_cm:.1f} cm, "
                  f"Volume: {volumetric.volume_cm3:.1f} cm³")
            
            # Step 4: Severity
            sev = severity_cls.classify(
                volumetric.avg_depth_cm,
                volumetric.area_cm2,
                volumetric.volume_cm3
            )
            print(f"  ⚠️ Severity: {sev.level} (score {sev.score}/10) — {sev.label_id}")
            
            # Step 5: Repair
            rep = repair_adv.recommend(
                volumetric.volume_cm3,
                volumetric.avg_depth_cm,
                volumetric.area_cm2,
                sev.level
            )
            print(f"  🔧 Repair: {rep.method_id} | {rep.material_kg}kg | "
                  f"Rp {rep.total_cost_idr:,.0f}")
            
        except Exception as e:
            print(f"  ⚠️ Volume calculation error: {e}")
            sev = None
        
        # Save results
        # Annotated image with severity label
        if 'annotated_image' in seg_results:
            annotated = seg_results['annotated_image'].copy()
            if sev:
                label = f"{sev.level} (score {sev.score})"
                cv2.putText(annotated, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            out = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"result_{img_path.name}"), out)
        
        # Depth map visualization
        depth_colored = depth_estimator.visualize_depth(depth_map, cv2.COLORMAP_INFERNO)
        depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"depth_{img_path.name}"), depth_bgr)
        
        # Side-by-side: original | annotated | depth
        h, w = image.shape[:2]
        depth_resized = cv2.resize(depth_bgr, (w, h))
        annotated_bgr = cv2.cvtColor(seg_results['annotated_image'], cv2.COLOR_RGB2BGR)
        comparison = np.hstack([image, annotated_bgr, depth_resized])
        cv2.imwrite(str(output_dir / f"compare_{img_path.name}"), comparison)
        
        print()
    
    print(f"{'='*70}")
    print(f"✅ Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Use images that are most likely to have potholes
    val_dir = Path("data/processed/rdd2022_yolo/images/val")
    
    # Pick specific images + random ones
    specific = [
        val_dir / "India_002321.jpg",  # Had 0.78 conf detection
        val_dir / "India_000010.jpg",   # Had detections at low conf
        val_dir / "India_000011.jpg",
        val_dir / "India_000158.jpg",
        val_dir / "India_000054.jpg",
    ]
    
    # Filter to existing files
    images = [p for p in specific if p.exists()]
    
    # Add a few random ones
    import random
    all_val = list(val_dir.glob("*.jpg"))
    extra = random.sample(all_val, min(5, len(all_val)))
    images.extend([p for p in extra if p not in images])
    
    run_full_pipeline(images[:8])
