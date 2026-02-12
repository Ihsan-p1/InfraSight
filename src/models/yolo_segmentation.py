"""
YOLOv8 Segmentation for pothole and reference object detection
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Detection result container"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray  # Binary mask


class PotholeSegmenter:
    """YOLO segmentation inference for pothole and reference object detection"""
    
    def __init__(self, weights_path: str, conf_threshold: float = 0.25):
        """
        Initialize YOLO segmentation model
        
        Args:
            weights_path: Path to trained weights (.pt file)
            conf_threshold: Confidence threshold for detections
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")
        
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.class_names = {0: 'pothole', 1: 'reference_object'}
    
    def detect(
        self,
        image: np.ndarray,
        visualize: bool = False
    ) -> Dict[str, any]:
        """
        Detect potholes and reference objects in image
        
        Args:
            image: RGB image (H, W, 3)
            visualize: If True, return annotated image
            
        Returns:
            {
                'detections': List[Detection],
                'pothole_masks': List[np.ndarray],
                'reference_masks': List[np.ndarray],
                'annotated_image': np.ndarray (if visualize=True)
            }
        """
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        detections = []
        pothole_masks = []
        reference_masks = []
        
        # Extract results
        if results.masks is not None:
            for i, (box, mask_data, cls) in enumerate(zip(
                results.boxes.xyxy,
                results.masks.data,
                results.boxes.cls
            )):
                class_id = int(cls.item())
                confidence = float(results.boxes.conf[i].item())
                
                # Convert box to integers
                x1, y1, x2, y2 = map(int, box.tolist())
                
                # Convert mask to binary numpy array (resize to original image size)
                mask = mask_data.cpu().numpy()
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                mask = (mask > 0.5).astype(np.uint8)
                
                detection = Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, 'unknown'),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    mask=mask
                )
                
                detections.append(detection)
                
                # Separate by class
                if class_id == 0:  # Pothole
                    pothole_masks.append(mask)
                elif class_id == 1:  # Reference object
                    reference_masks.append(mask)
        
        result = {
            'detections': detections,
            'pothole_masks': pothole_masks,
            'reference_masks': reference_masks
        }
        
        # Visualization
        if visualize:
            annotated = image.copy()
            
            for det in detections:
                # Draw mask
                color = (0, 255, 0) if det.class_id == 0 else (255, 0, 0)
                colored_mask = np.zeros_like(annotated)
                colored_mask[det.mask == 1] = color
                annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.4, 0)
                
                # Draw bounding box
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
            
            result['annotated_image'] = annotated
        
        return result
    
    def get_largest_detection(
        self,
        detections: List[Detection],
        class_id: int
    ) -> Optional[Detection]:
        """
        Get largest detection of specific class (by mask area)
        
        Args:
            detections: List of detections
            class_id: Class to filter (0=pothole, 1=reference_object)
            
        Returns:
            Largest detection or None
        """
        filtered = [d for d in detections if d.class_id == class_id]
        
        if not filtered:
            return None
        
        # Sort by mask area
        filtered.sort(key=lambda d: np.sum(d.mask), reverse=True)
        return filtered[0]


if __name__ == "__main__":
    # Example usage
    segmenter = PotholeSegmenter("models/weights/yolov8_seg.pt")
    
    # Load test image
    image = cv2.imread("test_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect
    results = segmenter.detect(image, visualize=True)
    
    print(f"Found {len(results['pothole_masks'])} potholes")
    print(f"Found {len(results['reference_masks'])} reference objects")
    
    # Show annotated image
    if 'annotated_image' in results:
        cv2.imshow("Detections", cv2.cvtColor(results['annotated_image'], cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
