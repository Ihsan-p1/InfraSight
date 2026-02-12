"""
Volumetric calculation using surface-vs-bottom heuristic approach
CRITICAL MODULE: Implements the core depth estimation and volume calculation
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VolumetricResult:
    """Container for volumetric measurement results"""
    area_cm2: float
    avg_depth_cm: float
    max_depth_cm: float
    volume_cm3: float
    confidence: str  # 'High' or 'Low'
    reference_type: str  # 'card' or 'coin_rp500'


class VolumetricCalculator:
    """
    Calculate pothole volume using Surface vs Bottom heuristic
    
    IMPORTANT: This uses RELATIVE depth (not metric), calibrated with empirical constant
    Expected accuracy: ±30% (acceptable for monocular depth estimation MVP)
    """
    
    def __init__(self, calibration_constant: float = 30.0):
        """
        Initialize calculator
        
        Args:
            calibration_constant: Empirical constant to convert normalized depth to cm
                                 Must be tuned with 3-5 manual ground truth measurements
                                 Default: 30.0 (multiply normalized depth by 30 to get cm)
        """
        self.calibration_constant = calibration_constant
    
    def calculate_volume(
        self,
        pothole_mask: np.ndarray,
        reference_mask: np.ndarray,
        pothole_bbox: Tuple[int, int, int, int],
        depth_map: np.ndarray,
        reference_real_area: float,
        reference_type: str
    ) -> VolumetricResult:
        """
        Calculate volumetric measurements using SURFACE vs BOTTOM approach
        
        REFINED ALGORITHM (Heuristic for Relative Depth):
        
        1. GROUND PLANE ESTIMATION (d_surface)
           - Combine reference object depth + healthy asphalt around pothole
           - This represents our "zero level" (flat road)
        
        2. POTHOLE BOTTOM ESTIMATION (d_bottom)
           - Use bottom 10% deepest pixels to avoid noise
        
        3. RELATIVE DEPTH DIFFERENCE
           - depth_diff_normalized = abs(d_bottom - d_surface)
        
        4. CALIBRATION TO REAL-WORLD
           - real_depth_cm = depth_diff_normalized × calibration_constant
        
        5. AREA CALCULATION
           - Convert pixel area to cm² using reference object
        
        6. VOLUME ESTIMATION
           - volume_cm³ = area_cm² × average_depth_cm
        
        Args:
            pothole_mask: Binary mask (H, W) of pothole
            reference_mask: Binary mask (H, W) of reference object
            pothole_bbox: Bounding box (x1, y1, x2, y2) of pothole
            depth_map: Relative depth map (H, W) - normalized [0, 1]
            reference_real_area: Real area of reference object in cm² (45.9 for card)
            reference_type: 'card' or 'coin_rp500'
            
        Returns:
            VolumetricResult with measurements
        """
        # Step 1: GROUND PLANE ESTIMATION
        d_surface = self._estimate_ground_plane(
            reference_mask,
            pothole_mask,
            pothole_bbox,
            depth_map
        )
        
        # Step 2: POTHOLE BOTTOM ESTIMATION
        d_bottom = self._estimate_pothole_bottom(
            pothole_mask,
            depth_map,
            percentile=10  # Use bottom 10% deepest pixels
        )
        
        # Step 3: RELATIVE DEPTH DIFFERENCE
        depth_diff_normalized = abs(d_bottom - d_surface)
        
        # Step 4: CALIBRATION TO REAL-WORLD
        # CRITICAL: This constant must be empirically tuned!
        real_depth_cm = depth_diff_normalized * self.calibration_constant
        
        # Calculate max depth (single deepest pixel)
        pothole_depths = depth_map[pothole_mask == 1]
        if len(pothole_depths) > 0:
            max_depth_normalized = abs(np.min(pothole_depths) - d_surface)
            max_depth_cm = max_depth_normalized * self.calibration_constant
        else:
            max_depth_cm = 0.0
        
        # Step 5: AREA CALCULATION
        pothole_pixels = np.sum(pothole_mask)
        reference_pixels = np.sum(reference_mask)
        
        if reference_pixels == 0:
            raise ValueError("Reference mask is empty")
        
        px_per_cm2 = reference_pixels / reference_real_area
        pothole_area_cm2 = pothole_pixels / px_per_cm2
        
        # Step 6: VOLUME ESTIMATION
        volume_cm3 = pothole_area_cm2 * real_depth_cm
        
        # Confidence based on reference type
        confidence = 'High' if reference_type == 'card' else 'Low'
        
        return VolumetricResult(
            area_cm2=pothole_area_cm2,
            avg_depth_cm=real_depth_cm,
            max_depth_cm=max_depth_cm,
            volume_cm3=volume_cm3,
            confidence=confidence,
            reference_type=reference_type
        )
    
    def _estimate_ground_plane(
        self,
        reference_mask: np.ndarray,
        pothole_mask: np.ndarray,
        pothole_bbox: Tuple[int, int, int, int],
        depth_map: np.ndarray
    ) -> float:
        """
        Estimate ground plane depth by combining:
        - Reference object depth
        - Healthy asphalt around pothole (inside bbox but outside pothole mask)
        
        Returns:
            Average ground plane depth value
        """
        # Get reference object depths
        ref_depths = depth_map[reference_mask == 1]
        
        # Get healthy asphalt depths
        x1, y1, x2, y2 = pothole_bbox
        bbox_mask = np.zeros_like(pothole_mask)
        bbox_mask[y1:y2, x1:x2] = 1
        
        # Healthy asphalt = inside bbox but outside pothole
        healthy_asphalt_mask = (bbox_mask == 1) & (pothole_mask == 0)
        asphalt_depths = depth_map[healthy_asphalt_mask]
        
        # Combine both sources
        all_surface_depths = np.concatenate([ref_depths, asphalt_depths])
        
        if len(all_surface_depths) == 0:
            # Fallback: use only reference
            if len(ref_depths) > 0:
                return np.mean(ref_depths)
            else:
                raise ValueError("Cannot estimate ground plane: no reference data")
        
        # Use median for robustness against outliers
        d_surface = np.median(all_surface_depths)
        
        return d_surface
    
    def _estimate_pothole_bottom(
        self,
        pothole_mask: np.ndarray,
        depth_map: np.ndarray,
        percentile: int = 10
    ) -> float:
        """
        Estimate pothole bottom depth using bottom percentile of deepest pixels
        
        Args:
            pothole_mask: Binary mask of pothole
            depth_map: Depth map
            percentile: Percentile to use (10 = bottom 10% deepest pixels)
            
        Returns:
            Average depth of bottom percentile
        """
        pothole_depths = depth_map[pothole_mask == 1]
        
        if len(pothole_depths) == 0:
            raise ValueError("Pothole mask is empty")
        
        # Sort depths (assuming lower values = deeper, CHECK YOUR MODEL!)
        # For Depth Anything V2: typically lower values = closer (shallower)
        # We want DEEPEST pixels, so we look for MIN values if inverted
        # OR MAX values if depth increases with distance
        
        # CRITICAL: Adjust based on your depth model's output convention!
        # For now, assume lower values = closer to camera = shallower
        # Higher values = farther = DEEPER into pothole
        # So we want the HIGHEST values (percentile 90-100)
        
        percentile_threshold = np.percentile(pothole_depths, 100 - percentile)
        deepest_pixels = pothole_depths[pothole_depths >= percentile_threshold]
        
        if len(deepest_pixels) == 0:
            return np.mean(pothole_depths)
        
        d_bottom = np.mean(deepest_pixels)
        
        return d_bottom
    
    def set_calibration_constant(self, new_constant: float):
        """
        Update calibration constant based on empirical measurements
        
        Workflow:
        1. Measure 3-5 potholes manually (tape measure)
        2. Run system on same potholes
        3. Calculate: constant = manual_depth_cm / normalized_depth
        4. Average the constants
        5. Set here
        
        Args:
            new_constant: New empirical calibration constant
        """
        self.calibration_constant = new_constant
        print(f"✓ Calibration constant updated to {new_constant:.2f}")


if __name__ == "__main__":
    # Example usage
    
    # Simulated data
    h, w = 480, 640
    
    # Create simulated pothole mask
    pothole_mask = np.zeros((h, w), dtype=np.uint8)
    pothole_mask[200:300, 250:400] = 1  # 100x150 pixels
    
    # Create simulated reference mask
    reference_mask = np.zeros((h, w), dtype=np.uint8)
    reference_mask[350:400, 100:200] = 1  # ~50x100 pixels
    
    # Create simulated depth map
    depth_map = np.random.rand(h, w) * 0.3 + 0.5  # [0.5, 0.8]
    # Make pothole deeper
    depth_map[pothole_mask == 1] += 0.2  # Deeper values
    
    # Pothole bbox
    bbox = (250, 200, 400, 300)
    
    # Calculate
    calculator = VolumetricCalculator(calibration_constant=30.0)
    result = calculator.calculate_volume(
        pothole_mask=pothole_mask,
        reference_mask=reference_mask,
        pothole_bbox=bbox,
        depth_map=depth_map,
        reference_real_area=45.9,  # Card
        reference_type='card'
    )
    
    print(f"Pothole Area: {result.area_cm2:.2f} cm²")
    print(f"Average Depth: {result.avg_depth_cm:.2f} cm")
    print(f"Max Depth: {result.max_depth_cm:.2f} cm")
    print(f"Volume: {result.volume_cm3:.2f} cm³")
    print(f"Confidence: {result.confidence}")
