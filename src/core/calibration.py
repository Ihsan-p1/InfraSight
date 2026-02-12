"""
Calibration utilities for pixel-to-real-world conversion
"""
import numpy as np
from typing import Tuple, Optional
import math


class Calibrator:
    """Convert pixel measurements to real-world units using reference object"""
    
    # Reference object specifications (in cm)
    CARD_SPECS = {
        'width': 8.5,
        'height': 5.4,
        'area_cm2': 45.9,  # 8.5 × 5.4
        'aspect_ratio': 8.5 / 5.4  # ~1.57
    }
    
    COIN_SPECS = {
        'diameter': 2.7,
        'area_cm2': 5.73,  # π × (1.35)²
        'aspect_ratio': 1.0
    }
    
    @staticmethod
    def detect_reference_type(
        mask: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> str:
        """
        Detect whether reference object is CARD or COIN based on aspect ratio
        
        Args:
            mask: Binary mask of reference object
            bbox: Bounding box (x1, y1, x2, y2) - optional for faster computation
            
        Returns:
            'card' or 'coin_rp500'
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
        else:
            # Calculate from mask
            coords = np.where(mask == 1)
            if len(coords[0]) == 0:
                return 'unknown'
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            width = x_max - x_min
            height = y_max - y_min
        
        aspect_ratio = max(width, height) / (min(width, height) + 1e-8)
        
        # Card aspect ratio ~1.57, Coin ~1.0
        if aspect_ratio > 1.3:  # Closer to 1.57
            return 'card'
        else:
            return 'coin_rp500'
    
    @staticmethod
    def get_reference_specs(ref_type: str) -> dict:
        """Get specifications for reference object type"""
        if ref_type == 'card':
            return Calibrator.CARD_SPECS
        elif ref_type == 'coin_rp500':
            return Calibrator.COIN_SPECS
        else:
            raise ValueError(f"Unknown reference type: {ref_type}")
    
    @staticmethod
    def calculate_pixels_per_cm2(
        mask: np.ndarray,
        ref_type: str
    ) -> float:
        """
        Calculate pixels per cm² ratio from reference object
        
        Args:
            mask: Binary mask of reference object
            ref_type: 'card' or 'coin_rp500'
            
        Returns:
            pixels_per_cm2: Conversion factor
        """
        specs = Calibrator.get_reference_specs(ref_type)
        
        # Count pixels in mask
        reference_pixels = np.sum(mask)
        
        if reference_pixels == 0:
            raise ValueError("Reference mask is empty")
        
        # Calculate conversion factor
        px_per_cm2 = reference_pixels / specs['area_cm2']
        
        return px_per_cm2
    
    @staticmethod
    def pixels_to_cm2(
        pixel_area: float,
        px_per_cm2: float
    ) -> float:
        """
        Convert pixel area to cm²
        
        Args:
            pixel_area: Area in pixels
            px_per_cm2: Conversion factor from calculate_pixels_per_cm2
            
        Returns:
            area_cm2: Area in square centimeters
        """
        return pixel_area / px_per_cm2
    
    @staticmethod
    def get_confidence_level(ref_type: str) -> str:
        """
        Get confidence level based on reference object type
        CARD = High (easier detection, larger)
        COIN = Low (harder detection, smaller)
        
        Args:
            ref_type: Reference object type
            
        Returns:
            'High' or 'Low'
        """
        return 'High' if ref_type == 'card' else 'Low'


if __name__ == "__main__":
    # Example usage
    
    # Simulated reference object mask (100x100 pixels)
    ref_mask = np.zeros((500, 500), dtype=np.uint8)
    ref_mask[100:200, 100:300] = 1  # Rectangle 100x200 pixels
    
    # Detect type
    ref_type = Calibrator.detect_reference_type(ref_mask)
    print(f"Detected reference type: {ref_type}")
    
    # Calculate calibration
    px_per_cm2 = Calibrator.calculate_pixels_per_cm2(ref_mask, ref_type)
    print(f"Pixels per cm²: {px_per_cm2:.2f}")
    
    # Convert pothole area
    pothole_pixels = 50000
    pothole_cm2 = Calibrator.pixels_to_cm2(pothole_pixels, px_per_cm2)
    print(f"Pothole area: {pothole_cm2:.2f} cm²")
