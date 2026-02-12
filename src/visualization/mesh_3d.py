"""
3D Mesh visualization using Plotly (browser-compatible)
"""
import numpy as np
import plotly.graph_objects as go
from typing import Optional


class Mesh3DVisualizer:
    """Create interactive 3D mesh visualizations of potholes using Plotly"""
    
    @staticmethod
    def create_pothole_mesh(
        depth_map: np.ndarray,
        pothole_mask: Optional[np.ndarray] = None,
        title: str = "Pothole 3D Profile",
        colorscale: str = "Viridis"
    ) -> go.Figure:
        """
        Create interactive 3D surface plot of pothole
        
        Args:
            depth_map: Depth map (H, W) - will be inverted to show depression
            pothole_mask: Optional binary mask to show only pothole area
            title: Plot title
            colorscale: Plotly colorscale ('Viridis', 'Inferno', 'Jet', etc.)
            
        Returns:
            Plotly Figure object (interactive in Streamlit)
        """
        # Apply mask if provided
        if pothole_mask is not None:
            # Invert depth map and apply mask
            z_values = -depth_map * pothole_mask
            # Set non-pothole areas to NaN for transparency
            z_values = z_values.astype(float)
            z_values[pothole_mask == 0] = np.nan
        else:
            # Invert entire depth map to show depression
            z_values = -depth_map
        
        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            z=z_values,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="Depth (relative)")
        )])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Depth (relative)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)  # Good viewing angle
                )
            ),
            width=700,
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_pothole_mesh_cropped(
        depth_map: np.ndarray,
        pothole_mask: np.ndarray,
        padding: int = 20
    ) -> go.Figure:
        """
        Create 3D mesh zoomed to pothole region (cropped)
        
        Args:
            depth_map: Full depth map
            pothole_mask: Binary mask of pothole
            padding: Pixels to pad around pothole bbox
            
        Returns:
            Plotly Figure with cropped view
        """
        # Find pothole bounding box
        coords = np.where(pothole_mask == 1)
        if len(coords[0]) == 0:
            raise ValueError("Pothole mask is empty")
        
        y_min = max(0, coords[0].min() - padding)
        y_max = min(depth_map.shape[0], coords[0].max() + padding)
        x_min = max(0, coords[1].min() - padding)
        x_max = min(depth_map.shape[1], coords[1].max() + padding)
        
        # Crop
        depth_cropped = depth_map[y_min:y_max, x_min:x_max]
        mask_cropped = pothole_mask[y_min:y_max, x_min:x_max]
        
        # Create mesh
        return Mesh3DVisualizer.create_pothole_mesh(
            depth_cropped,
            mask_cropped,
            title="Pothole 3D Profile (Zoomed)",
            colorscale="Inferno"
        )
    
    @staticmethod
    def create_side_by_side_comparison(
        depth_map: np.ndarray,
        pothole_mask: np.ndarray,
        reference_mask: np.ndarray
    ) -> go.Figure:
        """
        Create side-by-side comparison of full scene and pothole closeup
        
        Args:
            depth_map: Depth map
            pothole_mask: Pothole mask
            reference_mask: Reference object mask
            
        Returns:
            Plotly Figure with subplots
        """
        from plotly.subplots import make_subplots
        
        # Create figure with 1 row, 2 columns
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Full Scene", "Pothole Closeup"),
            specs=[[{'type': 'surface'}, {'type': 'surface'}]]
        )
        
        # Full scene (with both masks highlighted)
        z_full = -depth_map.copy()
        z_full[pothole_mask == 1] -= 0.2  # Highlight pothole
        z_full[reference_mask == 1] += 0.2  # Highlight reference
        
        fig.add_trace(
            go.Surface(z=z_full, colorscale="Viridis", showscale=False),
            row=1, col=1
        )
        
        # Pothole closeup
        coords = np.where(pothole_mask == 1)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            depth_crop = depth_map[y_min:y_max, x_min:x_max]
            mask_crop = pothole_mask[y_min:y_max, x_min:x_max]
            
            z_crop = -depth_crop * mask_crop
            z_crop[mask_crop == 0] = np.nan
            
            fig.add_trace(
                go.Surface(z=z_crop, colorscale="Inferno", showscale=True),
                row=1, col=2
            )
        
        fig.update_layout(
            height=500,
            width=1200,
            title_text="3D Depth Analysis"
        )
        
        return fig


if __name__ == "__main__":
    # Example usage
    
    # Create simulated depth map
    h, w = 300, 400
    depth_map = np.random.rand(h, w) * 0.5 + 0.3
    
    # Add pothole depression
    y_center, x_center = 150, 200
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((y - y_center)**2 + (x - x_center)**2)
            if dist < 50:
                depth_map[y, x] += 0.3 * (1 - dist/50)  # Deeper in center
    
    # Create mask
    pothole_mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if np.sqrt((y - y_center)**2 + (x - x_center)**2) < 50:
                pothole_mask[y, x] = 1
    
    # Create 3D visualization
    visualizer = Mesh3DVisualizer()
    fig = visualizer.create_pothole_mesh(depth_map, pothole_mask)
    
    # Show (requires browser)
    fig.show()
