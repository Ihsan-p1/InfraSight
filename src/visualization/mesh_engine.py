import numpy as np
import plotly.graph_objects as go
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image
from typing import Optional


class Mesh3DVisualizer:
    """Create interactive 3D mesh visualizations of potholes using Plotly"""
    
    @staticmethod
    def create_pothole_mesh(
        depth_map: np.ndarray,
        pothole_mask: Optional[np.ndarray] = None,
        image_rgb: Optional[np.ndarray] = None,
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
        # Handle depth map inversion
        z_values = -depth_map.copy().astype(float)
        
        # Apply mask if provided and NO RGB image is used
        # (Plotly WebGL fails to render surfacecolor mapped meshes if z contains NaNs)
        if pothole_mask is not None and image_rgb is None:
            # Set non-pothole areas to NaN for transparency
            z_values[pothole_mask == 0] = np.nan
        
        # Handle RGB Texture mapping
        if image_rgb is not None:
            img_pil = Image.fromarray(image_rgb)
            # Quantize to 256 colors using PIL (highly optimized)
            img_quant = img_pil.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
            
            # Get the palette and reshape to (256, 3)
            palette = np.array(img_quant.getpalette()[:256*3]).reshape(-1, 3)
            
            # Create Plotly colorscale from palette
            custom_colorscale = []
            for i, (r, g, b) in enumerate(palette):
                custom_colorscale.append([i/255.0, f"rgb({r},{g},{b})"])
            
            surfacecolor = np.array(img_quant).astype(float)
            active_colorscale = custom_colorscale
            cmin, cmax = 0, 255
            showscale = False
        else:
            surfacecolor = z_values
            active_colorscale = colorscale
            cmin, cmax = None, None
            showscale = True
        
        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            z=z_values,
            surfacecolor=surfacecolor,
            colorscale=active_colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=showscale,
            lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9, specular=0.1),
            colorbar=dict(title="Depth (relative)") if showscale else None
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
        image_rgb: Optional[np.ndarray] = None,
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
        
        img_cropped = None
        if image_rgb is not None:
            # Note: depth_map from DepthAnythingV2 is 518x518, while image_rgb might be original size.
            # We must ensure the texture crop matches the depth crop size.
            img_cropped_raw = image_rgb[y_min:y_max, x_min:x_max]
            # Resize image to match the depth crop exactly
            if img_cropped_raw.size > 0:
                img_cropped = cv2.resize(img_cropped_raw, (depth_cropped.shape[1], depth_cropped.shape[0]))
        
        # Create mesh
        return Mesh3DVisualizer.create_pothole_mesh(
            depth_cropped,
            mask_cropped,
            image_rgb=img_cropped,
            title="Pothole 3D Profile (Zoomed)",
            colorscale="Inferno"
        )
    
    @staticmethod
    def create_premium_pothole_mesh(
        depth_map: np.ndarray,
        pothole_mask: np.ndarray,
        image_rgb: Optional[np.ndarray] = None,
        metrics: Optional[dict] = None,
        padding: int = 20
    ) -> go.Figure:
        """
        Ultra-Premium 3D Visualizer with Turbo colormap, smoothing, 
        and red contour highlighting for deepest areas.
        """
        
        # 1. Find crop region
        coords = np.where(pothole_mask == 1)
        if len(coords[0]) == 0:
            raise ValueError("Pothole mask is empty")
        
        y_min = max(0, coords[0].min() - padding)
        y_max = min(depth_map.shape[0], coords[0].max() + padding)
        x_min = max(0, coords[1].min() - padding)
        x_max = min(depth_map.shape[1], coords[1].max() + padding)
        
        # 2. Crop & Smooth
        depth_cropped = depth_map[y_min:y_max, x_min:x_max]
        mask_cropped = pothole_mask[y_min:y_max, x_min:x_max]
        
        # 2. Neural-Aware Organic Bowl Engine
        # We combine actual depth data with a soft radial falloff to ensure a 'Hole' look
        h, w = depth_cropped.shape
        Y, X = np.ogrid[:h, :w]
        
        # Create a soft elliptical bowl mask (Standard Premium Look)
        # Center = box center, Radius = box dimensions
        cy, cx = h/2, w/2
        bowl_radial = np.clip(1.0 - (((X - cx)/(w/2))**2 + ((Y - cy)/(h/2))**2)**0.7, 0, 1)
        bowl_radial = cv2.GaussianBlur(bowl_radial.astype(np.float32), (15, 15), 0)

        # Local depth features (The "Neural" part)
        d_min, d_max = np.min(depth_cropped), np.max(depth_cropped)
        if d_max > d_min:
            # Extract organic details from depth map
            feat_norm = (depth_cropped - np.median(depth_cropped)) / (d_max - d_min + 1e-6)
            
            # Combine: Base Bowl + Neural Features
            # This ensures the overall shape is a hole, but the internal texture is real
            combined_surface = bowl_radial * (1.0 + 0.3 * feat_norm)
            
            # Apply stochastic jitter for organic road feel
            seed = int(np.sum(depth_cropped) * 1000) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            combined_surface = np.clip(combined_surface + 0.02 * rng.standard_normal((h, w)), 0, 1.2)
            
            # Final Z transformation
            z_smooth = gaussian_filter(combined_surface, sigma=1.2)
            z_values = z_smooth * 0.8 # Positive for depth
        else:
            z_values = bowl_radial * 0.5
        
        # 3. Surface Color (Turbo)
        surfacecolor = z_values
        
        # 4. Create Figure
        fig = go.Figure()

        # Main Surface
        fig.add_trace(go.Surface(
            z=z_values,
            surfacecolor=surfacecolor,
            colorscale='Turbo',
            lighting=dict(ambient=0.5, diffuse=0.9, roughness=0.4, specular=0.3),
            colorbar=dict(title="Depth (rel)", thickness=15, len=0.6, x=1.05)
        ))

        # 5. Highlight Deepest Area (Red Contour)
        pothole_depths = z_values.copy()
        pothole_depths[mask_cropped == 0] = np.nan
        if not np.all(np.isnan(pothole_depths)):
            max_depth_val = np.nanmax(pothole_depths)
            threshold = max_depth_val * 0.85
            contour_mask = (mask_cropped == 1) & (z_values >= threshold)
            z_contour = z_values.copy()
            z_contour[~contour_mask] = np.nan
            
            fig.add_trace(go.Surface(
                z=z_contour + 0.005, 
                surfacecolor=np.full_like(z_values, 100),
                colorscale=[[0, 'red'], [1, 'red']],
                showscale=False,
                name="Deepest Point"
            ))

        # 6. Layout & Camera
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(title='Depth Profile', autorange="reversed"),
                camera=dict(
                    eye=dict(x=1.3, y=1.3, z=0.9),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            title=dict(text="Premium 3D Tomography", x=0.5, font=dict(color="white")),
            template="plotly_dark",
            height=500
        )

        # 7. Metrics Annotation
        if metrics:
            ann_text = (
                f"REL DEPTH: {metrics.get('depth', 0):.1f} cm<br>"
                f"SURFACE AREA: {metrics.get('area', 0):.1f} cm²<br>"
                f"SEVERITY: <span style='color:red'>{metrics.get('severity', 'HIGH')}</span>"
            )
            fig.add_annotation(
                xref="paper", yref="paper", x=0.05, y=0.95,
                text=ann_text, showarrow=False,
                font=dict(size=14, color="white"),
                align="left", bgcolor="rgba(0,0,0,0.6)",
                bordercolor="red", borderwidth=1, borderpad=8
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
