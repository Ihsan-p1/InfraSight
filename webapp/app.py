"""
InfraSight - Pothole Volumetric Analysis
Main Streamlit Web Application
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_segmentation import PotholeSegmenter
from src.models.depth_estimation import DepthEstimator
from src.core.calibration import Calibrator
from src.core.volumetric import VolumetricCalculator
from src.visualization.mesh_3d import Mesh3DVisualizer


# Page configuration
st.set_page_config(
    page_title="InfraSight - Pothole Analysis",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(config_path="config/config.yaml"):
    """Load models (cached to avoid reloading)"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load YOLO
    yolo_weights = config['models']['yolo']['weights_path']
    if not Path(yolo_weights).exists():
        st.error(f"❌ YOLO weights not found: {yolo_weights}")
        st.info("Please train the model first using: python models/training/train_yolo.py")
        return None, None
    
    segmenter = PotholeSegmenter(
        weights_path=yolo_weights,
        conf_threshold=config['models']['yolo']['conf_threshold']
    )
    
    # Load Depth Model
    depth_estimator = DepthEstimator(
        model_name=config['models']['depth']['model_name'],
        device=config['models']['depth']['device']
    )
    
    return segmenter, depth_estimator, config


def process_image(
    image: np.ndarray,
    segmenter: PotholeSegmenter,
    depth_estimator: DepthEstimator,
    config: dict
):
    """Process image through full pipeline"""
    
    # Step 1: Segmentation
    with st.spinner("🔍 Detecting potholes and reference objects..."):
        seg_results = segmenter.detect(image, visualize=True)
    
    # Check detections
    if len(seg_results['detections']) == 0:
        st.warning("⚠️ No potholes or reference objects detected!")
        return None
    
    # Get largest pothole and reference object
    pothole_det = segmenter.get_largest_detection(seg_results['detections'], class_id=0)
    reference_det = segmenter.get_largest_detection(seg_results['detections'], class_id=1)
    
    if pothole_det is None:
        st.warning("⚠️ No pothole detected!")
        return None
    
    if reference_det is None:
        st.warning("⚠️ No reference object detected! Please include a card or coin in the image.")
        return None
    
    # Step 2: Depth Estimation
    with st.spinner("📏 Generating depth map..."):
        depth_map = depth_estimator.predict(image)
    
    # Step 3: Calibration
    ref_type = Calibrator.detect_reference_type(
        reference_det.mask,
        reference_det.bbox
    )
    ref_specs = Calibrator.get_reference_specs(ref_type)
    
    # Step 4: Volume Calculation
    calculator = VolumetricCalculator(
        calibration_constant=config['volumetric']['calibration_constant']
    )
    
    with st.spinner("🧮 Calculating volume..."):
        volumetric_result = calculator.calculate_volume(
            pothole_mask=pothole_det.mask,
            reference_mask=reference_det.mask,
            pothole_bbox=pothole_det.bbox,
            depth_map=depth_map,
            reference_real_area=ref_specs['area_cm2'],
            reference_type=ref_type
        )
    
    # Step 5: Visualization
    visualizer = Mesh3DVisualizer()
    mesh_3d_fig = visualizer.create_pothole_mesh_cropped(
        depth_map,
        pothole_det.mask,
        padding=30
    )
    
    # Depth visualization
    depth_colored = depth_estimator.visualize_depth(depth_map, cv2.COLORMAP_INFERNO)
    
    return {
        'segmentation': seg_results['annotated_image'],
        'depth_map': depth_colored,
        'mesh_3d': mesh_3d_fig,
        'volumetric': volumetric_result,
        'reference_type': ref_type
    }


def main():
    # Header
    st.markdown('<p class="main-header">🕳️ InfraSight</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-top: -20px;">Pothole Volumetric Analysis System</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Reference object info
        st.info("""
        **📌 Reference Object Required**
        
        Please include ONE of the following in your image:
        - ✅ **Card** (ATM/KTP/SIM - 8.5×5.4cm) - **Recommended**
        - ⚠️ Coin (Rp500 - 2.7cm) - Lower accuracy
        
        Place it near the pothole on the road surface.
        """)
        
        st.markdown("---")
        
        # Upload image
        uploaded_file = st.file_uploader(
            "📤 Upload Pothole Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing a pothole and reference object"
        )
        
        st.markdown("---")
        
        st.caption("**InfraSight MVP** • Computer Vision for Road Maintenance")
    
    # Main content
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📷 Input Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("📊 Analysis")
            
            # Load models
            models = load_models()
            if models[0] is None:
                return
            
            segmenter, depth_estimator, config = models
            
            # Process image
            if st.button("🚀 Analyze Pothole", use_container_width=True):
                try:
                    results = process_image(
                        image_np,
                        segmenter,
                        depth_estimator,
                        config
                    )
                    
                    if results is not None:
                        # Store in session state
                        st.session_state['results'] = results
                        st.success("✅ Analysis complete!")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
        
        # Display results
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            st.markdown("---")
            st.subheader("📈 Results")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            v_result = results['volumetric']
            
            with col1:
                st.metric(
                    "📐 Area",
                    f"{v_result.area_cm2:.1f} cm²"
                )
            
            with col2:
                st.metric(
                    "📏 Avg Depth",
                    f"{v_result.avg_depth_cm:.1f} cm"
                )
            
            with col3:
                st.metric(
                    "📏 Max Depth",
                    f"{v_result.max_depth_cm:.1f} cm"
                )
            
            with col4:
                st.metric(
                    "🧊 Volume",
                    f"{v_result.volume_cm3:.1f} cm³",
                    help=f"Confidence: {v_result.confidence}"
                )
            
            # Confidence indicator
            conf_color = "🟢" if v_result.confidence == "High" else "🟡"
            ref_name = "Card" if results['reference_type'] == 'card' else "Coin (Rp500)"
            st.info(f"{conf_color} **Confidence: {v_result.confidence}** | Reference: {ref_name}")
            
            st.markdown("---")
            
            # Visualizations
            tab1, tab2, tab3 = st.tabs(["🎨 Segmentation", "🌡️ Depth Map", "🔮 3D Visualization"])
            
            with tab1:
                st.image(results['segmentation'], caption="Segmentation Results", use_container_width=True)
                st.caption("🟢 Green = Pothole | 🔴 Red = Reference Object")
            
            with tab2:
                st.image(results['depth_map'], caption="Depth Map (Inferno Colormap)", use_container_width=True)
                st.caption("🟣 Purple/Dark = Shallow | 🔴 Red/Bright = Deep")
            
            with tab3:
                st.plotly_chart(results['mesh_3d'], use_container_width=True)
                st.caption("Interactive 3D profile - drag to rotate, scroll to zoom")
            
            # Download results
            st.markdown("---")
            if st.button("💾 Export Results (JSON)"):
                import json
                export_data = {
                    'area_cm2': float(v_result.area_cm2),
                    'avg_depth_cm': float(v_result.avg_depth_cm),
                    'max_depth_cm': float(v_result.max_depth_cm),
                    'volume_cm3': float(v_result.volume_cm3),
                    'confidence': v_result.confidence,
                    'reference_type': results['reference_type']
                }
                
                st.download_button(
                    "📥 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name="pothole_analysis.json",
                    mime="application/json"
                )
    
    else:
        # Welcome screen
        st.info("""
        ### 👋 Welcome to InfraSight!
        
        **How to use:**
        1. Take a photo of a pothole from ~1-2 meters height
        2. Place a **card** (ATM/KTP) or coin near the pothole
        3. Upload the image using the sidebar
        4. Click "Analyze Pothole" to get volumetric measurements
        
        **Features:**
        - 🎯 Instance segmentation (YOLOv8)
        - 📏 Depth estimation (Depth Anything V2)
        - 🧮 Volume calculation (Surface vs Bottom heuristic)
        - 🔮 Interactive 3D visualization
        """)
        
        st.image("https://via.placeholder.com/800x400.png?text=Upload+Image+to+Start", use_container_width=True)


if __name__ == "__main__":
    main()
