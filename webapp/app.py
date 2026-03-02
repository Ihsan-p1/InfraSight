"""
InfraSight - Pothole Volumetric Analysis
Premium Dashboard with Multi-page Navigation, Batch Processing, and History.
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import yaml
import os
import json
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_segmentation import PotholeSegmenter
from src.models.depth_estimation import DepthEstimator
from src.core.calibration import Calibrator
from src.core.volumetric import VolumetricCalculator
from src.core.severity import SeverityClassifier
from src.core.repair_advisor import RepairAdvisor
from src.visualization.mesh_3d import Mesh3DVisualizer
from src.models.material_classifier import MaterialClassifier
from src.core.history_manager import HistoryManager
from src.core.report_generator import ReportGenerator
from src.utils.gps_utils import extract_gps

# --- Page Configuration ---
st.set_page_config(
    page_title="InfraSight Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Premium Glassmorphism) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #0f172a;
        color: #f8fafc;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }
    
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 4px;
    }
    
    .severity-badge {
        padding: 6px 12px;
        border-radius: 99px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Gradient Button */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px #3b82f6;
        opacity: 0.9;
    }
    
    /* Header Gradient */
    .premium-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    .sidebar .sidebar-content {
        background: #1e293b;
    }
    
    [data-testid="stMetricValue"] {
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# --- Global Initialization ---
# --- Path Resolution ---
ROOT_DIR = Path(__file__).parent.parent.absolute()
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

@st.cache_resource
def get_models(conf_threshold=0.25, iou_threshold=0.45):
    """Load models once"""
    with st.spinner("Initializing models..."):
        if not CONFIG_PATH.exists():
            st.error(f"Config not found at {CONFIG_PATH}")
            return None, None, None

        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        yolo_config = config['models']['yolo']
        # Priority weights
        yolo_weights = yolo_config.get('weights_path')
        if not yolo_weights or not (ROOT_DIR / yolo_weights).exists():
            yolo_weights = yolo_config.get('weights_fallback')
            
        full_weights_path = ROOT_DIR / yolo_weights
        if not full_weights_path.exists():
            st.error(f"Weights not found at {full_weights_path}")
            return None, None, None

        st.info(f"Loading YOLO weights: {full_weights_path.name}...")
        segmenter = PotholeSegmenter(str(full_weights_path), conf_threshold=conf_threshold)
        # Update IOU if supported (Ultralytics model has it in predict)
        segmenter.iou_threshold = iou_threshold
        
        st.info("Loading Depth Anything V2 (this may take 30s for the first run)...")
        depth_estimator = DepthEstimator(config['models']['depth']['model_name'])
        
        st.info("Loading Material Classifier...")
        material_classifier = MaterialClassifier(config_path=str(CONFIG_PATH))
        
        st.success("Models loaded successfully.")
        return segmenter, depth_estimator, material_classifier, config

history_mgr = HistoryManager()
report_gen = ReportGenerator()

# --- Helper Functions ---
def run_analysis(image_np, segmenter, depth_estimator, material_classifier, config):
    """Run full pipeline on single image"""
    # Detection
    seg_results = segmenter.detect(image_np, visualize=True)
    if not seg_results['detections']:
        return None
    
    potholes = [det for det in seg_results['detections'] if det.class_id == 0]
    ref_det = segmenter.get_largest_detection(seg_results['detections'], 1)
    
    if not potholes:
        return None
        
    # Depth
    depth_map = depth_estimator.predict(image_np)
    
    # Calibration & Volumetric
    ref_type = 'estimated'
    ref_area = 45.9 # default card
    ref_mask = None
    
    if ref_det:
        ref_type = Calibrator.detect_reference_type(ref_det.mask, ref_det.bbox)
        ref_area = Calibrator.get_reference_specs(ref_type)['area_cm2']
        ref_mask = ref_det.mask
        
    calc = VolumetricCalculator(config['volumetric']['calibration_constant'])
    
    results_list = []
    
    for pothole_det in potholes:
        # Volumetric
        p_mask = pothole_det.mask
        combined_ref_mask = ref_mask if ref_mask is not None else p_mask
        
        vol_res = calc.calculate_volume(
            p_mask, combined_ref_mask, pothole_det.bbox, depth_map, 
            ref_area, ref_type
        )
        
        # Material Classification (Crop pothole from raw image)
        x1, y1, x2, y2 = pothole_det.bbox
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(image_np.shape[1], int(x2)), min(image_np.shape[0], int(y2))
        pothole_crop = image_np[y1:y2, x1:x2]
        
        mat_res = None
        surface_type = "asphalt" # default
        if pothole_crop.size > 0:
            mat_res = material_classifier.predict(pothole_crop)
            # Use detected material if confidence is high enough
            if mat_res['confidence'] >= config.get('models', {}).get('material', {}).get('confidence_threshold', 0.6):
                surface_type = mat_res['class']
                
        # Severity & Repair
        sev_res = SeverityClassifier().classify(vol_res.avg_depth_cm, vol_res.area_cm2, vol_res.volume_cm3)
        rep_res = RepairAdvisor().recommend(
            vol_res.volume_cm3, vol_res.avg_depth_cm, vol_res.area_cm2, 
            severity_level=sev_res.level, surface_type=surface_type
        )
        
        results_list.append({
            'volumetric': vol_res,
            'severity': sev_res,
            'repair': rep_res,
            'surface_type': surface_type,
            'surface_conf': mat_res['confidence'] if mat_res else 0.0,
            'pothole_mask': p_mask
        })
        
    highest_severity_pothole = sorted(results_list, key=lambda x: x['severity'].score, reverse=True)[0]
    total_area = sum(p['volumetric'].area_cm2 for p in results_list)
    total_vol = sum(p['volumetric'].volume_cm3 for p in results_list)
    total_cost = sum(p['repair'].total_cost_idr for p in results_list)
    
    return {
        'annotated': seg_results['annotated_image'],
        'depth_viz': depth_estimator.visualize_depth(depth_map),
        'potholes': results_list,
        'summary': {
            'area_cm2': total_area,
            'volume_cm3': total_vol,
            'severity_level': highest_severity_pothole['severity'].level,
            'severity_score': highest_severity_pothole['severity'].score,
            'repair_method': highest_severity_pothole['repair'].method_id if len(results_list) == 1 else "Multiple",
            'repair_cost_idr': total_cost,
            'repair_material_kg': sum(p['repair'].material_kg for p in results_list)
        },
        'depth_raw': depth_map,
        'original_rgb': image_np
    }

# --- Page: Analysis ---
def page_analyze():
    st.markdown('<h1 class="premium-header">Analyze</h1>', unsafe_allow_html=True)
    st.markdown("Upload satu atau lebih foto untuk analisis tomografi lubang jalan.")
    
    # Threshold Controls
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        conf_t = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05, help="Low confidence = more detections but more noise")
    with col_t2:
        iou_t = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05, help="Intersection Over Union for overlapping boxes")

    segmenter, depth_estimator, material_classifier, config = get_models(conf_threshold=conf_t, iou_threshold=iou_t)
    
    files = st.file_uploader("Upload Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if files:
        if st.button("MULAI ANALISIS BATCH"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Temporary storage for batch results
            batch_results = []
            
            for i, file in enumerate(files):
                status_text.text(f"Processing {file.name} ({i+1}/{len(files)})...")
                
                # Load
                img = Image.open(file).convert('RGB')
                img_np = np.array(img)
                
                # Analyze
                try:
                    res = run_analysis(img_np, segmenter, depth_estimator, material_classifier, config)
                    
                    if res:
                        # Save results
                        save_dir = ROOT_DIR / "data" / "processed" / "analysis_results"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_path = save_dir / f"annotated_{ts}_{file.name}"
                        cv2.imwrite(str(img_path), cv2.cvtColor(res['annotated'], cv2.COLOR_RGB2BGR))
                        
                        # GPS extraction
                        temp_path = ROOT_DIR / f"tmp_exif_{file.name}"
                        with open(temp_path, "wb") as f: f.write(file.getbuffer())
                        lat, lon = extract_gps(temp_path)
                        if temp_path.exists(): os.remove(temp_path)
                        
                        db_data = {
                            'image_name': file.name,
                            'image_path': str(img_path),
                            'area_cm2': res['summary']['area_cm2'],
                            'avg_depth_cm': res['potholes'][0]['volumetric'].avg_depth_cm, # representative
                            'volume_cm3': res['summary']['volume_cm3'],
                            'severity_level': res['summary']['severity_level'],
                            'severity_score': res['summary']['severity_score'],
                            'repair_method': res['summary']['repair_method'],
                            'repair_cost_idr': res['summary']['repair_cost_idr'],
                            'repair_material_kg': res['summary']['repair_material_kg'],
                            'latitude': lat,
                            'longitude': lon
                        }
                        entry_id = history_mgr.save_analysis(db_data)
                        
                        # Result for display
                        res['db_id'] = entry_id
                        res['image_name'] = file.name
                        batch_results.append(res)
                    else:
                        st.warning(f"{file.name}: Tidak ada lubang yang terdeteksi (Conf > {conf_t})")
                
                except Exception as e:
                    st.error(f"{file.name}: Error saat memproses — {str(e)}")
                
                progress_bar.progress((i + 1) / len(files))
            
            if batch_results:
                status_text.success(f"Berhasil memproses {len(batch_results)}/{len(files)} gambar!")
                st.session_state['batch_results'] = batch_results
            else:
                status_text.error("Gagal memproses gambar. Pastikan gambar berisi lubang jalan yang jelas atau turunkan Confidence Threshold.")

    # Display results
    if 'batch_results' in st.session_state:
        for i, res in enumerate(st.session_state['batch_results']):
            num_holes = len(res['potholes'])
            with st.expander(f"Image Analysis: {res['image_name']} - {num_holes} Pothole(s) Detected ({res['summary']['severity_level']})", expanded=(i==0)):
                cols = st.columns([1, 1, 1])
                with cols[0]: st.image(res['annotated'], caption="Detection Visualization")

                # Detailed analysis tabs
                pothole_tabs = st.tabs([f"Lubang #{p+1}" for p in range(num_holes)])
                
                for p_idx, p_res in enumerate(res['potholes']):
                    with pothole_tabs[p_idx]:
                        tab_cols = st.columns(2)
                        with tab_cols[0]:
                            # 3D Visualizer (Textured Replica)
                            viz = Mesh3DVisualizer()
                            fig = viz.create_pothole_mesh_cropped(
                                res['depth_raw'], 
                                p_res['pothole_mask'], 
                                image_rgb=res['original_rgb']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab_cols[1]:
                            # Severity & Repair Analysis
                            sev = p_res['severity']
                            st.markdown(f'<div class="severity-badge" style="background:{sev.color}">SEVERITY: {sev.level}</div>', unsafe_allow_html=True)
                            st.write(f"**Severity Score:** {sev.score}/10")
                            
                            rep = p_res['repair']
                            st.markdown("---")
                            st.write("### Rekomendasi Perbaikan")
                            st.write(f"**Metode:** {rep.method}")
                            st.write(f"**Material Jalan:** {p_res['surface_type'].capitalize()} ({p_res['surface_conf']:.2f})")
                            st.metric("Estimasi Biaya", f"Rp {rep.total_cost_idr:,.0f}")
                            
                            # Material Breakdown
                            st.write("**Detail Bahan:**")
                            st.write(f"- Utama ({rep.material_name}): {rep.material_kg:.2f} kg")
                            st.write(f"- Sealant: {p_res['volumetric'].area_cm2 * 0.0001:.3f} L")
                            
                # Report Generation Section
                st.markdown("---")
                
                # Use a specific function to handle download to avoid closure issues in loops
                def prepare_pdf_download(report_res, idx):
                    report_data = {
                        'image_name': report_res['image_name'],
                        'area_cm2': report_res['summary']['area_cm2'],
                        'avg_depth_cm': sum(p['volumetric'].avg_depth_cm for p in report_res['potholes'])/len(report_res['potholes']),
                        'volume_cm3': report_res['summary']['volume_cm3'],
                        'severity_level': report_res['summary']['severity_level'],
                        'severity_score': report_res['summary']['severity_score'],
                        'repair_method': report_res['summary']['repair_method'],
                        'repair_cost_idr': report_res['summary']['repair_cost_idr'],
                        'repair_material_kg': report_res['summary']['repair_material_kg'],
                        'annotated_path': None
                    }
                    pdf_path = report_gen.generate_pdf_report(report_data)
                    with open(pdf_path, "rb") as f:
                        return f.read()

                pdf_bytes = prepare_pdf_download(res, i)
                st.download_button(
                    label=f"Download Full Analysis Report PDF (Image #{i+1})",
                    data=pdf_bytes,
                    file_name=f"Report_{res['image_name']}.pdf",
                    mime="application/pdf",
                    key=f"dl_btn_{i}"
                )

# --- Page: History ---
def page_history():
    st.markdown('<h1 class="premium-header">History</h1>', unsafe_allow_html=True)
    
    history = history_mgr.get_all_history()
    if not history:
        st.info("Belum ada riwayat analisis.")
        return
        
    df_data = []
    for h in history:
        df_data.append({
            "ID": h.id,
            "Date": h.timestamp.strftime("%Y-%m-%d %H:%M"),
            "Image": h.image_name,
            "Severity": h.severity_level,
            "Volume (cm3)": f"{h.volume_cm3:.1f}",
            "Cost (IDR)": f"{h.repair_cost_idr:,.0f}"
        })
    st.table(df_data)

# --- Page: Map ---
def page_map():
    st.markdown('<h1 class="premium-header">Live Map</h1>', unsafe_allow_html=True)
    
    history = history_mgr.get_all_history()
    valid_coords = [h for h in history if h.latitude is not None and h.longitude is not None]
    
    if not valid_coords:
        st.warning("Tidak ada data GPS dalam histori. Upload foto dengan metadata GPS untuk melihat peta.")
        # Default center (Jakarta)
        m = folium.Map(location=[-6.2088, 106.8456], zoom_start=12)
    else:
        # Center at first valid coordinate
        m = folium.Map(location=[valid_coords[0].latitude, valid_coords[0].longitude], zoom_start=15)
        
        for h in valid_coords:
            color = 'red' if h.severity_level == 'CRITICAL' else 'orange' if h.severity_level == 'HIGH' else 'blue'
            folium.Marker(
                [h.latitude, h.longitude],
                popup=f"ID: {h.id}\nSev: {h.severity_level}\nCost: Rp {h.repair_cost_idr:,.0f}",
                tooltip=h.image_name,
                icon=folium.Icon(color=color)
            ).add_to(m)
            
    st_folium(m, width=1200, height=600)

# --- Main Logic ---
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown('<h2 style="color:white">InfraSight</h2>', unsafe_allow_html=True)
        selection = st.radio("Navigation", ["Dashboard & Analyze", "Analysis History", "Road Damage Map"])
        
        st.markdown("---")
        st.info("**Instructions:** Upload multiple images of potholes with a card/coin reference. The system will auto-classify severity and calculate precise repair costs.")

    if selection == "Dashboard & Analyze":
        page_analyze()
    elif selection == "Analysis History":
        page_history()
    elif selection == "Road Damage Map":
        page_map()

if __name__ == "__main__":
    main()
