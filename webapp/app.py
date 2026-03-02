"""
InfraSight — Pothole Volumetric Analysis
Premium Dashboard: Multi-page Navigation, Batch Processing, History & Map.
"""
import os
import sys
import yaml
import json
import cv2
import numpy as np
import pandas as pd
import folium
import streamlit as st

from PIL import Image
from pathlib import Path
from datetime import datetime
from streamlit_folium import st_folium

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent.absolute()
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
sys.path.insert(0, str(ROOT_DIR))

from src.models.yolo_segmentation  import PotholeSegmenter
from src.models.depth_estimation    import DepthEstimator
from src.models.material_classifier import MaterialClassifier
from src.core.calibration           import Calibrator
from src.core.volumetric            import VolumetricCalculator
from src.core.severity              import SeverityClassifier
from src.core.repair_advisor        import RepairAdvisor
from src.core.history_manager       import HistoryManager
from src.core.report_generator      import ReportGenerator
from src.visualization.mesh_engine  import Mesh3DVisualizer
from src.utils.gps_utils            import extract_gps
from src.utils.logger               import setup_logger

logger = setup_logger("WebApp")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InfraSight Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at top right, #1e293b, #0f172a);
    color: #f8fafc;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
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
    padding: 6px 14px;
    border-radius: 99px;
    font-weight: 700;
    font-size: 0.8rem;
    display: inline-block;
    letter-spacing: 0.04em;
}

.premium-header {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(to right, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.stButton > button {
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

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px -10px #3b82f6;
    opacity: 0.9;
}

[data-testid="stMetricValue"] { font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ── Singletons ─────────────────────────────────────────────────────────────────
history_mgr = HistoryManager()
report_gen  = ReportGenerator()


# ── Model loader (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def get_models(conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    with st.spinner("Initializing AI models…"):
        if not CONFIG_PATH.exists():
            st.error(f"Config not found: {CONFIG_PATH}")
            return None, None, None, None

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        yolo_cfg = config["models"]["yolo"]
        weights  = yolo_cfg.get("weights_path", "")
        if not weights or not (ROOT_DIR / weights).exists():
            weights = yolo_cfg.get("weights_fallback", "")

        full_path = ROOT_DIR / weights
        if not full_path.exists():
            st.error(f"YOLO weights not found: {full_path}")
            return None, None, None, None

        logger.info(f"Loading YOLO — {full_path.name}...")
        segmenter = PotholeSegmenter(str(full_path), conf_threshold=conf_threshold)
        segmenter.iou_threshold = iou_threshold

        logger.info("Loading Depth Anything V2 (first run may take ~30 s)...")
        depth_est = DepthEstimator(config["models"]["depth"]["model_name"])

        logger.info("Loading Material Classifier...")
        mat_clf = MaterialClassifier(config_path=str(CONFIG_PATH))

        return segmenter, depth_est, mat_clf, config


# ── Core analysis pipeline ─────────────────────────────────────────────────────
def run_analysis(image_np, segmenter, depth_estimator, material_classifier, config):
    """Run full pipeline on one image. Returns result dict or None."""
    seg_res  = segmenter.detect(image_np, visualize=True)
    if not seg_res["detections"]:
        return None

    potholes = [d for d in seg_res["detections"] if d.class_id == 0]
    ref_det  = segmenter.get_largest_detection(seg_res["detections"], 1)
    if not potholes:
        return None

    depth_map = depth_estimator.predict(image_np)

    # Calibration
    ref_type = "estimated"
    ref_area = 45.9          # default: standard card
    ref_mask = None
    if ref_det:
        ref_type = Calibrator.detect_reference_type(ref_det.mask, ref_det.bbox)
        ref_area = Calibrator.get_reference_specs(ref_type)["area_cm2"]
        ref_mask = ref_det.mask

    calc = VolumetricCalculator(config["volumetric"]["calibration_constant"])
    mat_conf_threshold = (
        config.get("models", {}).get("material", {}).get("confidence_threshold", 0.6)
    )

    results_list = []
    for p_det in potholes:
        p_mask        = p_det.mask
        combined_ref  = ref_mask if ref_mask is not None else p_mask

        vol_res = calc.calculate_volume(
            p_mask, combined_ref, p_det.bbox, depth_map, ref_area, ref_type
        )

        # Material classification on the cropped pothole patch
        x1, y1, x2, y2 = p_det.bbox
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(image_np.shape[1], int(x2)); y2 = min(image_np.shape[0], int(y2))
        crop = image_np[y1:y2, x1:x2]

        mat_res      = None
        surface_type = "asphalt"
        if crop.size > 0:
            mat_res = material_classifier.predict(crop)
            if mat_res["confidence"] >= mat_conf_threshold:
                surface_type = mat_res["class"]

        sev_res = SeverityClassifier().classify(
            vol_res.avg_depth_cm, vol_res.area_cm2, vol_res.volume_cm3
        )
        rep_res = RepairAdvisor().recommend(
            vol_res.volume_cm3, vol_res.avg_depth_cm, vol_res.area_cm2,
            severity_level=sev_res.level, surface_type=surface_type
        )

        results_list.append({
            "volumetric":    vol_res,
            "severity":      sev_res,
            "repair":        rep_res,
            "surface_type":  surface_type,
            "surface_conf":  mat_res["confidence"] if mat_res else 0.0,
            "pothole_mask":  p_mask,
        })

    top = sorted(results_list, key=lambda x: x["severity"].score, reverse=True)[0]

    return {
        "annotated":    seg_res["annotated_image"],
        "depth_viz":    depth_estimator.visualize_depth(depth_map),
        "potholes":     results_list,
        "summary": {
            "area_cm2":        sum(p["volumetric"].area_cm2    for p in results_list),
            "volume_cm3":      sum(p["volumetric"].volume_cm3  for p in results_list),
            "severity_level":  top["severity"].level,
            "severity_score":  top["severity"].score,
            "repair_method":   top["repair"].method if len(results_list) == 1 else "Multiple",
            "repair_cost_idr": sum(p["repair"].total_cost_idr  for p in results_list),
            "repair_material_kg": sum(p["repair"].material_kg  for p in results_list),
        },
        "depth_raw":    depth_map,
        "original_rgb": image_np,
    }


# ── Helper: PDF bytes ──────────────────────────────────────────────────────────
def _make_pdf_bytes(res: dict) -> bytes:
    potholes = res["potholes"]
    report_data = {
        "image_name":        res.get("image_name", "unknown"),
        "area_cm2":          res["summary"]["area_cm2"],
        "avg_depth_cm":      sum(p["volumetric"].avg_depth_cm for p in potholes) / len(potholes),
        "volume_cm3":        res["summary"]["volume_cm3"],
        "severity_level":    res["summary"]["severity_level"],
        "severity_score":    res["summary"]["severity_score"],
        "repair_method":     res["summary"]["repair_method"],
        "repair_cost_idr":   res["summary"]["repair_cost_idr"],
        "repair_material_kg":res["summary"]["repair_material_kg"],
        "annotated_path":    res.get("annotated_path"),
        "latitude":          res.get("latitude"),
        "longitude":         res.get("longitude"),
    }
    pdf_path = report_gen.generate_pdf_report(report_data)
    with open(pdf_path, "rb") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Analyze
# ══════════════════════════════════════════════════════════════════════════════
def page_analyze():
    st.markdown('<h1 class="premium-header">Analyze</h1>', unsafe_allow_html=True)
    st.markdown("Upload one or more photos for pothole tomography & volumetric analysis.")

    # ── Threshold controls ────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        conf_t = st.slider(
            "Confidence Threshold", 0.05, 1.0, 0.25, 0.05,
            help="Lower = more detections but more noise"
        )
    with col2:
        iou_t = st.slider(
            "IoU Threshold", 0.1, 1.0, 0.45, 0.05,
            help="Intersection-over-Union for overlapping bounding boxes"
        )

    segmenter, depth_est, mat_clf, config = get_models(conf_t, iou_t)
    if segmenter is None:
        st.stop()

    # ── File uploader ─────────────────────────────────────────────────
    files = st.file_uploader(
        "Upload Images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if files and st.button("START BATCH ANALYSIS"):
        progress = st.progress(0)
        status   = st.empty()
        batch_results = []

        for i, file in enumerate(files):
            status.text(f"Processing {file.name} ({i + 1}/{len(files)})…")
            img_np = np.array(Image.open(file).convert("RGB"))

            try:
                res = run_analysis(img_np, segmenter, depth_est, mat_clf, config)
                if res:
                    # Save annotated image
                    save_dir = ROOT_DIR / "data" / "processed" / "analysis_results"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = save_dir / f"annotated_{ts}_{file.name}"
                    cv2.imwrite(
                        str(img_path),
                        cv2.cvtColor(res["annotated"], cv2.COLOR_RGB2BGR)
                    )
                    res["annotated_path"] = str(img_path)

                    # GPS extraction
                    tmp = ROOT_DIR / f"tmp_exif_{file.name}"
                    tmp.write_bytes(file.getbuffer())
                    lat, lon = extract_gps(tmp)
                    if tmp.exists():
                        tmp.unlink()
                    res["latitude"]  = lat
                    res["longitude"] = lon

                    # Persist to DB
                    db_data = {
                        "image_name":        file.name,
                        "image_path":        str(img_path),
                        "area_cm2":          res["summary"]["area_cm2"],
                        "avg_depth_cm":      res["potholes"][0]["volumetric"].avg_depth_cm,
                        "volume_cm3":        res["summary"]["volume_cm3"],
                        "severity_level":    res["summary"]["severity_level"],
                        "severity_score":    res["summary"]["severity_score"],
                        "repair_method":     res["summary"]["repair_method"],
                        "repair_cost_idr":   res["summary"]["repair_cost_idr"],
                        "repair_material_kg":res["summary"]["repair_material_kg"],
                        "latitude":          lat,
                        "longitude":         lon,
                    }
                    res["db_id"]      = history_mgr.save_analysis(db_data)
                    res["image_name"] = file.name
                    batch_results.append(res)
                else:
                    st.warning(f"{file.name}: No potholes detected (conf > {conf_t})")

            except Exception as exc:
                st.error(f"{file.name}: Error — {exc}")

            progress.progress((i + 1) / len(files))

        if batch_results:
            status.success(f"Success: {len(batch_results)}/{len(files)} images processed.")
            st.session_state["batch_results"] = batch_results
        else:
            status.error(
                "No images were successfully processed. "
                "Make sure the images clearly show potholes, "
                "or lower the Confidence Threshold."
            )

    # ── Display results ───────────────────────────────────────────────
    if "batch_results" not in st.session_state:
        return

    for i, res in enumerate(st.session_state["batch_results"]):
        n_holes = len(res["potholes"])
        label   = (
            f"📸 {res['image_name']}  —  "
            f"{n_holes} pothole(s) detected  —  "
            f"Severity: {res['summary']['severity_level']}"
        )
        with st.expander(label, expanded=(i == 0)):
            # Top row: annotated image
            st.image(res["annotated"], caption="Detection Result (YOLO)", use_container_width=True)

            st.markdown("---")

            # Per-pothole tabs
            tabs = st.tabs([f"Pothole #{p + 1}" for p in range(n_holes)])
            for p_idx, (tab, p_res) in enumerate(zip(tabs, res["potholes"])):
                with tab:
                    col_3d, col_info = st.columns([1, 1])

                    # ── 3-D visualisation ────────────────────────────
                    with col_3d:
                        st.markdown("#### 🔬 3D Tomography")
                        show_3d = st.toggle("Open 3D Model", key=f"toggle_3d_{i}_{p_idx}")
                        
                        if show_3d:
                            viz = Mesh3DVisualizer()
                            fig = viz.create_premium_pothole_mesh(
                                res["depth_raw"],
                                p_res["pothole_mask"],
                                metrics={
                                    "depth":    p_res["volumetric"].avg_depth_cm,
                                    "area":     p_res["volumetric"].area_cm2,
                                    "severity": p_res["severity"].level,
                                }
                            )
                            st.plotly_chart(
                                fig, use_container_width=True,
                                key=f"plotly_{i}_{p_idx}"
                            )
                        else:
                            st.info("3D Tomography is disabled to save memory. Click the toggle above to view.")

                    # ── Metrics & repair info ─────────────────────────
                    with col_info:
                        sev = p_res["severity"]
                        st.markdown("#### 📊 Metrics & Recommendations")
                        st.markdown(
                            f'<span class="severity-badge" '
                            f'style="background:{sev.color};color:white">'
                            f'SEVERITY: {sev.level}</span>',
                            unsafe_allow_html=True
                        )
                        st.write("")

                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Severity Score", f"{sev.score}/10")
                        mc2.metric("Avg. Depth", f"{p_res['volumetric'].avg_depth_cm:.1f} cm")
                        mc3.metric("Volume", f"{p_res['volumetric'].volume_cm3:.0f} cm³")

                        st.metric("Surface Area", f"{p_res['volumetric'].area_cm2:.1f} cm²")

                        st.markdown("---")
                        rep = p_res["repair"]
                        st.write("**🔧 Repair Method:**", rep.method)
                        st.write(
                            f"**Road Material:** {p_res['surface_type'].capitalize()} "
                            f"(conf {p_res['surface_conf']:.2f})"
                        )
                        st.metric("Estimated Cost", f"Rp {rep.total_cost_idr:,.0f}")

                        st.markdown("**Material Details:**")
                        st.write(f"- {rep.material_name}: **{rep.material_kg:.2f} kg**")
                        st.write(
                            f"- Sealant / Tack Coat: "
                            f"**{p_res['volumetric'].area_cm2 * 0.0001:.3f} L**"
                        )
                        st.write(f"- Est. time: **{rep.estimated_time_hours:.1f} hours**")
                        st.write(f"- Durability: **{rep.durability_months} months**")

                        if sev.notes if hasattr(sev, "notes") else False:
                            st.info(sev.notes)

            # ── PDF download ─────────────────────────────────────────
            st.markdown("---")
            try:
                pdf_bytes = _make_pdf_bytes(res)
                st.download_button(
                    label=f"⬇️ Download PDF Report — {res['image_name']}",
                    data=pdf_bytes,
                    file_name=f"Report_{res['image_name']}.pdf",
                    mime="application/pdf",
                    key=f"dl_btn_{i}"
                )
            except Exception as exc:
                st.warning(f"Failed to generate PDF: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: History
# ══════════════════════════════════════════════════════════════════════════════
def page_history():
    st.markdown('<h1 class="premium-header">History</h1>', unsafe_allow_html=True)

    history = history_mgr.get_all_history()
    if not history:
        st.info("No analysis history available yet.")
        return

    df_rows = []
    for h in history:
        df_rows.append({
            "ID":          h.id,
            "Date":        h.timestamp.strftime("%Y-%m-%d %H:%M"),
            "Image":       h.image_name,
            "Severity":    h.severity_level,
            "Volume (cm³)":f"{h.volume_cm3:.1f}",
            "Cost (IDR)":  f"{h.repair_cost_idr:,.0f}",
            "Lat":         f"{h.latitude:.5f}" if h.latitude else "—",
            "Lon":         f"{h.longitude:.5f}" if h.longitude else "—",
        })

    df = pd.DataFrame(df_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Detailed View Section ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Search & Detailed View")
    
    if not df.empty:
        # Create option list for selectbox: "ID - ImageName"
        options = [f"{h.id} - {h.image_name}" for h in history]
        selected_option = st.selectbox("Select an entry to view details:", options)
        
        if selected_option:
            selected_id = int(selected_option.split(" - ")[0])
            # Find the history object
            h_detail = next((h for h in history if h.id == selected_id), None)
            
            if h_detail:
                col_img, col_metrics = st.columns([1, 1])
                
                with col_img:
                    if h_detail.image_path and Path(h_detail.image_path).exists():
                        st.image(h_detail.image_path, caption=f"Analyzed Image: {h_detail.image_name}", use_container_width=True)
                    else:
                        st.warning("Annotated image file not found on disk.")
                
                with col_metrics:
                    st.markdown(f"#### Analysis Details (ID: {h_detail.id})")
                    
                    # Severity Badge
                    color_map = {"CRITICAL": "#9C27B0", "HIGH": "#F44336", "MEDIUM": "#FF9800", "LOW": "#4CAF50"}
                    color = color_map.get(h_detail.severity_level, "#94a3b8")
                    st.markdown(
                        f'<span class="severity-badge" style="background:{color};color:white">'
                        f'SEVERITY: {h_detail.severity_level}</span>',
                        unsafe_allow_html=True
                    )
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Area", f"{h_detail.area_cm2:.1f} cm²")
                    m1.metric("Avg. Depth", f"{h_detail.avg_depth_cm:.1f} cm")
                    m2.metric("Volume", f"{h_detail.volume_cm3:.1f} cm³")
                    m2.metric("Score", f"{h_detail.severity_score:.1f}/10")
                    
                    st.markdown("---")
                    st.markdown("**🔧 Maintenance Info**")
                    st.write(f"**Method:** {h_detail.repair_method}")
                    st.write(f"**Est. Cost:** Rp {h_detail.repair_cost_idr:,.0f}")
                    st.write(f"**Material Needed:** {h_detail.repair_material_kg:.2f} kg")
                    
                    st.markdown("---")
                    st.markdown("**📍 Location**")
                    loc_str = f"{h_detail.latitude:.6f}, {h_detail.longitude:.6f}" if h_detail.latitude else "No GPS Data"
                    st.write(f"**Coordinates:** {loc_str}")
                    
                    # Delete button
                    if st.button("🗑️ Delete this Entry", key=f"del_{h_detail.id}"):
                        if history_mgr.delete_entry(h_detail.id):
                            st.success("Entry deleted. Refreshing...")
                            st.rerun()
                        else:
                            st.error("Failed to delete entry.")

    # Summary stats
    st.markdown("---")
    st.markdown("### 📈 Overall Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Analyses",  len(history))
    col2.metric("Critical", sum(1 for h in history if h.severity_level == "CRITICAL"))
    col3.metric("High",     sum(1 for h in history if h.severity_level == "HIGH"))
    col4.metric(
        "Total Estimated Cost",
        f"Rp {sum(h.repair_cost_idr or 0 for h in history):,.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Map
# ══════════════════════════════════════════════════════════════════════════════
def page_map():
    st.markdown('<h1 class="premium-header">Live Map</h1>', unsafe_allow_html=True)

    history      = history_mgr.get_all_history()
    valid_coords = [h for h in history if h.latitude and h.longitude]

    if not valid_coords:
        st.warning(
            "No GPS data found in history. "
            "Upload photos with GPS EXIF metadata to display markers on the map."
        )
        m = folium.Map(location=[-6.2088, 106.8456], zoom_start=12)
    else:
        m = folium.Map(
            location=[valid_coords[0].latitude, valid_coords[0].longitude],
            zoom_start=15
        )
        color_map = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "blue", "LOW": "green"}
        for h in valid_coords:
            color = color_map.get(h.severity_level, "gray")
            popup_html = (
                f"<b>{h.image_name}</b><br>"
                f"Severity: {h.severity_level}<br>"
                f"Volume: {h.volume_cm3:.1f} cm³<br>"
                f"Cost: Rp {h.repair_cost_idr:,.0f}"
            )
            folium.Marker(
                [h.latitude, h.longitude],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=h.image_name,
                icon=folium.Icon(color=color, icon="exclamation-sign")
            ).add_to(m)

    st_folium(m, width=1200, height=600)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    with st.sidebar:
        st.markdown('<h2 style="color:#60a5fa;font-weight:800">InfraSight</h2>',
                    unsafe_allow_html=True)
        st.markdown("*AI-Powered Pothole Analytics*")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["📷 Analyze", "📋 History", "🗺️ Damage Map"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.info(
            "**How to Use:**\n\n"
            "1. Upload pothole photos (include a card / coin as size reference).\n"
            "2. Press **START BATCH ANALYSIS**.\n"
            "3. View 3D visualization, metrics, and repair recommendations per pothole.\n"
            "4. Download the PDF report."
        )

    if page == "📷 Analyze":
        page_analyze()
    elif page == "📋 History":
        page_history()
    elif page == "🗺️ Damage Map":
        page_map()


if __name__ == "__main__":
    main()