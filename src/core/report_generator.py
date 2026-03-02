"""
Report Generator for InfraSight
Generates professional PDF maintenance reports for pothole analysis.
"""
from fpdf import FPDF
from pathlib import Path
import datetime
from src.utils.logger import setup_logger

logger = setup_logger("ReportGenerator")

class ReportGenerator:
    def __init__(self, output_dir=None):
        if output_dir is None:
            # Resolve relative to project root
            root_dir = Path(__file__).parent.parent.parent.absolute()
            self.output_dir = root_dir / "reports"
        else:
            self.output_dir = Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf_report(self, analysis_data):
        """
        Create a professional maintenance report for potholes.
        """
        pdf = FPDF()
        pdf.add_page()
        
        # --- Header Section ---
        # Draw a dark blue top banner
        pdf.set_fill_color(15, 23, 42) # #0f172a
        pdf.rect(0, 0, 210, 40, 'F')
        
        pdf.set_xy(10, 12)
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "INFRASIGHT ANALYTICS", ln=True, align="L")
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(148, 163, 184) # #94a3b8
        pdf.cell(0, 5, "Automated Pothole Tomography & Maintenance Report", ln=True, align="L")
        
        pdf.ln(20)
        
        # --- Report Metadata ---
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(95, 10, "REPORT DETAILS", ln=False)
        pdf.cell(95, 10, "LOCATION DATA", ln=True)
        
        pdf.set_font("Helvetica", "", 10)
        # Col 1
        pdf.cell(35, 7, "Date:", ln=False)
        pdf.cell(60, 7, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ln=False)
        # Col 2
        lat = analysis_data.get('latitude')
        lon = analysis_data.get('longitude')
        location_str = f"{lat:.6f}, {lon:.6f}" if lat and lon else "GPS Not Available"
        pdf.cell(35, 7, "Coordinates:", ln=False)
        pdf.cell(0, 7, location_str, ln=True)
        
        pdf.cell(35, 7, "Image ID:", ln=False)
        pdf.cell(60, 7, analysis_data.get('image_name', 'Unknown'), ln=False)
        pdf.cell(35, 7, "Status:", ln=False)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "ANALYSIS COMPLETE", ln=True)
        
        pdf.ln(10)
        
        # --- Metrics & Severity Section ---
        pdf.set_draw_color(226, 232, 240) # #e2e8f0
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Create two columns for Metrics and Severity
        start_y = pdf.get_y()
        
        # Metrics Table (Left)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(95, 10, "QUANTITATIVE METRICS", ln=True)
        pdf.set_font("Helvetica", "", 10)
        
        metrics = [
            ("Total Surface Area", f"{analysis_data.get('area_cm2', 0):.1f} cm²"),
            ("Avg. Target Depth", f"{analysis_data.get('avg_depth_cm', 0):.1f} cm"),
            ("Total Material Volume", f"{analysis_data.get('volume_cm3', 0):.1f} cm³"),
        ]
        
        for name, val in metrics:
            pdf.set_fill_color(248, 250, 252)
            pdf.cell(50, 8, name, border='B', fill=True)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(45, 8, val, border='B', fill=True, align='R')
            pdf.set_font("Helvetica", "", 10)
            pdf.ln()
            
        # Severity Badge (Right)
        pdf.set_xy(110, start_y)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(90, 10, "SEVERITY CLASSIFICATION", ln=True)
        
        severity = str(analysis_data.get('severity_level', 'UNKNOWN')).upper()
        score = analysis_data.get('severity_score', 0)
        
        pdf.set_x(110)
        if severity == 'CRITICAL': pdf.set_fill_color(220, 38, 38); pdf.set_text_color(255, 255, 255)
        elif severity == 'HIGH': pdf.set_fill_color(249, 115, 22); pdf.set_text_color(255, 255, 255)
        elif severity == 'MEDIUM': pdf.set_fill_color(234, 179, 8); pdf.set_text_color(0, 0, 0)
        else: pdf.set_fill_color(34, 197, 94); pdf.set_text_color(255, 255, 255)
        
        pdf.cell(90, 12, f"LEVEL: {severity} (Score: {score:.1f}/10)", ln=True, fill=True, align='C')
        pdf.set_text_color(0, 0, 0)
        
        pdf.ln(10)
        
        # --- Repair & Cost Section ---
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "MAINTENANCE RECOMMENDATION", ln=True)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(40, 8, "Repair Method:", ln=False)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, analysis_data.get('repair_method', 'N/A'), ln=True)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(40, 8, "Target Material:", ln=False)
        pdf.cell(0, 8, "Standard Hot-Mix Asphalt / Cold-Patch as required", ln=True)
        
        pdf.ln(3)
        # Cost Box
        pdf.set_fill_color(241, 245, 249)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(190, 15, f"ESTIMATED REPAIR COST: IDR {analysis_data.get('repair_cost_idr', 0):,.0f}", ln=True, fill=True, align="C")
        
        # --- Visualization Section ---
        annotated_path = analysis_data.get('annotated_path')
        if annotated_path and Path(annotated_path).exists():
            pdf.ln(8)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "COMPUTER VISION ANALYSIS (DETECTION MASK)", ln=True)
            # Try to center image
            pdf.image(annotated_path, x=15, w=180)
            
        # --- Footer ---
        pdf.set_y(-20)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 10, f"InfraSight Digital Twin Analysis Report | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")
        
        # Save
        filename = f"report_{analysis_data.get('image_name', 'analysis')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = self.output_dir / filename
        pdf.output(str(output_path))
        logger.info(f"Report generated: {output_path.name}")
        return str(output_path)
