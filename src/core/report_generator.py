"""
Report Generator for InfraSight
Generates professional PDF maintenance reports for pothole analysis.
"""
from fpdf import FPDF
from pathlib import Path
import datetime

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
        Create a maintenance report for a single pothole.
        analysis_data: dict containing metrics & analysis
        """
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        
        # Header
        pdf.set_text_color(40, 60, 120)
        pdf.cell(0, 15, "INFRA-SIGHT: POTHOLE MAINTENANCE REPORT", ln=True, align="C")
        pdf.ln(5)
        
        # Horizontal line
        pdf.set_draw_color(40, 60, 120)
        pdf.line(10, 30, 200, 30)
        pdf.ln(10)
        
        # General Info Section
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "General Information", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(50, 8, f"Report Date: ", ln=False)
        pdf.cell(0, 8, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ln=True)
        pdf.cell(50, 8, f"Image File: ", ln=False)
        pdf.cell(0, 8, analysis_data.get('image_name', 'N/A'), ln=True)
        
        # Location Info
        lat = analysis_data.get('latitude')
        lon = analysis_data.get('longitude')
        location_str = f"{lat:.6f}, {lon:.6f}" if lat and lon else "Not available (no EXIF GPS)"
        pdf.cell(50, 8, f"GPS Location: ", ln=False)
        pdf.cell(0, 8, location_str, ln=True)
        pdf.ln(5)
        
        # Metrics Section
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Pothole Metrics", ln=True)
        pdf.set_font("Helvetica", "", 12)
        
        # Metrics Table Header
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(60, 10, "Metric", border=1, fill=True)
        pdf.cell(60, 10, "Value", border=1, fill=True)
        pdf.ln()
        
        metrics = [
            ("Surface Area", f"{analysis_data.get('area_cm2', 0):.1f} cm²"),
            ("Average Depth", f"{analysis_data.get('avg_depth_cm', 0):.1f} cm"),
            ("Volume", f"{analysis_data.get('volume_cm3', 0):.1f} cm³"),
        ]
        
        for name, val in metrics:
            pdf.cell(60, 10, name, border=1)
            pdf.cell(60, 10, val, border=1)
            pdf.ln()
        
        pdf.ln(5)
        
        # Severity Section
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Severity Classification", ln=True)
        pdf.set_font("Helvetica", "B", 16)
        
        severity = analysis_data.get('severity_level', 'UNKNOWN')
        if severity == 'CRITICAL': pdf.set_text_color(220, 20, 60)
        elif severity == 'HIGH': pdf.set_text_color(255, 69, 0)
        elif severity == 'MEDIUM': pdf.set_text_color(218, 165, 32)
        else: pdf.set_text_color(34, 139, 34)
            
        pdf.cell(0, 10, f"LEVEL: {severity}", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Score: {analysis_data.get('severity_score', 0):.1f}/10", ln=True)
        pdf.ln(5)
        
        # Repair Section
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Repair Recommendation", ln=True)
        pdf.set_font("Helvetica", "", 12)
        
        pdf.cell(60, 8, "Method:", ln=False)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, analysis_data.get('repair_method', 'N/A'), ln=True)
        pdf.set_font("Helvetica", "", 12)
        
        # Bill of Materials (BoQ)
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 12)
        pdf.cell(0, 8, "Bill of Quantities (Estimated):", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(10)
        pdf.cell(60, 8, f"- Main Material: {analysis_data.get('repair_material_kg', 0):.2f} kg", ln=True)
        pdf.cell(10)
        pdf.cell(60, 8, f"- Tack Coat / Sealant: {(analysis_data.get('area_cm2', 0) * 0.0001):.3f} L", ln=True)
        pdf.ln(2)
        
        # Cost
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_fill_color(255, 235, 204)
        pdf.cell(120, 12, f"Total Estimated Cost: IDR {analysis_data.get('repair_cost_idr', 0):,.0f}", ln=True, fill=True, align="C")
        
        # Images (If possible)
        # We assume annotated_path is provided in analysis_data
        annotated_path = analysis_data.get('annotated_path')
        if annotated_path and Path(annotated_path).exists():
            pdf.ln(10)
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Analysis Visualization", ln=True)
            # Resize image to fit page
            pdf.image(annotated_path, w=160)
            
        # Footer
        pdf.set_y(-25)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, f"InfraSight generated report on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")
        
        # Save
        filename = f"report_{analysis_data.get('image_name', 'pothole')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = self.output_dir / filename
        pdf.output(str(output_path))
        return str(output_path)
