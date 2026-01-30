"""
report.py - Report Generation

This module handles the generation of PDF reports using ReportLab.
Includes comprehensive error handling and data validation.
"""

import io
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from exceptions import ReportGenerationError
from logging_config import get_logger

logger = get_logger("report")


# =============================================================================
# PDF REPORT GENERATION
# =============================================================================

def generate_pdf_report(res: Dict[str, Any]) -> bytes:
    """
    Generate comprehensive PDF report of analysis results.
    
    Parameters
    ----------
    res : dict
        Analysis results dictionary
    
    Returns
    -------
    bytes
        PDF file as bytes
    
    Raises
    ------
    ReportGenerationError
        If report generation fails
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, title="Seismic Analysis Report")
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Seismic Analysis Report", styles["Title"]))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))
        
        # Disclaimer
        story.append(Paragraph(
            "<b>DISCLAIMER:</b> Ground motions used in this analysis are synthetically generated "
            "for educational purposes. They are NOT actual recorded earthquake data.",
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))
        
        # Summary Section
        summary_rows = [
            ["Parameter", "Value"],
            ["Analysis Type", res.get("analysis_type", "-")],
            ["Earthquake Event", res.get("event_type", "-")],
            ["PGA (g)", f"{res.get('pga', 0):.2f}"],
            ["Duration (s)", f"{res.get('t', [0])[-1]:.2f}"],
            ["Number of Floors", str(res.get("n_floors", "-"))],
            ["Fundamental Period (s)", f"{res.get('periods', [0])[0]:.4f}"],
            ["Max Drift Ratio (%)", f"{res.get('max_drift_ratio', 0):.3f}"],
            ["Max Base Shear (kN)", f"{res.get('max_base_shear', 0):.1f}"],
            ["Performance Level", res.get("perf_level", "-")],
            ["Damage Index", f"{res.get('damage_index', 0):.3f}"],
        ]
        story.append(Paragraph("Summary", styles["Heading2"]))
        story.append(_styled_table(summary_rows))
        story.append(Spacer(1, 12))
        
        # Model Inputs
        input_rows = [
            ["Input", "Value"],
            ["Floor Height (m)", f"{res.get('floor_height', 0):.2f}"],
            ["Base Isolation", "Yes" if res.get('controls', {}).get('base_isolated') else "No"],
            ["Damper Type", res.get('controls', {}).get('damper_type', "-")],
        ]
        story.append(Paragraph("Model Inputs", styles["Heading2"]))
        story.append(_styled_table(input_rows))
        story.append(Spacer(1, 12))
        
        # Time history statistics
        time_series = res.get("t", np.array([0.0]))
        roof_disp = res.get("roof_disp", np.array([0.0]))
        roof_vel = res.get("roof_vel", np.array([0.0]))
        roof_acc = res.get("roof_acc", np.array([0.0]))
        base_shear = res.get("base_shear", np.array([0.0]))
        
        stats_rows = [
            ["Time History Stats", "Value"],
            ["Roof Displacement RMS (cm)", f"{np.sqrt(np.mean(roof_disp**2)):.2f}"],
            ["Roof Velocity RMS (cm/s)", f"{np.sqrt(np.mean(roof_vel**2)):.2f}"],
            ["Roof Acceleration RMS (g)", f"{np.sqrt(np.mean(roof_acc**2)):.3f}"],
            ["Base Shear RMS (kN)", f"{np.sqrt(np.mean(base_shear**2)):.1f}"],
            ["Peak Ground Accel (g)", f"{np.max(np.abs(res.get('ag', np.array([0.0])))) / 9.81:.3f}"],
            ["Duration (s)", f"{time_series[-1]:.2f}"],
        ]
        story.append(Paragraph("Time History Statistics", styles["Heading2"]))
        story.append(_styled_table(stats_rows))
        story.append(Spacer(1, 12))
        
        # Peak responses
        peaks = [
            ["Peak Response", "Value"],
            ["Peak Roof Displacement (cm)", f"{np.max(np.abs(res.get('roof_disp', [0]))):.2f}"],
            ["Peak Roof Acceleration (g)", f"{np.max(np.abs(res.get('roof_acc', [0]))):.3f}"],
            ["Peak Base Shear (kN)", f"{np.max(np.abs(res.get('base_shear', [0]))):.1f}"],
        ]
        story.append(Paragraph("Peak Responses", styles["Heading2"]))
        story.append(_styled_table(peaks))
        story.append(Spacer(1, 12))
        
        # Residual Drift (for nonlinear analysis)
        residual_drift = res.get("residual_drift", np.array([]))
        if len(residual_drift) > 0 and np.any(residual_drift != 0):
            residual_rows = [["Floor", "Residual Drift (%)"]]
            for i, rd in enumerate(residual_drift):
                residual_rows.append([f"F{i+1}", f"{rd:.3f}"])
            story.append(Paragraph("Residual Drift (Nonlinear Analysis)", styles["Heading2"]))
            story.append(_styled_table(residual_rows))
            story.append(Spacer(1, 12))
        
        # Energy summary
        energy = res.get("energy", {})
        if energy:
            energy_rows = [
                ["Energy Component", "Value (kJ)"],
                ["Input", f"{energy['input'][-1]/1000:.2f}"],
                ["Kinetic", f"{energy['kinetic'][-1]/1000:.2f}"],
                ["Strain", f"{energy['strain'][-1]/1000:.2f}"],
                ["Damping", f"{energy['damping'][-1]/1000:.2f}"],
                ["Hysteretic", f"{energy['hysteretic'][-1]/1000:.2f}"],
            ]
            story.append(Paragraph("Energy Balance", styles["Heading2"]))
            story.append(_styled_table(energy_rows))
            story.append(Spacer(1, 12))
        
        # Modal information
        periods = res.get("periods", np.array([]))
        if len(periods) > 0:
            modal_rows = [["Mode", "Period (s)", "Frequency (Hz)"]]
            max_modes = min(5, len(periods))
            for i in range(max_modes):
                modal_rows.append([
                    f"Mode {i+1}",
                    f"{periods[i]:.4f}",
                    f"{1.0 / max(periods[i], 1e-6):.3f}",
                ])
            story.append(Paragraph("Modal Summary", styles["Heading2"]))
            story.append(_styled_table(modal_rows))
            story.append(Spacer(1, 12))
        
        # Drift profile summary
        max_drifts = res.get("max_drifts_per_floor", np.array([]))
        if len(max_drifts) > 0:
            drift_rows = [["Floor", "Max Drift (%)"]]
            for i, d in enumerate(max_drifts):
                drift_rows.append([f"F{i+1}", f"{d:.3f}"])
            story.append(Paragraph("Interstorey Drift Summary", styles["Heading2"]))
            story.append(_styled_table(drift_rows))
            story.append(Spacer(1, 12))
        
        # Comparison summary
        comp = res.get("comparison", {})
        if comp:
            comp_rows = [
                ["Comparison Metric", "Value"],
                ["Max Drift (Controlled) %", f"{comp.get('max_drift_controlled', 0):.3f}"],
                ["Max Drift (Uncontrolled) %", f"{comp.get('max_drift_uncontrolled', 0):.3f}"],
                ["Reduction (%)", f"{comp.get('reduction_percent', 0):.1f}"],
            ]
            story.append(Paragraph("Control System Comparison", styles["Heading2"]))
            story.append(_styled_table(comp_rows))
            story.append(Spacer(1, 12))
        
        # IS 1893 summary
        if res.get("is_code", {}).get("enabled"):
            is_code = res["is_code"]
            is_rows = [
                ["IS 1893 Parameter", "Value"],
                ["Zone", is_code.get("zone")],
                ["Soil", is_code.get("soil")],
                ["Importance (I)", f"{is_code.get('importance', 0):.2f}"],
                ["Response Reduction (R)", f"{is_code.get('response_reduction', 0):.2f}"],
                ["Ah", f"{is_code.get('Ah', 0):.4f}"],
                ["Design Base Shear (kN)", f"{is_code.get('Vb', 0)/1000:.1f}"],
                ["Drift Limit (%)", f"{is_code.get('drift_limit_pct', 0):.2f}"],
            ]
            story.append(Paragraph("IS 1893 Design Summary", styles["Heading2"]))
            story.append(_styled_table(is_rows))
            story.append(Spacer(1, 12))
        
        # Recommendations
        recs = res.get("recommendations", [])
        if recs:
            story.append(Paragraph("Recommendations", styles["Heading2"]))
            for rec in recs:
                # Escape special characters for PDF
                safe_rec = rec.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe_rec, styles["Normal"]))
        
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        logger.info(f"PDF report generated successfully ({len(pdf_bytes)} bytes)")
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise ReportGenerationError("PDF", str(e))


def _styled_table(rows: List[List[str]]) -> Table:
    """
    Create a styled table for the PDF report.
    
    Parameters
    ----------
    rows : list
        Table rows as list of lists
    
    Returns
    -------
    Table
        Styled ReportLab table
    """
    # Sanitize all cell values
    safe_rows = []
    for row in rows:
        safe_row = []
        for cell in row:
            if cell is None:
                safe_row.append("-")
            else:
                safe_cell = str(cell).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                safe_row.append(safe_cell)
        safe_rows.append(safe_row)
    
    table = Table(safe_rows, hAlign="LEFT")
    return table
