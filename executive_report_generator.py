# Save this code in a new file: executive_report_generator.py

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import pandas as pd
import numpy as np
import os

# Import core analysis functions (assuming you copy them here or import)
# For simplicity and isolation, we'll redefine the necessary metric functions/values here.
# NOTE: In a professional app, you'd import these functions from their respective files.

# --- MOCKUP METRICS (Must run the other files to get real metrics) ---
def get_all_metrics():
    # Placeholder values for demonstration (REPLACE with actual function calls)
    metrics = {
        # Time Series Forecast (Project 1)
        'forecast_q1': 8500, # kg
        'forecast_q2': 9200, # kg
        
        # Reusability Analysis (Project 2)
        'disp_cost_M': 10550.00, # Total simulated disposable cost
        'reusable_cost_M': 9500.00, # Total simulated reusable cost
        'savings_percent': 10.0,
        'breakeven_flights': 3.5,
        
        # Benchmarking (Project 5)
        'isro_success_rate': 0.57,
        'isro_cost_per_kg': 9371,
        
        # Scenario Simulation (Project 6)
        'scenario_success_rate': 0.72,
        'scenario_exp_loss_M': 18.2, # Expected Loss in $M for the high-risk GSLV Mk III
        'scenario_exp_roi_M': 85.0 # Net Expected Financial Outcome in $M
    }
    
    # Run the actual cost calculation (simplified, must match reusability_analysis.py logic)
    # The actual calculation requires loading the original data, which we skip for brevity here.
    
    return metrics

# --- PDF Generation Logic ---
def generate_executive_report(metrics, filename="ISRO_Executive_Launch_Report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom style for title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=20, alignment=1, textColor=colors.navy)
    
    # Content flow
    story = []

    # Title
    story.append(Paragraph("ISRO Strategic Launch Analysis & Forecast Report", title_style))
    story.append(Spacer(1, 12))
    
    # --- Section 1: Launch Forecasting (Project 1) ---
    story.append(Paragraph("<b>1. Time-Series Payload Forecast (Future 6 Months)</b>", styles['h2']))
    story.append(Paragraph(f"The model forecasts a high payload volume, with an estimated **{metrics['forecast_q1']:,} kg** in Q1 2026 and **{metrics['forecast_q2']:,} kg** in Q2 2026, indicating aggressive scheduling.", styles['BodyText']))
    story.append(Spacer(1, 12))

    # --- Section 2: Financial Strategy - Reusability (Project 2 & 8) ---
    story.append(Paragraph("<b>2. Deep Dive: Reusability ROI</b>", styles['h2']))
    story.append(Paragraph("Simulating a 30% reuse saving model reveals significant long-term potential.", styles['BodyText']))
    
    reusability_data = [
        ['Metric', 'Value'],
        ['Total Disposable Cost (Simulated)', f"${metrics['disp_cost_M']:,.2f} M"],
        ['Total Reusable Cost (Simulated)', f"${metrics['reusable_cost_M']:,.2f} M"],
        ['**Total Savings**', f"**${metrics['disp_cost_M'] - metrics['reusable_cost_M']:,.2f} M**"],
        ['Percentage Savings', f"{metrics['savings_percent']:.2f} %"],
        ['**Break-Even Point (Flights)**', f"**{metrics['breakeven_flights']:.1f} Reusable Launches**"],
    ]
    t = Table(reusability_data, colWidths=[200, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # --- Section 3: Risk Assessment (Project 6 & 8) ---
    story.append(Paragraph("<b>3. Scenario Simulation & Risk Analysis</b>", styles['h2']))
    story.append(Paragraph("A high-risk GSLV Mk III mission scenario was run using 10,000 Monte Carlo trials to assess financial exposure.", styles['BodyText']))

    risk_data = [
        ['Metric', 'Value'],
        ['Simulated Success Rate', f"{metrics['scenario_success_rate']:.2%}"],
        ['**Expected Financial Loss (Risk)**', f"**${metrics['scenario_exp_loss_M']:,.1f} M**"],
        ['Net Expected Financial Outcome (ROI)', f"${metrics['scenario_exp_roi_M']:,.1f} M"],
    ]
    t_risk = Table(risk_data, colWidths=[200, 150])
    t_risk.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.red),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.lightsalmon),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t_risk)
    story.append(Spacer(1, 12))
    
    # --- Section 4: Benchmarking (Project 5) ---
    story.append(Paragraph("<b>4. Benchmarking and Competitiveness</b>", styles['h2']))
    story.append(Paragraph("ISRO maintains a cost-competitive position compared to global agencies.", styles['BodyText']))
    story.append(Paragraph(f"Calculated Success Rate: **{metrics['isro_success_rate']:.2%}**", styles['BodyText']))
    story.append(Paragraph(f"Estimated Cost per kg to LEO: **${metrics['isro_cost_per_kg']:,.0f} USD** (Highly competitive)", styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # --- Final Recommendation ---
    story.append(Paragraph("<b>Recommendation</b>", styles['h2']))
    story.append(Paragraph("Accelerate the RLV program, as the break-even point is achievable within an aggressive launch schedule. Implement the advanced Stacking Classifier model (`isro_launch_model_v2.pkl`) into operational launch risk review for a more accurate success probability (72% in the simulated scenario).", styles['BodyText']))
    
    doc.build(story)
    print("\n" + "="*50)
    print(f"EXECUTIVE REPORT SUCCESSFULLY GENERATED: {filename}")
    print("="*50)

# --- Main Execution ---
if __name__ == "__main__":
    
    # NOTE: You MUST ensure the metrics are accurately pulled from the other scripts
    # or the report will contain only the mockup numbers.
    
    # 1. Run advanced_feature_engineering.py
    # 2. Run main.py (to create the model)
    # 3. Run time_series_forecasting.py
    # 4. Run reusability_analysis.py
    # 5. Run scenario_simulator.py (to get the scenario result)

    # Assuming all previous analysis scripts have been run successfully:
    all_metrics = get_all_metrics()
    generate_executive_report(all_metrics)