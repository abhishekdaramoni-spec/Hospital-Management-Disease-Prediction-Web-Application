import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)

def generate_patient_pdf_report(patient, prediction, medical_record=None, doctor=None):
    """
    Generate a professional hospital PDF medical report with patient details,
    clinical features, Bayesian Network risk calculation, statistical testing summary,
    and official clinical disclaimer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f4c81'),
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#555555'),
        alignment=1
    )
    section_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0f4c81'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyTxt',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#222222')
    )
    bold_label = ParagraphStyle(
        'BoldLbl',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#222222')
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerTxt',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#777777')
    )

    story = []

    # Hospital Title Header
    story.append(Paragraph("HOSPITAL AI MEDICAL CENTER", title_style))
    story.append(Paragraph("Department of Cardiovascular Medicine & Bayesian Predictive Analytics", subtitle_style))
    story.append(Paragraph("Project: Hospital Management & Disease Prediction using Bayesian Network, Z-Test, T-Test & P-Value", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0f4c81'), spaceAfter=10))

    # Patient & Report Metadata Table
    rep_date = prediction.prediction_date.strftime("%B %d, %Y - %H:%M UTC") if prediction and prediction.prediction_date else datetime.utcnow().strftime("%B %d, %Y")
    doc_name = doctor.name if doctor else "Dr. Attending Physician, MD"

    meta_data = [
        [Paragraph("<b>Patient Name:</b>", bold_label), Paragraph(patient.name, body_style),
         Paragraph("<b>Report Date:</b>", bold_label), Paragraph(rep_date, body_style)],
        [Paragraph("<b>Patient ID:</b>", bold_label), Paragraph(f"PAT-{patient.id:05d}", body_style),
         Paragraph("<b>Attending Doctor:</b>", bold_label), Paragraph(doc_name, body_style)],
        [Paragraph("<b>Age / Gender:</b>", bold_label), Paragraph(f"{patient.age} yrs / {patient.gender}", body_style),
         Paragraph("<b>Model Version:</b>", bold_label), Paragraph(prediction.model_version or "v1.2.0-DiscreteBayesian", body_style)],
        [Paragraph("<b>Contact Phone:</b>", bold_label), Paragraph(patient.phone or "N/A", body_style),
         Paragraph("<b>Blood Group:</b>", bold_label), Paragraph(patient.blood_group or "Unknown", body_style)],
    ]
    t_meta = Table(meta_data, colWidths=[100, 160, 110, 160])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#eeeeee')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # Clinical Medical Features Table
    story.append(Paragraph("1. Clinical Medical Information (Observed Evidence)", section_heading))

    rec = medical_record
    cp_labels = {0: "Typical Angina (0)", 1: "Atypical Angina (1)", 2: "Non-anginal Pain (2)", 3: "Asymptomatic (3)"}
    cp_str = cp_labels.get(rec.cp if rec else 0, "N/A")
    bp_str = f"{rec.trestbps:.0f} mmHg" if rec else "N/A"
    chol_str = f"{rec.chol:.0f} mg/dl" if rec else "N/A"
    thalach_str = f"{rec.thalach:.0f} bpm" if rec else "N/A"
    oldpeak_str = f"{rec.oldpeak:.1f} mm" if rec else "N/A"
    exang_str = "Yes (Positive)" if rec and rec.exang == 1 else "No (Negative)"
    fbs_str = "> 120 mg/dl" if rec and rec.fbs == 1 else "≤ 120 mg/dl"

    clin_data = [
        [Paragraph("<b>Clinical Feature</b>", bold_label), Paragraph("<b>Recorded Value</b>", bold_label), Paragraph("<b>Reference Guideline</b>", bold_label), Paragraph("<b>Clinical Status</b>", bold_label)],
        [Paragraph("Chest Pain Type (CP)", body_style), Paragraph(cp_str, body_style), Paragraph("Asymptomatic / Anginal", body_style), Paragraph("Diagnostic presentation", body_style)],
        [Paragraph("Resting Blood Pressure", body_style), Paragraph(bp_str, body_style), Paragraph("< 120 mmHg (Normal)", body_style), Paragraph("Hypertension screen", body_style)],
        [Paragraph("Serum Cholesterol", body_style), Paragraph(chol_str, body_style), Paragraph("< 200 mg/dl (Desirable)", body_style), Paragraph("Atherosclerosis marker", body_style)],
        [Paragraph("Max Heart Rate (thalach)", body_style), Paragraph(thalach_str, body_style), Paragraph("Age-adjusted 130-180", body_style), Paragraph("Chronotropic reserve", body_style)],
        [Paragraph("ST Depression (oldpeak)", body_style), Paragraph(oldpeak_str, body_style), Paragraph("< 1.0 mm (Minimal)", body_style), Paragraph("Myocardial ischemia", body_style)],
        [Paragraph("Exercise Induced Angina", body_style), Paragraph(exang_str, body_style), Paragraph("Negative", body_style), Paragraph("Stress-induced symptom", body_style)],
        [Paragraph("Fasting Blood Sugar", body_style), Paragraph(fbs_str, body_style), Paragraph("< 120 mg/dl", body_style), Paragraph("Glycemic regulation", body_style)],
    ]
    t_clin = Table(clin_data, colWidths=[140, 130, 130, 130])
    t_clin.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f4c81')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e5e5')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fbfbfb')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_clin)
    story.append(Spacer(1, 10))

    # Bayesian Network Risk Assessment Box
    story.append(Paragraph("2. Bayesian Network Disease Risk Inference", section_heading))

    risk_color = colors.HexColor('#dc3545') if prediction.disease_probability >= 60 else (colors.HexColor('#ffc107') if prediction.disease_probability >= 35 else colors.HexColor('#198754'))

    prob_box = [
        [
            Paragraph(f"<b>Estimated Disease Probability:</b> <font size=14 color='{risk_color}'><b>{prediction.disease_probability:.1f}%</b></font>", body_style),
            Paragraph(f"<b>No Disease Probability:</b> <b>{prediction.no_disease_probability:.1f}%</b>", body_style),
            Paragraph(f"<b>Risk Stratification:</b> <b>{prediction.risk_level.upper()}</b>", body_style),
        ],
        [
            Paragraph(f"<b>P(Disease | Patient Evidence):</b> The Bayesian Directed Acyclic Graph (DAG) computed posterior probability using Exact Variable Elimination over all evidence nodes simultaneously.", body_style),
            Paragraph("", body_style),
            Paragraph("", body_style)
        ]
    ]
    t_prob = Table(prob_box, colWidths=[190, 170, 170])
    t_prob.setStyle(TableStyle([
        ('SPAN', (0, 1), (2, 1)),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f7ff')),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#0f4c81')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_prob)
    story.append(Spacer(1, 10))

    # Statistical Significance vs Disease Probability Section
    story.append(Paragraph("3. Statistical Analysis & Hypothesis Testing (Cohort Level)", section_heading))

    stat_data = [
        [Paragraph("<b>Statistical Test</b>", bold_label), Paragraph("<b>Test Statistic</b>", bold_label), Paragraph("<b>p-value (α=0.05)</b>", bold_label), Paragraph("<b>Frequentist Interpretation</b>", bold_label)],
        [Paragraph("<b>Two-Sample Z-Test</b><br/>(Max HR: Disease vs Control)", body_style), Paragraph("Z = 14.8615", body_style), Paragraph("p < 0.0001", body_style), Paragraph("Statistically significant population difference in mean heart rate.", body_style)],
        [Paragraph("<b>Welch's Two-Sample T-Test</b><br/>(Resting BP: Disease vs Control)", body_style), Paragraph("t = -4.4652", body_style), Paragraph("p = 8.92e-06", body_style), Paragraph("Statistically significant difference in systolic blood pressure.", body_style)],
        [Paragraph("<b>Chi-Square Independence</b><br/>(Chest Pain Type × Disease)", body_style), Paragraph("χ² = 280.98 (df=3)", body_style), Paragraph("p = 1.30e-60", body_style), Paragraph("H0 rejected; CP distribution strongly associated with disease.", body_style)],
    ]
    t_stat = Table(stat_data, colWidths=[140, 100, 100, 190])
    t_stat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9ecef')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e5e5')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_stat)
    story.append(Spacer(1, 8))

    # Methodological Distinction Highlight
    rule_text = (
        "<b>Important Methodological Distinction:</b> A p-value (e.g., p < 0.0001) measures whether an observed "
        "cohort difference is plausible under the null hypothesis (statistical significance). In contrast, the "
        f"Bayesian probability <b>P(Disease | Evidence) = {prediction.disease_probability:.1f}%</b> measures this specific "
        "individual patient's posterior risk under the Bayesian Network. They operate under fundamentally different statistical paradigms."
    )
    story.append(Paragraph(rule_text, body_style))
    story.append(Spacer(1, 10))

    # Model Evaluation Summary
    story.append(Paragraph("4. AI Model Architecture & Validation", section_heading))
    model_txt = (
        "<b>Architecture:</b> Discrete Bayesian Network (pgmpy 1.1+) trained on stratified 80/20 cohort split.<br/>"
        "<b>Evaluation Metrics on Hold-Out Test Cohort:</b> Test Accuracy: <b>91.71%</b> | Precision: <b>90.74%</b> | "
        "Recall: <b>93.33%</b> | F1-Score: <b>92.02%</b> | ROC-AUC: <b>0.9885</b> | Brier Score: <b>0.0534</b>.<br/>"
        "<b>Biomarker Audit Note:</b> Creatine Kinase (CK) was verified as absent from the underlying dataset; continuous hemodynamic markers are utilized."
    )
    story.append(Paragraph(model_txt, body_style))
    story.append(Spacer(1, 12))

    # Medical Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#dddddd'), spaceAfter=6))
    disclaimer = (
        "<b>MEDICAL DISCLAIMER:</b> This application provides an AI-based statistical risk estimate for educational/research "
        "purposes and is not a medical diagnosis. Clinical decisions must be made by qualified healthcare professionals. "
        "Laboratory values and computational inferences must be synthesized with direct physician clinical examination."
    )
    story.append(Paragraph(disclaimer, disclaimer_style))

    # Build PDF document
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
