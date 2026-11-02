"""
Interoperability & Clinical Reporting Module:
Generates HL7 FHIR v4-Compatible Clinical Risk Export resources
and downloadable Decision-Support Consultation Summaries.
Includes structural schema validation functions.
"""

import json
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, Any, List, Tuple
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

DISCLAIMER_TEXT = (
    "NOTICE: This software is a research and educational clinical decision-support prototype. "
    "Risk estimates are mathematical model outputs and do not constitute a medical diagnosis, "
    "prescription, or substitute for professional clinical judgment. "
    "Clinical decisions must be made by qualified healthcare professionals."
)

def generate_fhir_risk_assessment(
    patient_id: str,
    patient_data: Dict[str, Any],
    risk_results: Dict[str, Any],
    shap_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Constructs an HL7 FHIR v4-Compatible RiskAssessment resource.
    Follows HL7 FHIR US Core Implementation Guide structure.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    tier_code = risk_results.get("tier_code", "GREEN")
    fhir_qualitative = {
        "GREEN": {"code": "low", "display": "Low risk (Project-Defined Tier)"},
        "YELLOW": {"code": "moderate", "display": "Moderate risk (Project-Defined Tier)"},
        "RED": {"code": "high", "display": "High risk (Project-Defined Tier)"}
    }.get(tier_code, {"code": "moderate", "display": "Moderate risk"})
    
    drivers = [d["label"] for d in shap_summary.get("top_risk_drivers", [])[:3]]
    rationale_text = f"Model Composite Multi-Morbidity Hazard: {risk_results.get('composite_hazard_percent')}%. "
    if drivers:
        rationale_text += f"Key model risk drivers: {', '.join(drivers)}. "
    rationale_text += risk_results.get("clinical_recommendation", "")
    
    fhir_resource = {
        "resourceType": "RiskAssessment",
        "id": f"risk-eval-{str(patient_id).lower()}-{int(datetime.now().timestamp())}",
        "meta": {
            "versionId": "1",
            "lastUpdated": now_iso,
            "profile": ["http://hl7.org/fhir/StructureDefinition/RiskAssessment"]
        },
        "status": "final",
        "subject": {
            "reference": f"Patient/{patient_id}",
            "display": f"Patient Record #{patient_id}"
        },
        "occurrenceDateTime": now_iso,
        "performer": {
            "display": "Smart AI Healthcare Decision-Support Prototype"
        },
        "basis": [
            {
                "reference": f"#obs-vitals-{patient_id}",
                "display": f"Vitals & Biomarkers: SBP {patient_data.get('systolic_bp')}, Glucose {patient_data.get('fasting_glucose')}, BMI {patient_data.get('bmi')}"
            }
        ],
        "prediction": [
            {
                "outcome": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "44808001",
                            "display": "Synergistic Multi-Morbidity & Cardiovascular Decompensation Risk"
                        }
                    ],
                    "text": "Cardiovascular & Metabolic Risk Estimation"
                },
                "probabilityDecimal": round(risk_results.get("composite_hazard_percent", 0.0) / 100.0, 4),
                "qualitativeRisk": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                            "code": fhir_qualitative["code"],
                            "display": fhir_qualitative["display"]
                        }
                    ]
                },
                "rationale": rationale_text
            }
        ],
        "mitigation": risk_results.get("clinical_recommendation", ""),
        "note": [
            {
                "text": DISCLAIMER_TEXT
            }
        ]
    }
    return fhir_resource

def validate_fhir_risk_assessment_structure(fhir_obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates structural compliance of the generated RiskAssessment resource.
    """
    errors = []
    if fhir_obj.get("resourceType") != "RiskAssessment":
        errors.append("Invalid or missing resourceType (must be 'RiskAssessment').")
    if fhir_obj.get("status") not in ["registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error", "unknown"]:
        errors.append(f"Invalid status: {fhir_obj.get('status')}")
    if "subject" not in fhir_obj or "reference" not in fhir_obj["subject"]:
        errors.append("Missing required subject reference.")
    if "prediction" not in fhir_obj or not isinstance(fhir_obj["prediction"], list) or len(fhir_obj["prediction"]) == 0:
        errors.append("Missing required prediction list.")
    else:
        pred = fhir_obj["prediction"][0]
        prob = pred.get("probabilityDecimal")
        if prob is None or not (0.0 <= prob <= 1.0):
            errors.append(f"probabilityDecimal must be bounded between 0.0 and 1.0; observed: {prob}")
        if "qualitativeRisk" not in pred:
            errors.append("Missing qualitativeRisk coding in prediction.")
            
    return (len(errors) == 0, errors)

def generate_clinical_consultation_summary(
    patient_id: str,
    patient_data: Dict[str, Any],
    risk_results: Dict[str, Any],
    shap_summary: Dict[str, Any],
    recourse_results: Dict[str, Any]
) -> str:
    """
    Generates a structured decision-support consultation summary.
    """
    now_str = datetime.now().strftime("%B %d, %Y - %H:%M")
    
    summary = f"""# CLINICAL DECISION-SUPPORT RISK ESTIMATION REPORT
**Patient Identifier**: {patient_id}
**Assessment Timestamp**: {now_str}
**Evaluating System**: Smart AI Healthcare Decision-Support Prototype (CM-SRF / MEC-CR Engines)

> **IMPORTANT DISCLAIMER**: {DISCLAIMER_TEXT}

---

### 1. PATIENT-ENTERED PHYSIOLOGICAL DATA
- **Age / Sex**: {patient_data.get('age', 'N/A')} years | {'Male' if patient_data.get('sex') == 1 else 'Female'}
- **Blood Pressure**: {patient_data.get('systolic_bp', 'N/A')} / {patient_data.get('diastolic_bp', 'N/A')} mmHg
- **Resting Heart Rate**: {patient_data.get('resting_hr', 'N/A')} bpm
- **Oxygen Saturation (SpO2)**: {patient_data.get('spo2', 'N/A')}%
- **Body Mass Index (BMI)**: {patient_data.get('bmi', 'N/A')} kg/m²
- **Fasting Blood Glucose**: {patient_data.get('fasting_glucose', 'N/A')} mg/dL (HbA1c: {patient_data.get('hba1c', 'N/A')}%)
- **Serum Lipids**: Total Chol: {patient_data.get('cholesterol', 'N/A')} | LDL: {patient_data.get('ldl', 'N/A')} | HDL: {patient_data.get('hdl', 'N/A')} mg/dL
- **Lifestyle**: Smoking: {'Yes' if patient_data.get('smoking_status') == 1 else 'No'} | Physical Activity: {patient_data.get('physical_activity_min', 'N/A')} min/week | Sleep: {patient_data.get('sleep_hours', 'N/A')} hrs/night

---

### 2. MODEL RISK STRATIFICATION
- **Composite Hazard Score**: **{risk_results.get('composite_hazard_percent')}%**
- **Project-Defined Risk Tier**: **{risk_results.get('risk_tier')}**
- **Sub-Hazard Estimates**:
  - Cardiovascular Risk Estimate: {risk_results.get('sub_hazards', {}).get('cardiovascular_risk_percent')}%
  - Diabetes / Metabolic Risk Estimate: {risk_results.get('sub_hazards', {}).get('diabetes_metabolic_risk_percent')}%
  - Hypertensive Hazard Estimate: {risk_results.get('sub_hazards', {}).get('hypertensive_stroke_risk_percent')}%
  - Multi-Morbidity Compounding Synergy Active: {'Yes (Heuristic supervisory multiplier gamma=1.25)' if risk_results.get('sub_hazards', {}).get('synergistic_multiplier_active') else 'No'}
- **Mathematical Derivation & Probability Coherence**:
  Composite hazard is derived from the weighted linear combination of domain channels, compounded by the heuristic synergy multiplier (gamma = 1.25 when both channels > 40%), and bounded below by the axiomatic probability floor: P(Composite) >= max(P_cvd, P_diab). This prevents the composite score from dropping below its individual components.

---

### 3. EXPLAINABLE AI (SHAP) ATTRIBUTION ANALYSIS
{shap_summary.get('clinical_summary', 'Homeostatic profile.')}

**Top Risk Drivers (Increasing Model Risk)**:
"""
    for d in shap_summary.get("top_risk_drivers", [])[:4]:
        summary += f"- **{d['label']}** ({d['patient_value']}): +{d['shap_value']:.3f} SHAP contribution\n"

    summary += "\n**Top Protective Factors (Buffering Model Risk)**:\n"
    for p in shap_summary.get("top_protective_factors", [])[:3]:
        summary += f"- **{p['label']}** ({p['patient_value']}): {p['shap_value']:.3f} SHAP contribution\n"

    target_reached = recourse_results.get("target_reached", False)
    achieved_risk = recourse_results.get("achieved_risk_percent", "N/A")
    achieved_tier = recourse_results.get("achieved_tier", "High Risk")
    initial_risk = recourse_results.get("initial_risk_percent", "N/A")

    if target_reached:
        recourse_section = f"""---

### 4. MODEL-GENERATED HYPOTHETICAL RECOURSE SCENARIOS
*Algorithmic minimal-effort hypothetical adjustments to reach target tier (<= 25%):*

- **Feasibility Status**: **Target Achieved**
- **Achieved Projected Risk**: **{achieved_risk}%** ({achieved_tier}) [Initial: {initial_risk}%]
- **Hypothetical Parameter Adjustments**:
"""
        if recourse_results.get("prescriptions"):
            for rx in recourse_results["prescriptions"]:
                recourse_section += f"- {rx['action']}\n"
        else:
            recourse_section += "- Current model risk estimate was already within or below target bounds.\n"
    else:
        recourse_section = f"""---

### 4. MODEL-GENERATED HYPOTHETICAL RECOURSE SCENARIOS
*Algorithmic minimal-effort hypothetical adjustments to reach target tier (<= 25%):*

- **Feasibility Status**: **No feasible path exists to reach target tier (<= 25.0%) within clinically safe bounds for modifiable parameters.**
- **Nearest Achievable Model Risk**: **{achieved_risk}%** ({achieved_tier}) [Initial: {initial_risk}%]
- **Dominance of Locked Variables**: Risk de-escalation is heavily constrained by locked/unmodifiable clinical variables (such as age, biological sex, genetic family history, and established chronic end-organ disease). Even with maximal hypothetical optimization of modifiable parameters, residual model risk remains elevated.
- **Hypothetical Adjustments Toward Nearest Achievable Risk**:
"""
        if recourse_results.get("prescriptions"):
            for rx in recourse_results["prescriptions"]:
                recourse_section += f"- {rx['action']}\n"
        else:
            recourse_section += "- Modifiable variables are fully saturated within safe physiological limits; locked clinical variables dominate risk score.\n"

    summary += recourse_section

    summary += f"""
---

### 5. DECISION-SUPPORT INFORMATION
**Information Summary**: {risk_results.get('clinical_recommendation', 'Routine health maintenance.')}
"""
    return summary


def generate_patient_record_pdf(
    patient_id: str,
    patient_name: str,
    patient_data: Dict[str, Any],
    risk_results: Dict[str, Any],
    shap_summary: Dict[str, Any],
    recourse_results: Dict[str, Any],
) -> bytes:
    """Build a downloadable PDF snapshot of a patient record and model outputs.

    This is a research/demo report, not a medical certificate or diagnostic document.
    The function returns PDF bytes so Streamlit can offer a download without writing
    sensitive patient information to a persistent server-side file.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Smart AI Health Patient Report - {patient_id}",
        author="Smart AI Health",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=19, leading=23, textColor=colors.HexColor("#0f4c5c"),
        alignment=TA_LEFT, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", parent=styles["Heading2"], fontSize=11,
        leading=14, textColor=colors.HexColor("#0f4c5c"),
        spaceBefore=9, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="SmallMuted", parent=styles["BodyText"], fontSize=8,
        leading=10, textColor=colors.HexColor("#52616b"),
    ))
    styles.add(ParagraphStyle(
        name="Disclaimer", parent=styles["BodyText"], fontSize=8,
        leading=10, textColor=colors.HexColor("#7c4a03"),
        backColor=colors.HexColor("#fff7e6"), borderColor=colors.HexColor("#f1c46b"),
        borderWidth=0.5, borderPadding=7, spaceBefore=5, spaceAfter=8,
    ))

    def val(key, default="Not recorded"):
        value = patient_data.get(key, default)
        if value is None or value == "":
            return default
        return str(value)

    def para(text, style="BodyText"):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    def table(rows, widths=None):
        prepared = [[para(cell, "SmallMuted" if r else "BodyText") for cell in row]
                    for r, row in enumerate(rows)]
        t = Table(prepared, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf3f5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#12343b")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d7e1e5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbfc")]),
        ]))
        return t

    generated_at = datetime.now().strftime("%d %b %Y, %H:%M")
    sub = risk_results.get("sub_hazards", {})
    story = [
        para("Smart AI Health", "ReportTitle"),
        para("Patient record and clinical decision-support summary", "Heading3"),
        Spacer(1, 3 * mm),
        table([
            ["Patient", "Patient ID", "Report generated"],
            [patient_name, patient_id, generated_at],
        ], widths=[65 * mm, 45 * mm, 55 * mm]),
        Spacer(1, 2 * mm),
        para(DISCLAIMER_TEXT, "Disclaimer"),
        para("1. Demographics and measurements", "SectionHeading"),
        table([
            ["Measurement", "Recorded value", "Measurement", "Recorded value"],
            ["Age", f"{val('age')} years", "Biological sex", "Male" if patient_data.get("sex") == 1 else "Female"],
            ["Blood pressure", f"{val('systolic_bp')}/{val('diastolic_bp')} mmHg", "Resting heart rate", f"{val('resting_hr')} bpm"],
            ["Oxygen saturation", f"{val('spo2')}%", "BMI", f"{val('bmi')} kg/m²"],
            ["Fasting glucose", f"{val('fasting_glucose')} mg/dL", "HbA1c", f"{val('hba1c')}%"],
            ["Total cholesterol", f"{val('cholesterol')} mg/dL", "LDL / HDL", f"{val('ldl')} / {val('hdl')} mg/dL"],
            ["Smoking status", "Active smoker" if patient_data.get("smoking_status") == 1 else "Non-smoker", "Physical activity", f"{val('physical_activity_min')} min/week"],
            ["Sleep", f"{val('sleep_hours')} hours/night", "Diet score", f"{val('diet_score')} / 10"],
        ], widths=[35 * mm, 50 * mm, 35 * mm, 45 * mm]),
        para("2. Model risk estimates", "SectionHeading"),
        table([
            ["Model output", "Estimate"],
            ["Composite model risk score", f"{risk_results.get('composite_hazard_percent', 'N/A')}%"],
            ["Project-defined risk tier", risk_results.get("risk_tier", "Not available")],
            ["Cardiovascular model estimate", f"{sub.get('cardiovascular_risk_percent', 'N/A')}%"],
            ["Diabetes / metabolic model estimate", f"{sub.get('diabetes_metabolic_risk_percent', 'N/A')}%"],
            ["Hypertensive hazard heuristic", f"{sub.get('hypertensive_stroke_risk_percent', 'N/A')}%"],
            ["Cross-domain synergy active", "Yes" if sub.get("synergistic_multiplier_active") else "No"],
        ], widths=[85 * mm, 80 * mm]),
        para("Model outputs are statistical estimates from a research prototype. Risk tiers are project-defined and are not a clinical diagnosis.", "SmallMuted"),
        para("3. Explanation highlights", "SectionHeading"),
        para(shap_summary.get("clinical_summary", "No explanation summary was available.")),
    ]

    drivers = shap_summary.get("top_risk_drivers", [])[:5]
    if drivers:
        story.extend([Spacer(1, 2 * mm), table(
            [["Feature", "Patient value", "Model contribution"]] + [
                [d.get("label", d.get("feature", "Feature")),
                 f"{d.get('patient_value', 'N/A')} {d.get('unit', '')}".strip(),
                 f"{float(d.get('shap_value', 0.0)):+.3f} SHAP value"]
                for d in drivers
            ], widths=[65 * mm, 45 * mm, 55 * mm]
        )])

    target_reached = recourse_results.get("target_reached", False)
    status_msg = "Target tier (<= 25%) achieved" if target_reached else f"Target infeasible (<= 25%); Nearest achievable risk: {recourse_results.get('achieved_risk_percent', 'N/A')}% (Constrained by locked variables)"

    story.extend([
        para("4. Hypothetical what-if summary", "SectionHeading"),
        para(f"Initial model risk: {recourse_results.get('initial_risk_percent', 'N/A')}%"),
        para(f"Feasibility status: {status_msg}"),
        para(f"Projected model risk: {recourse_results.get('achieved_risk_percent', 'N/A')}% ({recourse_results.get('achieved_tier', 'N/A')})"),
        para("These changes are algorithm-generated simulations only. They do not alter the patient record and are not treatment recommendations.", "SmallMuted"),
    ])
    prescriptions = recourse_results.get("prescriptions", [])
    if prescriptions:
        story.append(Spacer(1, 2 * mm))
        story.append(table([["Hypothetical adjustment", "Current", "Illustrative target"]] + [
            [rx.get("action", "Adjustment"), str(rx.get("current_value", "N/A")), str(rx.get("target_value", "N/A"))]
            for rx in prescriptions[:8]
        ], widths=[80 * mm, 40 * mm, 45 * mm]))

    story.extend([
        Spacer(1, 5 * mm),
        para(f"Clinical decision-support note: {risk_results.get('clinical_recommendation', 'No summary available.')}", "BodyText"),
        Spacer(1, 3 * mm),
        para("Generated by Smart AI Health. Demonstration data may be synthetic. Verify all values against an authorized clinical record before use.", "SmallMuted"),
    ])

    def draw_page(canvas, doc_obj):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d7e1e5"))
        canvas.line(17 * mm, 12 * mm, A4[0] - 17 * mm, 12 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#52616b"))
        canvas.drawString(17 * mm, 7 * mm, "Smart AI Health • Research and educational prototype")
        canvas.drawRightString(A4[0] - 17 * mm, 7 * mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return buffer.getvalue()
