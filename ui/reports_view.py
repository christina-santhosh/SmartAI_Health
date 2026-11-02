"""Downloadable patient reports for the Smart AI Health prototype."""

import json
import re
from datetime import datetime

import streamlit as st

from backend.report_generator import (
    DISCLAIMER_TEXT,
    generate_clinical_consultation_summary,
    generate_fhir_risk_assessment,
    generate_patient_record_pdf,
    validate_fhir_risk_assessment_structure,
)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "patient"


def render_reports_view(
    patient_id,
    patient_name,
    patient_data,
    risk_results,
    xai_engine,
    recourse_engine,
):
    """Render PDF, Markdown and FHIR exports for the active patient."""
    privacy_mode = st.session_state.get("privacy_mode", False)
    display_name = f"Patient #{patient_id}" if privacy_mode else patient_name

    st.markdown("### Reports & Exports")
    st.caption("Generate a point-in-time report from the active demonstration patient record.")
    st.info(DISCLAIMER_TEXT, icon="ℹ️")

    shap_summary = xai_engine.explain_cardiovascular_risk(patient_data)
    recourse_summary = recourse_engine.compute_optimal_recourse(patient_data)
    fhir_obj = generate_fhir_risk_assessment(
        str(patient_id), patient_data, risk_results, shap_summary
    )
    fhir_valid, fhir_errors = validate_fhir_risk_assessment_structure(fhir_obj)
    summary_md = generate_clinical_consultation_summary(
        str(patient_id), patient_data, risk_results, shap_summary, recourse_summary
    )
    pdf_bytes = generate_patient_record_pdf(
        patient_id=str(patient_id),
        patient_name=display_name,
        patient_data=patient_data,
        risk_results=risk_results,
        shap_summary=shap_summary,
        recourse_results=recourse_summary,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    stem = f"smart_ai_health_{_safe_filename(patient_id)}_{stamp}"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Patient report")
        st.write("A formatted PDF with patient measurements, model estimates, explanation highlights and disclaimers.")
        st.download_button(
            "Download patient report (PDF)",
            data=pdf_bytes,
            file_name=f"{stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )
    with c2:
        st.markdown("#### Consultation summary")
        st.write("A plain-text Markdown report that is easy to archive or edit.")
        st.download_button(
            "Download summary (MD)",
            data=summary_md,
            file_name=f"{stem}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### Report preview")
    with st.expander("View the consultation summary", expanded=False):
        st.markdown(summary_md)
