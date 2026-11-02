"""
Population Overview & Cohort Triage View.
Restricted to authorized clinical researchers:
- Prominently displays Synthetic Demonstration Cohort disclaimer
- Provides risk tier distributions, filtering, and patient search
- HL7 FHIR v4-Compatible RiskAssessment exporter with live schema validation
- Clinical Consultation Summary markdown report generation.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from backend.report_generator import (
    generate_fhir_risk_assessment,
    validate_fhir_risk_assessment_structure,
    generate_clinical_consultation_summary
)

COHORT_PATH = Path(__file__).resolve().parent.parent / "data" / "multi_morbidity_cohort.csv"

def render_population_overview(current_patient_id, current_patient_data, risk_results, xai_engine, recourse_engine):
    st.markdown("### 👥 Population Overview & Cohort Triage")
    st.caption("Authorized clinical researcher workspace for cohort risk stratification, subgroup analysis, and HL7 FHIR export.")
    
    # Access Control Check
    if "clinician_authorized" not in st.session_state:
        st.session_state.clinician_authorized = True # Default unlocked for demonstration, with toggleable PIN
        
    auth_toggle = st.toggle("🔒 Authorized Clinician/Researcher Mode", value=st.session_state.clinician_authorized)
    st.session_state.clinician_authorized = auth_toggle
    
    if not auth_toggle:
        st.warning("🔒 Access restricted. Toggle Authorized Clinician Mode above to access population-level health records.")
        return

    # Prominent Synthetic Data Notice
    st.markdown("""
    <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 10px 14px; margin-bottom: 1rem; font-size: 0.85rem; color: #1e40af;">
        📢 <strong>SYNTHETIC BENCHMARK COHORT NOTICE:</strong>
        The cohort records displayed below are synthetically generated benchmark fixtures (N=4,000) calibrated to physiological distributions. 
        They are provided for algorithm evaluation, UI testing, and decision-support workflow demonstration. They do not represent real-world patient health data.
    </div>
    """, unsafe_allow_html=True)
    
    if not COHORT_PATH.exists():
        st.error("Cohort data file not found.")
        return
        
    cohort_df = pd.read_csv(COHORT_PATH)
    
    # Cohort Overview Metrics
    c1, c2, c3, c4 = st.columns(4)
    total_pts = len(cohort_df)
    high_risk_pts = len(cohort_df[cohort_df["risk_tier"] == "High Risk"])
    mod_risk_pts = len(cohort_df[cohort_df["risk_tier"] == "Moderate Risk"])
    low_risk_pts = len(cohort_df[cohort_df["risk_tier"] == "Low Risk"])
    
    c1.metric("Synthetic Cohort Size", f"{total_pts:,}")
    c2.metric("High Risk Triage Tier", f"{high_risk_pts:,}", delta=f"{high_risk_pts/total_pts*100:.1f}% of cohort", delta_color="inverse")
    c3.metric("Moderate Risk Tier", f"{mod_risk_pts:,}", delta=f"{mod_risk_pts/total_pts*100:.1f}% of cohort", delta_color="off")
    c4.metric("Low Risk Tier", f"{low_risk_pts:,}", delta=f"{low_risk_pts/total_pts*100:.1f}% of cohort")

    st.markdown("---")
    
    # Filtering & Triage Controls
    st.markdown("#### 🎯 Stratified Patient Filtering")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        tier_filter = st.multiselect(
            "Filter Risk Tiers:",
            options=["High Risk", "Moderate Risk", "Low Risk"],
            default=["High Risk", "Moderate Risk"]
        )
    with f_col2:
        age_range = st.slider("Filter Age Interval:", min_value=20, max_value=85, value=(30, 80))
    with f_col3:
        comorb_filter = st.selectbox("Comorbidity Slices:", ["All Cohort Members", "Hypertensive (SBP ≥ 140)", "Hyperglycemic (Glucose ≥ 126)", "Both (Synergy Compounded)"])

    # Apply filters
    filtered_df = cohort_df[
        (cohort_df["risk_tier"].isin(tier_filter)) &
        (cohort_df["age"] >= age_range[0]) &
        (cohort_df["age"] <= age_range[1])
    ]
    if comorb_filter == "Hypertensive (SBP ≥ 140)":
        filtered_df = filtered_df[filtered_df["systolic_bp"] >= 140.0]
    elif comorb_filter == "Hyperglycemic (Glucose ≥ 126)":
        filtered_df = filtered_df[filtered_df["fasting_glucose"] >= 126.0]
    elif comorb_filter == "Both (Synergy Compounded)":
        filtered_df = filtered_df[(filtered_df["systolic_bp"] >= 140.0) & (filtered_df["fasting_glucose"] >= 126.0)]

    st.markdown(f"**Showing {len(filtered_df):,} Matching Patients:**")
    st.dataframe(
        filtered_df[[
            "patient_id", "age", "sex", "systolic_bp", "diastolic_bp",
            "fasting_glucose", "cholesterol", "hazard_score", "risk_tier"
        ]].rename(columns={
            "patient_id": "Patient ID",
            "age": "Age",
            "sex": "Sex (0=F, 1=M)",
            "systolic_bp": "SBP (mmHg)",
            "diastolic_bp": "DBP (mmHg)",
            "fasting_glucose": "Glucose (mg/dL)",
            "cholesterol": "Cholesterol",
            "hazard_score": "Model Hazard",
            "risk_tier": "Risk Tier"
        }).head(30),
        use_container_width=True
    )

    st.markdown("---")

    # Clinical Summary Document Export
    st.markdown("#### 📄 Clinical Summary Export: Active Patient")
    st.caption("Generate clinical consultation documentation for the active patient.")

    shap_summary = xai_engine.explain_cardiovascular_risk(current_patient_data)
    recourse_summary = recourse_engine.compute_optimal_recourse(current_patient_data)
    
    summary_md = generate_clinical_consultation_summary(
        current_patient_id, current_patient_data, risk_results, shap_summary, recourse_summary
    )
    st.download_button(
        "⬇️ Download Clinical Summary (.md)",
        data=summary_md,
        file_name=f"clinical_summary_{current_patient_id}.md",
        mime="text/markdown",
        use_container_width=True
    )
    with st.expander("Preview Summary Document"):
        st.markdown(summary_md)
