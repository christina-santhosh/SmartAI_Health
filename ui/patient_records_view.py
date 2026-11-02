"""
Patient Records & Structured Input Form View.
Organized into 4 clinical sections:
A. Demographics (Immutable attributes)
B. Vital Signs (Hemodynamics & oxygenation)
C. Laboratory Measurements (Metabolic & lipid panels)
D. Lifestyle & Medical History (Behaviors, symptoms, genetics)
Includes validation, range auditing, and Save/Cancel/Review workflow.
"""

import streamlit as st
from datetime import datetime
from ui.components import render_patient_summary_card
from backend.patient_repository import validate_patient_vitals, save_patient_profile, generate_patient_history

def render_patient_records(patient_id, patient_name, patient_data, read_only=False):
    privacy_mode = st.session_state.get("privacy_mode", False)
    render_patient_summary_card(patient_id, patient_name, patient_data, privacy_mode=privacy_mode)
    
    st.markdown("### 📋 Patient Clinical Record & Intake")
    if read_only:
        st.caption("Read-only view of the measurements currently saved for your linked profile.")
        rows = [
            ("Demographics", [("Age", f"{patient_data.get('age', '—')} years"), ("Biological sex", "Male" if patient_data.get("sex") == 1 else "Female"), ("BMI", f"{patient_data.get('bmi', '—')} kg/m²")]),
            ("Vitals", [("Blood pressure", f"{patient_data.get('systolic_bp', '—')}/{patient_data.get('diastolic_bp', '—')} mmHg"), ("Resting heart rate", f"{patient_data.get('resting_hr', '—')} bpm"), ("SpO₂", f"{patient_data.get('spo2', '—')}%")]),
            ("Laboratory", [("Fasting glucose", f"{patient_data.get('fasting_glucose', '—')} mg/dL"), ("HbA1c", f"{patient_data.get('hba1c', '—')}%"), ("Total cholesterol", f"{patient_data.get('cholesterol', '—')} mg/dL"), ("HDL / LDL", f"{patient_data.get('hdl', '—')} / {patient_data.get('ldl', '—')} mg/dL")]),
            ("Lifestyle & history", [("Smoking", "Yes" if patient_data.get("smoking_status") == 1 else "No"), ("Physical activity", f"{patient_data.get('physical_activity_min', '—')} min/week"), ("Sleep", f"{patient_data.get('sleep_hours', '—')} hours/night"), ("Family history CVD", "Yes" if patient_data.get("family_history_cvd") == 1 else "No"), ("Family history diabetes", "Yes" if patient_data.get("family_history_diabetes") == 1 else "No")]),
        ]
        for section, values in rows:
            st.markdown(f"#### {section}")
            cols = st.columns(min(4, len(values)))
            for i, (label, value) in enumerate(values):
                cols[i % len(cols)].metric(label, value)
        st.info("Only a doctor account can edit this prototype record. If any displayed measurement looks wrong, contact the care team; do not rely on this demo for clinical decisions.")
        return
    st.caption("Update clinical parameters, record new vital observations, and validate measurements against physiological reference intervals.")
    
    # Initialize editable form state if not present
    if "form_patient_data" not in st.session_state or st.session_state.get("form_patient_id") != patient_id:
        st.session_state.form_patient_data = dict(patient_data)
        st.session_state.form_patient_id = patient_id

    form_data = st.session_state.form_patient_data
    # Widget keys include patient ID so Streamlit cannot reuse one patient's unsaved values for another.
    widget_key = lambda field: f"patient_{patient_id}_{field}"

    with st.form("patient_clinical_form", clear_on_submit=False):
        # Section A: Demographics
        st.markdown("#### A. Patient Demographics & Identification")
        st.caption("Immutable demographic variables utilized for risk stratification equations.")
        col_demo1, col_demo2, col_demo3 = st.columns(3)
        with col_demo1:
            in_id = st.text_input("Patient ID", value=str(patient_id), disabled=True, key=widget_key("id"))
        with col_demo2:
            in_age = st.number_input("Age (years)", min_value=18, max_value=105, value=int(form_data.get("age", 45)), step=1, key=widget_key("age"))
        with col_demo3:
            sex_val = int(form_data.get("sex", 1))
            in_sex = st.selectbox("Biological Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=sex_val, key=widget_key("sex"))

        st.markdown("---")

        # Section B: Vital Signs
        st.markdown("#### B. Vital Signs & Hemodynamics")
        st.caption("Physiological telemetry from clinical check-in or calibrated wearable devices.")
        col_vit1, col_vit2, col_vit3, col_vit4 = st.columns(4)
        with col_vit1:
            in_sbp = st.number_input("Systolic BP (mmHg)", min_value=80.0, max_value=240.0, value=float(form_data.get("systolic_bp", 120.0)), step=1.0, key=widget_key("systolic_bp"))
        with col_vit2:
            in_dbp = st.number_input("Diastolic BP (mmHg)", min_value=50.0, max_value=140.0, value=float(form_data.get("diastolic_bp", 80.0)), step=1.0, key=widget_key("diastolic_bp"))
        with col_vit3:
            in_hr = st.number_input("Resting Heart Rate (bpm)", min_value=40.0, max_value=180.0, value=float(form_data.get("resting_hr", 72.0)), step=1.0, key=widget_key("resting_hr"))
        with col_vit4:
            in_spo2 = st.number_input("Oxygen Saturation - SpO2 (%)", min_value=75.0, max_value=100.0, value=float(form_data.get("spo2", 98.0)), step=0.5, key=widget_key("spo2"))

        st.markdown("---")

        # Section C: Laboratory Measurements
        st.markdown("#### C. Laboratory Biomarkers & Serum Chemistry")
        st.caption("Standard clinical laboratory blood chemistry assays.")
        col_lab1, col_lab2, col_lab3 = st.columns(3)
        with col_lab1:
            in_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=60.0, max_value=350.0, value=float(form_data.get("fasting_glucose", 90.0)), step=1.0, key=widget_key("fasting_glucose"))
            in_hba1c = st.number_input("Glycated Hemoglobin - HbA1c (%)", min_value=4.0, max_value=16.0, value=float(form_data.get("hba1c", 5.4)), step=0.1, key=widget_key("hba1c"))
        with col_lab2:
            in_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=float(form_data.get("cholesterol", 190.0)), step=1.0, key=widget_key("cholesterol"))
            in_hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=15.0, max_value=120.0, value=float(form_data.get("hdl", 50.0)), step=1.0, key=widget_key("hdl"))
        with col_lab3:
            in_ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=40.0, max_value=300.0, value=float(form_data.get("ldl", 110.0)), step=1.0, key=widget_key("ldl"))
            in_bmi = st.number_input("Body Mass Index - BMI (kg/m²)", min_value=15.0, max_value=55.0, value=float(form_data.get("bmi", 25.0)), step=0.1, key=widget_key("bmi"))

        st.markdown("---")

        # Section D: Lifestyle, Symptoms & Medical History
        st.markdown("#### D. Lifestyle Behaviors, Symptoms & Family History")
        col_life1, col_life2, col_life3 = st.columns(3)
        with col_life1:
            smoke_val = int(form_data.get("smoking_status", 0))
            in_smoking = st.selectbox("Tobacco Smoking Status", options=[0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Active Smoker", index=smoke_val, key=widget_key("smoking_status"))
            in_activity = st.number_input("Physical Activity (min/week)", min_value=0, max_value=600, value=int(form_data.get("physical_activity_min", 120)), step=15, key=widget_key("physical_activity_min"))
        with col_life2:
            in_sleep = st.number_input("Nightly Sleep Duration (hours)", min_value=3.0, max_value=12.0, value=float(form_data.get("sleep_hours", 7.0)), step=0.5, key=widget_key("sleep_hours"))
            in_diet = st.slider("Dietary Quality Score (1-10)", min_value=1, max_value=10, value=int(form_data.get("diet_score", 6)), key=widget_key("diet_score"))
        with col_life3:
            in_fam_cvd = st.checkbox("Family History of Cardiovascular Disease", value=bool(form_data.get("family_history_cvd", 0)), key=widget_key("family_history_cvd"))
            in_fam_diab = st.checkbox("Family History of Diabetes Mellitus", value=bool(form_data.get("family_history_diabetes", 0)), key=widget_key("family_history_diabetes"))
            in_cp = st.checkbox("Exertional Chest Discomfort / Angina", value=bool(form_data.get("chest_pain", 0)), key=widget_key("chest_pain"))
            in_sob = st.checkbox("Dyspnea / Exertional Shortness of Breath", value=bool(form_data.get("shortness_of_breath", 0)), key=widget_key("shortness_of_breath"))

        st.markdown("---")
        
        # Form submission buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
        with btn_col1:
            submitted = st.form_submit_button("💾 Save Changes", use_container_width=True)
        with btn_col2:
            reset = st.form_submit_button("🔄 Reset to Default", use_container_width=True)

    if submitted:
        candidate_dict = {
            "age": in_age, "sex": in_sex, "bmi": in_bmi,
            "systolic_bp": in_sbp, "diastolic_bp": in_dbp, "resting_hr": in_hr, "spo2": in_spo2,
            "cholesterol": in_chol, "hdl": in_hdl, "ldl": in_ldl, "fasting_glucose": in_glucose,
            "hba1c": in_hba1c, "insulin": float(form_data.get("insulin", 12.0)),
            "smoking_status": in_smoking, "physical_activity_min": in_activity, "sleep_hours": in_sleep,
            "diet_score": in_diet,
            "family_history_cvd": 1 if in_fam_cvd else 0,
            "family_history_diabetes": 1 if in_fam_diab else 0,
            "chest_pain": 1 if in_cp else 0,
            "shortness_of_breath": 1 if in_sob else 0
        }
        
        is_valid, errors, warnings = validate_patient_vitals(candidate_dict)
        if not is_valid:
            for err in errors:
                st.error(f"❌ Validation Error: {err}")
        else:
            if warnings:
                for warn in warnings:
                    st.warning(f"⚠️ Clinical Warning: {warn}")
            
            st.session_state.current_patient = candidate_dict
            st.session_state.form_patient_data = dict(candidate_dict)
            existing_profile = {
                "id": str(patient_id),
                "name": str(patient_name),
                "scenario": st.session_state.get("scenario", "user_updated"),
                "description": "Updated local patient record",
                "data": dict(candidate_dict),
            }
            try:
                save_patient_profile(existing_profile)
                st.session_state.history_df = generate_patient_history(
                    candidate_dict,
                    scenario=st.session_state.get("scenario", "stable_healthy"),
                    n_days=45,
                    n_logs=24,
                )
                st.session_state.history_patient_id = str(patient_id)
                st.success("✅ Record validated and saved to the local patient registry.")
            except (OSError, ValueError) as exc:
                st.error(f"The record updated in this session but could not be persisted: {exc}")
            st.rerun()

    if reset:
        st.session_state.form_patient_data = dict(st.session_state.current_patient)
        for key in list(st.session_state.keys()):
            if str(key).startswith(f"patient_{patient_id}_"):
                st.session_state.pop(key, None)
        st.info("Form reset to current saved patient profile.")
        st.rerun()
