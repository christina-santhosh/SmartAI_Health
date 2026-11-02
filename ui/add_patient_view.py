"""Doctor-only patient registration form for the local demonstration registry."""
import streamlit as st
from backend.patient_repository import save_patient_profile, validate_patient_vitals, generate_patient_history

DEFAULTS = {
    "age": 45, "sex": 0, "bmi": 25.0, "systolic_bp": 120.0, "diastolic_bp": 80.0,
    "resting_hr": 72.0, "spo2": 98.0, "fasting_glucose": 95.0, "hba1c": 5.5,
    "insulin": 12.0, "cholesterol": 190.0, "hdl": 50.0, "ldl": 110.0,
    "smoking_status": 0, "physical_activity_min": 120, "sleep_hours": 7.0,
    "diet_score": 6, "family_history_cvd": 0, "family_history_diabetes": 0,
    "chest_pain": 0, "shortness_of_breath": 0,
}


def render_add_patient_view(existing_profiles):
    st.markdown("## Add a patient")
    st.caption("Create a local demonstration record. Enter verified measurements only; this prototype is not an EHR.")
    with st.form("add_patient_registration", clear_on_submit=False):
        name = st.text_input("Patient full name", placeholder="Enter patient name")
        a, b, c = st.columns(3)
        with a:
            age = st.number_input("Age (years)", 18, 110, 45)
            sex = st.selectbox("Biological sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            bmi = st.number_input("BMI (kg/m²)", 12.0, 65.0, 25.0, step=0.1)
        with b:
            sbp = st.number_input("Systolic BP (mmHg)", 70.0, 260.0, 120.0, step=1.0)
            dbp = st.number_input("Diastolic BP (mmHg)", 40.0, 160.0, 80.0, step=1.0)
            hr = st.number_input("Resting heart rate (bpm)", 35.0, 200.0, 72.0, step=1.0)
            spo2 = st.number_input("SpO₂ (%)", 70.0, 100.0, 98.0, step=0.5)
        with c:
            glucose = st.number_input("Fasting glucose (mg/dL)", 40.0, 500.0, 95.0, step=1.0)
            hba1c = st.number_input("HbA1c (%)", 3.5, 18.0, 5.5, step=0.1)
            cholesterol = st.number_input("Total cholesterol (mg/dL)", 80.0, 500.0, 190.0, step=1.0)
            hdl = st.number_input("HDL (mg/dL)", 10.0, 150.0, 50.0, step=1.0)
            ldl = st.number_input("LDL (mg/dL)", 30.0, 350.0, 110.0, step=1.0)
        st.markdown("#### Lifestyle and history")
        x, y, z = st.columns(3)
        with x:
            smoking = st.selectbox("Smoking", [0, 1], format_func=lambda v: "Non-smoker" if v == 0 else "Smoker")
            activity = st.number_input("Physical activity (min/week)", 0, 1200, 120, step=15)
        with y:
            sleep = st.number_input("Sleep (hours/night)", 2.0, 16.0, 7.0, step=0.5)
            diet = st.slider("Diet score (1–10)", 1, 10, 6)
        with z:
            fam_cvd = st.checkbox("Family history of cardiovascular disease")
            fam_diabetes = st.checkbox("Family history of diabetes")
            chest_pain = st.checkbox("Chest pain / angina symptom")
            sob = st.checkbox("Shortness of breath symptom")
        submitted = st.form_submit_button("Save new patient", type="primary", use_container_width=True)

    if submitted:
        clean_name = name.strip()
        if not clean_name:
            st.error("Enter the patient's name.")
            return
        data = {
            **DEFAULTS, "age": int(age), "sex": int(sex), "bmi": float(bmi),
            "systolic_bp": float(sbp), "diastolic_bp": float(dbp), "resting_hr": float(hr), "spo2": float(spo2),
            "fasting_glucose": float(glucose), "hba1c": float(hba1c), "cholesterol": float(cholesterol),
            "hdl": float(hdl), "ldl": float(ldl), "smoking_status": int(smoking),
            "physical_activity_min": int(activity), "sleep_hours": float(sleep), "diet_score": int(diet),
            "family_history_cvd": int(fam_cvd), "family_history_diabetes": int(fam_diabetes),
            "chest_pain": int(chest_pain), "shortness_of_breath": int(sob),
        }
        valid, errors, warnings = validate_patient_vitals(data)
        if not valid:
            for error in errors:
                st.error(error)
            return
        if warnings:
            for warning in warnings:
                st.warning(warning)
        existing_ids = {str(p.get("id", "")) for p in existing_profiles}
        number = 1
        while f"PT-{600 + number:03d}" in existing_ids:
            number += 1
        patient_id = f"PT-{600 + number:03d}"
        profile = {"id": patient_id, "name": clean_name, "scenario": "user_added", "description": "User-added local patient record", "data": data}
        try:
            save_patient_profile(profile)
        except (OSError, ValueError) as exc:
            st.error(f"Could not save the patient record: {exc}")
            return
        st.session_state.current_patient = dict(data)
        st.session_state.patient_id = patient_id
        st.session_state.patient_name = clean_name
        st.session_state.scenario = "user_added"
        st.session_state.history_df = generate_patient_history(data, scenario="stable_healthy", n_days=45, n_logs=24)
        st.session_state.history_patient_id = patient_id
        st.session_state.pending_active_patient_id = patient_id
        st.session_state.pending_nav_selection = "📊 Dashboard"
        for key in ("form_patient_data", "form_patient_id"):
            st.session_state.pop(key, None)
        st.success(f"Saved {clean_name} as {patient_id}. The record is stored in data/patient_registry.json.")
        st.rerun()
