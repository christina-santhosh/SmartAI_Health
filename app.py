"""Smart AI Health: role-based clinical decision-support prototype."""
import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import joblib

from backend.risk_predictor import CrossMorbidityRiskPredictor
from backend.baseline_tracker import SparseTemporalBaselineTracker
from backend.xai_shap import ClinicalExplainabilityEngine
from backend.counterfactual_engine import CounterfactualRecourseEngine
from backend.patient_repository import load_patient_profiles, generate_patient_history
from ui.components import apply_healthcare_theme, render_safety_banner
from ui.dashboard_view import render_dashboard
from ui.patient_records_view import render_patient_records
from ui.health_trends_view import render_health_trends
from ui.risk_insights_view import render_risk_insights
from ui.what_if_view import render_what_if_scenarios
from ui.population_view import render_population_overview
from ui.settings_view import render_settings
from ui.reports_view import render_reports_view
from ui.worklist_view import render_patient_worklist
from ui.add_patient_view import render_add_patient_view
from ui.data_ml_view import render_data_ml_view

st.set_page_config(page_title="Smart AI Health", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
apply_healthcare_theme()

@st.cache_resource
def get_system_engines():
    risk_pred = CrossMorbidityRiskPredictor(use_calibrated=True)
    baseline_tr = SparseTemporalBaselineTracker(tau_days=21.0, z_threshold=2.0)
    xai_eng = ClinicalExplainabilityEngine(use_calibrated=False)
    recourse_eng = CounterfactualRecourseEngine(risk_pred)
    return risk_pred, baseline_tr, xai_eng, recourse_eng

@st.cache_data
def get_training_metadata():
    meta_path = ROOT_DIR / "models" / "training_summary.pkl"
    if meta_path.exists():
        try:
            return joblib.load(meta_path)
        except Exception:
            return {}
    return {}


def _authenticate(role, username, password):
    """Demo-only credential check. Use a real identity provider before production."""
    if role == "Doctor":
        expected_user = os.getenv("SMARTAI_DOCTOR_USERNAME", "doctor")
        expected_pass = os.getenv("SMARTAI_DOCTOR_PASSWORD", "doctor123")
        return username.strip() == expected_user and password == expected_pass
    expected_user = os.getenv("SMARTAI_PATIENT_USERNAME", "patient")
    expected_pass = os.getenv("SMARTAI_PATIENT_PASSWORD", "patient123")
    return username.strip() == expected_user and password == expected_pass


def _login_screen():
    st.markdown("<div style='padding: 2rem 0 1rem 0;'><span style='font-size:2.4rem'>🩺</span><h1 style='margin:0'>Smart AI Health</h1><p style='color:#64748b'>Clinical decision-support & patient monitoring prototype</p></div>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Welcome back")
        st.write("Choose your access type to continue to the appropriate workspace.")
        st.markdown("- **Doctor:** patient worklist, record management, model insights, population and dataset information.")
        st.markdown("- **Patient / Caregiver:** read-only access to the linked demo profile, trends and reports.")
        st.info("This is a local academic demo. Do not enter real patient identifiers or use these demo credentials for sensitive information.")
    with right:
        with st.container(border=True):
            st.markdown("### Sign in")
            role = st.radio("Login as", ["Doctor", "Patient / Caregiver"], horizontal=False, key="login_role")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", value="doctor" if role == "Doctor" else "patient")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                if _authenticate(role, username, password):
                    st.session_state.authenticated = True
                    st.session_state.user_role = "doctor" if role == "Doctor" else "patient"
                    st.session_state.login_username = username.strip()
                    profiles = load_patient_profiles()
                    linked_id = os.getenv("SMARTAI_PATIENT_PROFILE_ID", "PT-101")
                    chosen = next((p for p in profiles if str(p.get("id")) == linked_id), profiles[0] if profiles else {})
                    st.session_state.patient_id = str(chosen.get("id", "PT-101"))
                    st.session_state.patient_name = chosen.get("name", "Demo Patient")
                    st.session_state.current_patient = dict(chosen.get("data", {}))
                    st.session_state.scenario = chosen.get("scenario", "stable_healthy")
                    st.session_state.active_patient_id = st.session_state.patient_id
                    st.session_state.history_df = generate_patient_history(st.session_state.current_patient, scenario=st.session_state.scenario, n_days=45, n_logs=24)
                    st.session_state.history_patient_id = st.session_state.patient_id
                    st.session_state.nav_selection = "📊 Dashboard" if st.session_state.user_role == "patient" else "🗂️ Patient Worklist"
                    st.rerun()
                st.error("Incorrect username or password for the selected login type.")
            with st.expander("Demo sign-in details"):
                st.markdown("**Doctor:** `doctor` / `doctor123`  \n**Patient/Caregiver:** `patient` / `patient123`")
                st.caption("Demo credentials only. Override with SMARTAI_DOCTOR_USERNAME, SMARTAI_DOCTOR_PASSWORD, SMARTAI_PATIENT_USERNAME, and SMARTAI_PATIENT_PASSWORD environment variables for a local demonstration.")


if not st.session_state.get("authenticated", False):
    _login_screen()
    st.stop()

sample_profiles = load_patient_profiles()
role = st.session_state.get("user_role", "patient")
# Apply redirects requested by a page after the previous run has ended, before widgets are created.
if st.session_state.get("pending_active_patient_id"):
    st.session_state.active_patient_id = st.session_state.pop("pending_active_patient_id")
if st.session_state.get("pending_nav_selection"):
    st.session_state.nav_selection = st.session_state.pop("pending_nav_selection")

# Reconcile selected patient with latest persistent registry on every rerun.
profile_by_id = {str(p.get("id")): p for p in sample_profiles if p.get("id")}
if role == "patient":
    linked_id = os.getenv("SMARTAI_PATIENT_PROFILE_ID", "PT-101")
    if linked_id not in profile_by_id and sample_profiles:
        linked_id = str(sample_profiles[0]["id"])
    active_id = linked_id
else:
    active_id = str(st.session_state.get("patient_id", sample_profiles[0].get("id", "PT-101") if sample_profiles else "PT-101"))
    if active_id not in profile_by_id and sample_profiles:
        active_id = str(sample_profiles[0]["id"])

if active_id in profile_by_id:
    active_profile = profile_by_id[active_id]
    # On first entry, and after a registry update, profile data are authoritative.
    if st.session_state.get("loaded_profile_id") != active_id:
        st.session_state.current_patient = dict(active_profile.get("data", {}))
        st.session_state.patient_id = active_id
        st.session_state.patient_name = active_profile.get("name", "Demo patient")
        st.session_state.scenario = active_profile.get("scenario", "stable_healthy")
        st.session_state.history_df = generate_patient_history(st.session_state.current_patient, scenario=st.session_state.scenario, n_days=45, n_logs=24)
        st.session_state.history_patient_id = active_id
        st.session_state.loaded_profile_id = active_id


def _switch_active_patient():
    previous_id = str(st.session_state.get("patient_id", ""))
    chosen_id = str(st.session_state.get("active_patient_id", ""))
    if previous_id and previous_id != chosen_id:
        for key in list(st.session_state.keys()):
            if str(key).startswith(f"patient_{previous_id}_"):
                st.session_state.pop(key, None)
    chosen = next((p for p in load_patient_profiles() if str(p.get("id")) == chosen_id), None)
    if not chosen:
        return
    st.session_state.patient_id = chosen_id
    st.session_state.patient_name = chosen.get("name", "Demo patient")
    st.session_state.current_patient = dict(chosen.get("data", {}))
    st.session_state.scenario = chosen.get("scenario", "stable_healthy")
    st.session_state.history_df = generate_patient_history(st.session_state.current_patient, scenario=st.session_state.scenario, n_days=45, n_logs=24)
    st.session_state.history_patient_id = chosen_id
    st.session_state.loaded_profile_id = chosen_id
    for key in ("form_patient_data", "form_patient_id"):
        st.session_state.pop(key, None)

if role == "patient":
    # Patient/caregiver accounts are deliberately bound to one profile.
    active_id = str(st.session_state.patient_id)
    nav_options = ["📊 Dashboard", "📋 Patient Records", "📈 Health Trends", "📄 Reports"]
else:
    nav_options = [
        "🗂️ Patient Worklist", "📊 Dashboard", "📋 Patient Records", "📈 Health Trends",
        "🔍 Risk Insights", "🎛️ What-If Scenarios", "👥 Population Overview", "📄 Reports",
        "➕ Add Patient", "⚙️ Settings"
    ]

with st.sidebar:
    st.markdown("### 🩺 Smart AI Health")
    st.caption("Clinical decision-support & monitoring prototype")
    st.markdown(f"**Signed in:** {st.session_state.get('login_username', 'user')}  \n**Role:** {'Doctor' if role == 'doctor' else 'Patient / Caregiver'}")
    if st.button("Sign out", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown("---")
    if role == "doctor" and sample_profiles:
        st.markdown("##### 👤 Active patient")
        profile_names = {str(p["id"]): f"{p.get('name', 'Unnamed')} · {p['id']}" for p in sample_profiles}
        if st.session_state.get("active_patient_id") not in profile_names:
            st.session_state.active_patient_id = str(st.session_state.patient_id)
        st.selectbox(
            "Select patient record", options=list(profile_names.keys()),
            format_func=lambda pid: profile_names.get(str(pid), str(pid)),
            key="active_patient_id", on_change=_switch_active_patient, label_visibility="collapsed"
        )
        st.caption("Patient values and history update when you change the active record.")
    st.markdown("---")
    st.markdown("##### 🧭 Navigation")
    current_nav = st.session_state.get("nav_selection", nav_options[0])
    if current_nav not in nav_options:
        current_nav = nav_options[0]
        st.session_state.nav_selection = current_nav
    nav_selection = st.selectbox("Open section", nav_options, index=nav_options.index(current_nav), key="nav_selection", label_visibility="collapsed")
    st.markdown("---")
    st.caption("Model estimates are research outputs, not diagnoses or treatment instructions.")

render_safety_banner()
risk_predictor, baseline_tracker, xai_engine, recourse_engine = get_system_engines()
training_meta = get_training_metadata()
risk_results = risk_predictor.predict_patient_risk(st.session_state.current_patient)
patient_id = st.session_state.patient_id
patient_name = st.session_state.patient_name
patient_data = st.session_state.current_patient

if nav_selection == "🗂️ Patient Worklist" and role == "doctor":
    render_patient_worklist(sample_profiles, risk_predictor)
elif nav_selection == "📊 Dashboard":
    render_dashboard(patient_id, patient_name, patient_data, risk_results, baseline_tracker, st.session_state.history_df)
elif nav_selection == "📋 Patient Records":
    render_patient_records(patient_id, patient_name, patient_data, read_only=(role != "doctor"))
elif nav_selection == "📈 Health Trends":
    render_health_trends(patient_id, patient_name, patient_data, baseline_tracker, st.session_state.history_df)
elif nav_selection == "🔍 Risk Insights" and role == "doctor":
    render_risk_insights(patient_id, patient_name, patient_data, risk_results, xai_engine, training_meta)
elif nav_selection == "🎛️ What-If Scenarios" and role == "doctor":
    render_what_if_scenarios(patient_id, patient_name, patient_data, recourse_engine, risk_predictor)
elif nav_selection == "👥 Population Overview" and role == "doctor":
    render_population_overview(patient_id, patient_data, risk_results, xai_engine, recourse_engine)
elif nav_selection == "📄 Reports":
    render_reports_view(patient_id, patient_name, patient_data, risk_results, xai_engine, recourse_engine)
elif nav_selection == "➕ Add Patient" and role == "doctor":
    render_add_patient_view(sample_profiles)
elif nav_selection == "⚙️ Settings" and role == "doctor":
    render_settings(baseline_tracker, risk_predictor)
