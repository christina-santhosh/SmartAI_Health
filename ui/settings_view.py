"""
Settings & System Configuration View.
Provides:
- Research parameter controls (Baseline half-life tau, deviation threshold z, synergy gamma)
- Privacy & Anonymization settings
- Audit log & session diagnostics
- Framework & environment version reporting
"""

import streamlit as st
import sys
import sklearn
import pandas

def render_settings(baseline_tracker, risk_predictor):
    st.markdown("### ⚙️ System Settings & Clinical Research Parameters")
    st.caption("Configure algorithmic thresholds, privacy controls, and model pipeline operational parameters.")
    
    # 1. Algorithmic Research Parameters
    st.markdown("#### 🔬 Algorithmic & Baseline Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        new_tau = st.number_input("Longitudinal Baseline Half-Life (τ days)", min_value=7.0, max_value=90.0, value=float(baseline_tracker.tau), step=1.0)
        baseline_tracker.tau = new_tau
    with c2:
        new_z = st.number_input("Baseline Deviation Threshold (|z| score)", min_value=1.5, max_value=3.5, value=float(baseline_tracker.z_threshold), step=0.1)
        baseline_tracker.z_threshold = new_z
    with c3:
        new_gamma = st.number_input("Cross-Morbidity Synergy Multiplier (γ)", min_value=1.0, max_value=1.5, value=float(risk_predictor.gamma_default), step=0.05)
        risk_predictor.gamma_default = new_gamma

    st.markdown("---")

    # 2. Privacy & Clinical Security Controls
    st.markdown("#### 🛡️ Data Privacy & Access Controls")
    p1, p2 = st.columns(2)
    with p1:
        curr_priv = st.session_state.get("privacy_mode", False)
        priv_toggle = st.toggle("🔒 Patient De-Identification (Privacy Mode)", value=curr_priv)
        st.session_state.privacy_mode = priv_toggle
        st.caption("When enabled, real patient names are masked as 'Patient #[ID]' across all visual cards.")
    with p2:
        calib_toggle = st.toggle("📈 Use Platt Sigmoid Calibrated Probabilities", value=risk_predictor.use_calibrated)
        risk_predictor.use_calibrated = calib_toggle
        st.caption("Ensures model probabilities reflect true empirical risk frequencies rather than uncalibrated log-odds.")

    st.markdown("---")

    # 3. Session State & Audit Tools
    st.markdown("#### 🧹 Session State & Diagnostics")
    d1, d2 = st.columns(2)
    with d1:
        if st.button("🗑️ Reset Session & Reload Default Profiles", use_container_width=True):
            st.session_state.clear()
            st.success("Session state cleared. Refreshing application...")
            st.rerun()
    with d2:
        st.caption("Clears active patient form cache and resets all simulation sliders to initial baseline states.")

    st.markdown("---")

    # 4. Environment & Compliance Inventory
    with st.expander("ℹ️ System Environment & Standards Inventory"):
        st.markdown(f"""
        - **Python Runtime:** `{sys.version.split()[0]}`
        - **Scikit-Learn Version:** `{sklearn.__version__}`
        - **Pandas Version:** `{pandas.__version__}`
        - **Interoperability Standard:** `HL7 FHIR v4.0.1 (US Core / RiskAssessment Profile)`
        - **Explainability Standard:** `SHAP Tree/Linear Unified Attribution Engine`
        - **Regulatory Posture:** Research & Clinical Decision-Support Prototype (Non-Diagnostic)
        """)
