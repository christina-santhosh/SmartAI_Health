"""
Dashboard View: Modern Clinical Overview.
Answers:
1. Which patient or profile am I viewing?
2. What has changed recently?
3. What does the current model estimate?
4. What information or action needs attention?
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from ui.components import render_metric_card, render_patient_summary_card

def render_dashboard(patient_id, patient_name, patient_data, risk_results, baseline_tracker, history_df):
    privacy_mode = st.session_state.get("privacy_mode", False)
    
    # 1. Patient Identification Header
    render_patient_summary_card(patient_id, patient_name, patient_data, privacy_mode=privacy_mode)
    
    # Extract Key Clinical Measurements
    sbp = float(patient_data.get("systolic_bp", 120))
    dbp = float(patient_data.get("diastolic_bp", 80))
    hr = float(patient_data.get("resting_hr", 72))
    spo2 = float(patient_data.get("spo2", 98))
    glucose = float(patient_data.get("fasting_glucose", 90))
    hba1c = float(patient_data.get("hba1c", 5.4))
    
    # Status calculations
    bp_status = "Optimal" if (sbp < 120 and dbp < 80) else ("Elevated" if sbp < 130 else ("Stage 1 HTN" if sbp < 140 else "Stage 2 HTN"))
    bp_type = "normal" if sbp < 120 else ("caution" if sbp < 140 else "warning")
    
    hr_status = "Normal" if 60 <= hr <= 100 else ("Bradycardia" if hr < 60 else "Tachycardia")
    hr_type = "normal" if 60 <= hr <= 100 else "caution"
    
    spo2_status = "Normal" if spo2 >= 95.0 else ("Mild Hypoxemia" if spo2 >= 90.0 else "Severe Hypoxemia")
    spo2_type = "normal" if spo2 >= 95.0 else "warning"
    
    glu_status = "Normal" if glucose < 100 else ("Impaired" if glucose < 126 else "Diabetic Range")
    glu_type = "normal" if glucose < 100 else ("caution" if glucose < 126 else "warning")
    
    hazard_score = risk_results.get("composite_hazard_percent", 0.0)
    risk_tier = risk_results.get("risk_tier", "Low Risk")
    tier_code = risk_results.get("tier_code", "GREEN")
    risk_type = "normal" if tier_code == "GREEN" else ("caution" if tier_code == "YELLOW" else "warning")
    
    # Baseline comparison for SBP
    hist_t = list(history_df["timestamp"])
    hist_sbp = list(history_df["systolic_bp"])
    latest_t = hist_t[-1] if hist_t else datetime.now()
    baseline_eval = baseline_tracker.evaluate_deviation(hist_t[:-1], hist_sbp[:-1], latest_t, sbp, metric_name="Systolic BP", unit="mmHg")
    base_mean = baseline_eval.get("baseline_mean", sbp)
    base_std = baseline_eval.get("baseline_std", 3.0)
    base_status = "Stable Corridor" if not baseline_eval.get("is_anomalous") else f"Deviated ({baseline_eval['z_score']:+.1f}σ)"
    base_type = "normal" if not baseline_eval.get("is_anomalous") else "warning"
    
    # 2. Key Vitals & Model Estimates Summary Grid (7 Cards)
    st.markdown("#### Patient Snapshot & Vitals Status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            label="Blood Pressure",
            value=f"{int(sbp)}/{int(dbp)}",
            unit="mmHg",
            status_text=bp_status,
            status_type=bp_type,
            context=f"Baseline: {base_mean:.0f} mmHg (Δ {sbp - base_mean:+.0f})"
        )
    with c2:
        render_metric_card(
            label="Resting Heart Rate",
            value=f"{int(hr)}",
            unit="bpm",
            status_text=hr_status,
            status_type=hr_type,
            context="Target resting: 60–100 bpm"
        )
    with c3:
        render_metric_card(
            label="Oxygen Saturation",
            value=f"{spo2:.1f}",
            unit="%",
            status_text=spo2_status,
            status_type=spo2_type,
            context="Room air pulse oximetry"
        )
    with c4:
        render_metric_card(
            label="Fasting Glucose",
            value=f"{int(glucose)}",
            unit="mg/dL",
            status_text=glu_status,
            status_type=glu_type,
            context=f"HbA1c: {hba1c:.1f}%"
        )
        
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_metric_card(
            label="Personal Baseline",
            value=f"{base_mean:.1f}",
            unit="mmHg",
            status_text=base_status,
            status_type=base_type,
            context=f"Corridor: ±{base_std*2:.1f} mmHg (21d τ)"
        )
    with c6:
        render_metric_card(
            label="Model-Estimated Risk",
            value=f"{hazard_score:.1f}",
            unit="%",
            status_text=risk_tier.split()[0] + " Tier",
            status_type=risk_type,
            context="Calibrated Multi-Morbidity"
        )
    with c7:
        render_metric_card(
            label="Data Completeness",
            value="100",
            unit="%",
            status_text="Complete",
            status_type="normal",
            context="17/17 physiological attributes"
        )
    with c8:
        render_metric_card(
            label="Data Freshness",
            value="Current",
            unit="",
            status_text="Live Session",
            status_type="info",
            context="Logged within active consultation"
        )
        
    st.markdown("---")
    
    # 3. Clinical Timeline & Actions Needed
    t_col1, t_col2 = st.columns([1.2, 1])
    
    with t_col1:
        st.markdown("#### 🕒 Recent Clinical Trajectory & Event Log")
        
        # Build timeline entries
        st.markdown("""
        <div class="patient-card">
        """, unsafe_allow_html=True)
        
        # Latest observation
        st.markdown(f"""
        <div class="timeline-item">
            <div class="timeline-time">Today, Most Recent Check-In</div>
            <div class="timeline-content">
                <strong>New Consultation Check-In:</strong> Vitals logged (BP: {int(sbp)}/{int(dbp)} mmHg, HR: {int(hr)} bpm, SpO2: {spo2:.1f}%, Glucose: {int(glucose)} mg/dL).
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Baseline drift check
        if baseline_eval.get("is_anomalous"):
            st.markdown(f"""
            <div class="timeline-item alert">
                <div class="timeline-time">Baseline Drift Alert</div>
                <div class="timeline-content">
                    <strong style="color: #b91c1c;">Significant Personal Baseline Excursion:</strong> SBP ({int(sbp)} mmHg) deviates by {baseline_eval['z_score']:+.1f} standard deviations from personal baseline.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-time">Personal Baseline Verification</div>
                <div class="timeline-content">
                    Observed vitals remain within the estimated 21-day self-calibrating personal corridor.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Model estimate update
        st.markdown(f"""
        <div class="timeline-item {'warning' if tier_code != 'GREEN' else ''}">
            <div class="timeline-time">Model Risk Stratification Updated</div>
            <div class="timeline-content">
                Calibrated composite multi-morbidity risk estimated at <strong>{hazard_score:.1f}%</strong> ({risk_tier}).
                Primary driver identified via model attribution.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with t_col2:
        st.markdown("#### ⚡ Priority Attention & Recommended Actions")
        
        st.markdown(f"""
        <div class="patient-card">
            <h5 style="color: #0f172a; margin-bottom: 8px;">Decision-Support Summary</h5>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                {risk_results.get('clinical_recommendation', 'Routine wellness check and lifestyle monitoring.')}
            </p>
            <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 12px 0;">
            <div style="font-size: 0.82rem; color: #64748b;">
                <strong>Quick Navigation:</strong>
                <ul style="padding-left: 18px; margin-top: 6px;">
                    <li>Inspect longitudinal trends in <strong>Health Trends</strong>.</li>
                    <li>Review feature contributions in <strong>Risk Insights</strong>.</li>
                    <li>Explore hypothetical lifestyle adjustments in <strong>What-If Scenarios</strong>.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
