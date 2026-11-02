"""
What-If Scenarios & Minimal-Effort Recourse Simulator View.
Allows authorized users to simulate hypothetical lifestyle and physiological adjustments:
- Freezes immutable demographic & genetic attributes (Age, Biological Sex, Family History)
- Evaluates constrained changes within physiological safety bounds
- Computes priority-weighted minimal-effort recourse to reach target risk tier (<= 25%)
- Never mutates the patient's actual medical record.
"""

import streamlit as st
from ui.components import render_patient_summary_card

def render_what_if_scenarios(patient_id, patient_name, patient_data, recourse_engine, risk_predictor):
    privacy_mode = st.session_state.get("privacy_mode", False)
    render_patient_summary_card(patient_id, patient_name, patient_data, privacy_mode=privacy_mode)
    
    st.markdown("### 🎛️ What-If Scenarios & Minimal-Effort Recourse")
    st.caption("Explore hypothetical adjustments across modifiable variables to simulate expected model risk de-escalation.")
    
    # Prominent Disclaimer
    st.info(
        "ℹ️ **Hypothetical Simulation Mode:** Sliders and recourse calculations represent mathematical model projections. "
        "Simulated changes are strictly exploratory, do not mutate actual patient records, and do not constitute clinical treatment plans."
    )
    
    col_sim1, col_sim2 = st.columns([1, 1.1])
    
    with col_sim1:
        st.markdown("#### ⚡ Algorithmic Recourse Search")
        st.caption("Algorithmically searches for the minimal, prioritized adjustment vector to reach the Low Risk tier (≤ 25%).")
        
        # Display frozen immutable variables
        with st.expander("🔒 Immutable Variables (Strictly Locked)", expanded=False):
            st.markdown(f"""
            - **Age:** `{patient_data.get('age', 45)} years` (Unchangeable)
            - **Biological Sex:** `{'Male' if patient_data.get('sex') == 1 else 'Female'}` (Unchangeable)
            - **Family History CVD:** `{'Yes' if patient_data.get('family_history_cvd') == 1 else 'No'}` (Genetic / Locked)
            - **Family History Diabetes:** `{'Yes' if patient_data.get('family_history_diabetes') == 1 else 'No'}` (Genetic / Locked)
            """)
            
        target_tier = st.slider("Target Risk Threshold (%):", min_value=15.0, max_value=35.0, value=25.0, step=1.0)
        
        if st.button("🚀 Calculate Minimal-Effort Path", use_container_width=True):
            with st.spinner("Executing priority-weighted coordinate search..."):
                rec_out = recourse_engine.compute_optimal_recourse(patient_data, target_risk_percent=target_tier)
                st.session_state.recourse_result = rec_out
                
        if "recourse_result" in st.session_state:
            rec = st.session_state.recourse_result
            st.markdown("---")
            target_reached = rec.get("target_reached", False)
            
            if target_reached:
                st.success(f"🎉 **Target Risk Reached (≤ {target_tier}%):** Feasible recourse path identified.")
                st.markdown(f"**Baseline Model Risk:** `{rec['initial_risk_percent']}%` ({rec['initial_tier']})")
                st.markdown(f"**Projected Achieved Risk:** `{rec['achieved_risk_percent']}%` ({rec['achieved_tier']})")
                st.markdown(f"**Normalized Compliance Effort:** `{rec.get('normalized_cost', 0.0)}`")
                if rec.get("prescriptions"):
                    st.markdown("##### 📌 Recommended Hypothetical Adjustments:")
                    for rx in rec["prescriptions"]:
                        st.markdown(f"• **{rx['action']}** (from `{rx['current_value']}` to `{rx['target_value']}` {rx['unit']})")
                else:
                    st.info("Current model estimate was already within or below target bounds.")
            else:
                st.warning(
                    f"⚠️ **Target Risk Infeasible (≤ {target_tier}%):** No feasible path exists within clinically safe bounds "
                    f"for modifiable parameters to reach the target tier."
                )
                st.markdown(f"**Baseline Model Risk:** `{rec['initial_risk_percent']}%` ({rec['initial_tier']})")
                st.markdown(f"**Nearest Achievable Risk:** `{rec['achieved_risk_percent']}%` ({rec['achieved_tier']})")
                st.markdown(f"**Normalized Compliance Effort:** `{rec.get('normalized_cost', 0.0)}`")
                
                st.info(
                    "🔒 **Dominance of Locked / Unmodifiable Variables:** Risk de-escalation is strictly constrained by non-modifiable "
                    f"clinical factors (Age: {patient_data.get('age', 'N/A')} years, biological sex, genetic family history, and established chronic severity). "
                    "Even with maximal safe optimization of modifiable parameters, model risk remains elevated."
                )
                if rec.get("prescriptions"):
                    st.markdown("##### 📌 Hypothetical Adjustments Toward Nearest Achievable Risk:")
                    for rx in rec["prescriptions"]:
                        st.markdown(f"• **{rx['action']}** (from `{rx['current_value']}` to `{rx['target_value']}` {rx['unit']})")
                else:
                    st.info("Modifiable parameters are saturated within safe physiological limits; locked variables dominate risk.")
                    
    with col_sim2:
        st.markdown("#### 🎚️ Interactive Parameter Tuning")
        st.caption("Manually adjust hypothetical vitals and habits to observe instantaneous model response.")
        
        sim_sbp = st.slider(
            "Hypothetical Systolic BP (mmHg)",
            min_value=105.0, max_value=180.0,
            value=float(patient_data.get("systolic_bp", 120.0)),
            step=2.0
        )
        sim_glucose = st.slider(
            "Hypothetical Fasting Glucose (mg/dL)",
            min_value=75.0, max_value=220.0,
            value=float(patient_data.get("fasting_glucose", 95.0)),
            step=5.0
        )
        sim_act = st.slider(
            "Hypothetical Physical Activity (min/week)",
            min_value=0, max_value=360,
            value=int(patient_data.get("physical_activity_min", 60)),
            step=15
        )
        sim_smoke = st.selectbox(
            "Hypothetical Tobacco Cessation",
            options=[0, 1],
            format_func=lambda x: "Non-Smoker (Quit)" if x == 0 else "Active Smoker",
            index=int(patient_data.get("smoking_status", 0))
        )
        sim_sleep = st.slider(
            "Hypothetical Sleep Duration (hours)",
            min_value=4.0, max_value=10.0,
            value=float(patient_data.get("sleep_hours", 7.0)),
            step=0.5
        )
        
        # Evaluate hypothetical scenario
        sim_eval = recourse_engine.simulate_what_if(
            patient_data,
            {
                "systolic_bp": sim_sbp,
                "fasting_glucose": sim_glucose,
                "physical_activity_min": sim_act,
                "smoking_status": sim_smoke,
                "sleep_hours": sim_sleep
            }
        )
        
        delta = sim_eval["risk_delta_percent"]
        delta_color = "#15803d" if delta < 0 else ("#b91c1c" if delta > 0 else "#475569")
        
        st.markdown(f"""
        <div class="patient-card" style="margin-top: 10px;">
            <span class="metric-label">Simulated Model Outcome</span>
            <div style="display: flex; align-items: baseline; gap: 8px; margin-top: 4px;">
                <span style="font-size: 2.2rem; font-weight: 700; color: #0f172a;">{sim_eval['simulated_risk_percent']}%</span>
                <span style="font-size: 1.1rem; font-weight: 600; color: {delta_color};">
                    {delta:+.1f}% vs baseline ({sim_eval['original_risk_percent']}%)
                </span>
            </div>
            <div style="margin-top: 6px; font-size: 0.88rem; color: #475569;">
                Projected Tier: <strong>{sim_eval['simulated_tier']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Locked variables explanation
        locked_factors = sim_eval.get("locked_factors", [])
        if locked_factors or sim_eval.get("locked_variables_dominated"):
            with st.expander("🔒 Why is Risk Reduction Constrained? (Dominance of Locked Variables)", expanded=True):
                st.markdown("""
                **Locked Clinical Variables Constraining De-escalation:**  
                Even with optimal adjustments across modifiable sliders, residual model risk remains anchored by non-modifiable factors:
                """)
                for lf in locked_factors:
                    st.markdown(f"- 🔒 **{lf}**")
                st.caption(
                    "Clinical Insight: In multi-morbidity patients, lifestyle adjustments reduce acute strain but cannot overcome "
                    "advanced chronological age and established systemic chronic illness."
                )
