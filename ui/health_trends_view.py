"""
Health Trends & Personalized Baseline Tracking View.
Visualizes:
- Longitudinal observations with timestamps
- Exponential-decay personalized baseline corridor (w_i = exp(-delta/tau))
- Standard clinical diagnostic thresholds plotted separately from personalized deviations
- Statistically grounded deviation scores (z-scores) and clinical interpretations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from ui.components import render_patient_summary_card

# Standard clinical reference thresholds
CLINICAL_THRESHOLDS = {
    "systolic_bp": [
        {"name": "Elevated Threshold (AHA/ACC)", "value": 120.0, "color": "#f59e0b", "dash": "dot"},
        {"name": "Stage 2 Hypertension (AHA/ACC)", "value": 140.0, "color": "#ef4444", "dash": "dash"}
    ],
    "diastolic_bp": [
        {"name": "Stage 1 Diastolic HTN", "value": 80.0, "color": "#f59e0b", "dash": "dot"},
        {"name": "Stage 2 Diastolic HTN", "value": 90.0, "color": "#ef4444", "dash": "dash"}
    ],
    "resting_hr": [
        {"name": "Bradycardia Lower Bound", "value": 60.0, "color": "#3b82f6", "dash": "dot"},
        {"name": "Tachycardia Upper Bound", "value": 100.0, "color": "#ef4444", "dash": "dash"}
    ],
    "fasting_glucose": [
        {"name": "Impaired Fasting Glucose (ADA)", "value": 100.0, "color": "#f59e0b", "dash": "dot"},
        {"name": "Diabetes Diagnostic Threshold (ADA)", "value": 126.0, "color": "#ef4444", "dash": "dash"}
    ],
    "spo2": [
        {"name": "Hypoxemia Threshold", "value": 95.0, "color": "#f59e0b", "dash": "dot"},
        {"name": "Severe Hypoxemia Boundary", "value": 90.0, "color": "#ef4444", "dash": "dash"}
    ]
}

STREAM_METADATA = {
    "systolic_bp": ("Systolic Blood Pressure", "mmHg", 120.0),
    "diastolic_bp": ("Diastolic Blood Pressure", "mmHg", 80.0),
    "resting_hr": ("Resting Heart Rate", "bpm", 72.0),
    "fasting_glucose": ("Fasting Blood Glucose", "mg/dL", 95.0),
    "spo2": ("Oxygen Saturation (SpO2)", "%", 98.0)
}

def render_health_trends(patient_id, patient_name, patient_data, baseline_tracker, history_df):
    privacy_mode = st.session_state.get("privacy_mode", False)
    render_patient_summary_card(patient_id, patient_name, patient_data, privacy_mode=privacy_mode)
    
    st.markdown("### 📈 Longitudinal Health Trends & Personalized Baselines")
    st.caption("Tracks self-calibrating physiological baselines and identifies excursions from personal expected corridors.")
    
    # Configuration Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1, 1])
    with col_ctrl1:
        stream_choice = st.selectbox(
            "Select Physiological Stream:",
            options=list(STREAM_METADATA.keys()),
            format_func=lambda k: f"{STREAM_METADATA[k][0]} ({STREAM_METADATA[k][1]})",
            index=0
        )
    with col_ctrl2:
        time_window = st.selectbox("Inspection Window:", ["Last 14 Days", "Last 30 Days", "Last 60 Days", "All Available Data"], index=2)
    with col_ctrl3:
        tau_val = st.slider("Memory Half-Life (τ days):", min_value=7, max_value=60, value=int(baseline_tracker.tau), step=1)
        baseline_tracker.tau = float(tau_val)

    # Filter dataframe by time window
    now = datetime.now()
    if time_window == "Last 14 Days":
        cutoff = now - timedelta(days=14)
    elif time_window == "Last 30 Days":
        cutoff = now - timedelta(days=30)
    elif time_window == "Last 60 Days":
        cutoff = now - timedelta(days=60)
    else:
        cutoff = now - timedelta(days=365)
        
    filtered_df = history_df[history_df["timestamp"] >= cutoff].copy().reset_index(drop=True)
    if filtered_df.empty:
        filtered_df = history_df.copy().reset_index(drop=True)

    stream_name, stream_unit, normal_val = STREAM_METADATA[stream_choice]
    
    # Compute baseline corridors and z-scores over trajectory
    corridors = []
    z_scores = []
    anomalies = []
    reliabilities = []
    clinical_overrides = []
    breached_limits = []
    
    for i in range(len(filtered_df)):
        hist_t = list(filtered_df["timestamp"].iloc[:i])
        hist_v = list(filtered_df[stream_choice].iloc[:i])
        cur_t = filtered_df["timestamp"].iloc[i]
        cur_v = float(filtered_df[stream_choice].iloc[i])
        
        eval_res = baseline_tracker.evaluate_deviation(
            hist_t, hist_v, cur_t, cur_v, metric_name=stream_name, unit=stream_unit
        )
        corridors.append({
            "mean": eval_res["baseline_mean"],
            "lower": eval_res["corridor"][0],
            "upper": eval_res["corridor"][1]
        })
        z_scores.append(eval_res["z_score"])
        anomalies.append(eval_res["is_anomalous"])
        reliabilities.append(eval_res["is_reliable"])
        clinical_overrides.append(eval_res.get("clinical_override_active", False))
        breached_limits.append(eval_res.get("breached_clinical_limit", None))
        
    filtered_df["baseline_mean"] = [c["mean"] for c in corridors]
    filtered_df["lower_bound"] = [c["lower"] for c in corridors]
    filtered_df["upper_bound"] = [c["upper"] for c in corridors]
    filtered_df["z_score"] = z_scores
    filtered_df["is_anomalous"] = anomalies
    filtered_df["is_reliable"] = reliabilities
    filtered_df["clinical_override"] = clinical_overrides
    filtered_df["breached_limit"] = breached_limits

    # Latest reading assessment
    latest_row = filtered_df.iloc[-1]
    cur_val = float(latest_row[stream_choice])
    cur_base = float(latest_row["baseline_mean"])
    cur_z = float(latest_row["z_score"])
    is_anom = bool(latest_row["is_anomalous"])
    is_rel = bool(latest_row["is_reliable"])
    is_override = bool(latest_row["clinical_override"])
    breached_lim = latest_row["breached_limit"]
    
    # Top Metrics Banner
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Current {stream_name}", f"{cur_val:.1f} {stream_unit}", delta=f"{cur_val - cur_base:+.1f} vs baseline")
    m2.metric("Personal Baseline (Mean)", f"{cur_base:.1f} {stream_unit}", help=f"Computed with τ={tau_val} days exponential decay. Anchored from first observation to prevent cold-start distortion.")
    m3.metric("Baseline Standard Deviation", f"±{abs(cur_val - cur_base) / (abs(cur_z) + 1e-4):.1f} {stream_unit}", help="Personal variability band")
    
    if is_anom:
        delta_status = "Excursion Alert"
        delta_col = "inverse"
    elif is_override:
        delta_status = "Clinical Override (Pathological)"
        delta_col = "inverse"
    else:
        delta_status = "Within Corridor"
        delta_col = "normal"
        
    m4.metric("Deviation Score (z-score)", f"{cur_z:+.2f} σ", delta=delta_status, delta_color=delta_col)

    # Plotly Trend Chart
    fig = go.Figure()
    
    # 1. Baseline Corridor Shading
    fig.add_trace(go.Scatter(
        x=filtered_df["timestamp"], y=filtered_df["upper_bound"],
        mode='lines', line=dict(width=0), showlegend=False, name='Upper Corridor'
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df["timestamp"], y=filtered_df["lower_bound"],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(2, 132, 199, 0.12)', name='Personal Baseline Corridor (±2.0σ)'
    ))
    
    # 2. Personalized Baseline Centerline
    fig.add_trace(go.Scatter(
        x=filtered_df["timestamp"], y=filtered_df["baseline_mean"],
        mode='lines', line=dict(color='#0284c7', dash='dash', width=2),
        name=f'Personal Baseline (τ={tau_val}d)'
    ))
    
    # 3. Standard Clinical Reference Guidelines (plotted separately)
    for thresh in CLINICAL_THRESHOLDS.get(stream_choice, []):
        fig.add_trace(go.Scatter(
            x=[filtered_df["timestamp"].min(), filtered_df["timestamp"].max()],
            y=[thresh["value"], thresh["value"]],
            mode='lines',
            line=dict(color=thresh["color"], dash=thresh["dash"], width=1.5),
            name=thresh["name"]
        ))
        
    # 4. Normal Observations
    normal_obs = filtered_df[~filtered_df["is_anomalous"]]
    fig.add_trace(go.Scatter(
        x=normal_obs["timestamp"], y=normal_obs[stream_choice],
        mode='markers+lines',
        marker=dict(color='#0f172a', size=7),
        line=dict(color='#94a3b8', width=1.2),
        name='Recorded Observation'
    ))
    
    # 5. Deviated Observations
    anom_obs = filtered_df[filtered_df["is_anomalous"]]
    if not anom_obs.empty:
        fig.add_trace(go.Scatter(
            x=anom_obs["timestamp"], y=anom_obs[stream_choice],
            mode='markers',
            marker=dict(color='#dc2626', size=12, symbol='diamond'),
            name='Baseline Deviation (|z| ≥ 2.0)'
        ))
        
    fig.update_layout(
        title=dict(
            text=f"Personalized Longitudinal Trajectory: {stream_name}",
            y=0.98, x=0.01, xanchor="left", yanchor="top",
            font=dict(size=14)
        ),
        xaxis_title="Measurement Timestamp (Irregular Logs)",
        yaxis_title=f"{stream_name} ({stream_unit})",
        height=600,
        margin=dict(l=20, r=20, t=40, b=140),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1
        ),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Clinical Interpretation Summary
    st.markdown("#### 💡 Evidence-Grounded Trend Interpretation")
    if is_anom:
        direction = "higher" if cur_z > 0 else "lower"
        st.warning(
            f"**Baseline Excursion Detected:** Recent {stream_name.lower()} readings ({cur_val:.1f} {stream_unit}) "
            f"are statistically {direction} than your estimated personal baseline ({cur_base:.1f} {stream_unit}, z = {cur_z:+.2f}). "
            f"This statistical deviation highlights a shift from personal equilibrium; evaluate in context with standard clinical thresholds."
        )
    elif is_override:
        st.error(
            f"🚨 **Absolute Clinical Threshold Override (Disease Normalization Prevented):** "
            f"Recent {stream_name.lower()} readings ({cur_val:.1f} {stream_unit}) fall within the patient's personal baseline corridor "
            f"({latest_row['lower_bound']:.1f} to {latest_row['upper_bound']:.1f} {stream_unit}, baseline = {cur_base:.1f} {stream_unit}, z = {cur_z:+.2f}), "
            f"**but breach standard clinical diagnostic limits ({breached_lim})**. "
            f"Statistical concordance with a chronically elevated personal baseline must NOT normalize disease or delay clinical intervention."
        )
    else:
        st.success(
            f"**Stable Baseline Concordance:** Recent {stream_name.lower()} observations ({cur_val:.1f} {stream_unit}) "
            f"fall comfortably within your personalized expected corridor ({latest_row['lower_bound']:.1f} to {latest_row['upper_bound']:.1f} {stream_unit}, z = {cur_z:+.2f}) "
            f"and remain within standard clinical boundaries."
        )
        
    # Trajectory Data Table
    with st.expander("📊 View Detailed Trajectory Observation Table"):
        display_cols = ["date_str", stream_choice, "baseline_mean", "z_score", "is_anomalous", "clinical_override", "is_reliable"]
        st.dataframe(
            filtered_df[display_cols].rename(columns={
                "date_str": "Date & Time",
                stream_choice: f"Observed ({stream_unit})",
                "baseline_mean": f"Baseline Mean ({stream_unit})",
                "z_score": "Z-Score",
                "is_anomalous": "Deviation Alert",
                "clinical_override": "Clinical Override",
                "is_reliable": "Baseline Reliable"
            }).tail(12),
            use_container_width=True
        )
