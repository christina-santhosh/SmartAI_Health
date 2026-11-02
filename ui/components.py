"""
Healthcare UI Components: Polished, Accessible Clinical SaaS Design System.
Provides cards, badges, charts, and header components.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, Optional

def apply_healthcare_theme():
    """Injects modern, accessible healthcare SaaS CSS styling."""
    st.markdown("""
    <style>
        /* Base typography & colors */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #0f172a;
        }
        
        /* Container & Page Header */
        .clinic-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 0.8rem;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 1.2rem;
        }
        .clinic-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .clinic-subtitle {
            font-size: 0.92rem;
            color: #64748b;
            margin-top: 0.2rem;
        }
        
        /* Cards & Containers */
        .patient-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
            margin-bottom: 1rem;
        }
        
        .metric-tile {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
        }
        .metric-label {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            margin-bottom: 4px;
        }
        .metric-val {
            font-size: 1.7rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.1;
        }
        .metric-unit {
            font-size: 0.85rem;
            font-weight: 500;
            color: #64748b;
            margin-left: 3px;
        }
        .metric-sub {
            font-size: 0.80rem;
            color: #475569;
            margin-top: 6px;
        }
        
        /* Semantic Status Badges */
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .badge-green {
            background-color: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
        }
        .badge-amber {
            background-color: #fffbeb;
            color: #b45309;
            border: 1px solid #fde68a;
        }
        .badge-red {
            background-color: #fef2f2;
            color: #b91c1c;
            border: 1px solid #fecaca;
        }
        .badge-blue {
            background-color: #f0f9ff;
            color: #0369a1;
            border: 1px solid #bae6fd;
        }
        .badge-gray {
            background-color: #f8fafc;
            color: #475569;
            border: 1px solid #e2e8f0;
        }
        
        /* Research Prototype Safety Banner */
        .prototype-banner {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-left: 4px solid #d97706;
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 0.82rem;
            color: #92400e;
            margin-bottom: 1rem;
            line-height: 1.4;
        }
        
        /* Timeline styling */
        .timeline-item {
            position: relative;
            padding-left: 20px;
            margin-bottom: 12px;
            border-left: 2px solid #e2e8f0;
        }
        .timeline-item::before {
            content: "";
            position: absolute;
            left: -5px;
            top: 4px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #3b82f6;
        }
        .timeline-item.alert::before {
            background-color: #ef4444;
        }
        .timeline-item.warning::before {
            background-color: #f59e0b;
        }
        .timeline-time {
            font-size: 0.75rem;
            color: #94a3b8;
            font-weight: 500;
        }
        .timeline-content {
            font-size: 0.86rem;
            color: #1e293b;
            margin-top: 2px;
        }
        /* Application shell: calm, clinic-oriented teal palette */
        .stApp { background: #f4f8fa; }
        [data-testid="stHeader"] { background: rgba(244, 248, 250, 0.96); }
        [data-testid="stSidebar"] {
            background: #edf4f6;
            border-right: 1px solid #d8e5e9;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #38515a; }
        .block-container { padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1600px; }
        h1, h2, h3, h4 { color: #12343b; letter-spacing: -0.015em; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d7e3e8;
            border-radius: 12px;
            padding: 12px 14px;
        }
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 9px;
            min-height: 2.45rem;
            font-weight: 600;
            border-color: #b8cbd2;
        }
        div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
            background: #0f7480;
            border-color: #0f7480;
            color: #ffffff;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            border-color: #0f7480;
            color: #0f5963;
        }
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-radius: 8px;
        }
        [data-testid="stDataFrame"] { border: 1px solid #d7e3e8; border-radius: 10px; overflow: hidden; }
        [data-testid="stExpander"] { background: #ffffff; border: 1px solid #d7e3e8; border-radius: 10px; }
        .stAlert { border-radius: 10px; }

    </style>
    """, unsafe_allow_html=True)

def render_safety_banner():
    """Renders the unobtrusive research prototype notice."""
    st.markdown("""
    <div class="prototype-banner">
        ⚖️ <strong>RESEARCH & CLINICAL DECISION-SUPPORT PROTOTYPE:</strong> 
        Risk predictions and baseline alerts are statistical estimations for research and educational exploration. 
        They do not establish a medical diagnosis or replace licensed clinical judgment.
    </div>
    """, unsafe_allow_html=True)

def render_patient_summary_card(patient_id: str, patient_name: str, patient_dict: Dict[str, Any], privacy_mode: bool = False):
    """Renders the top patient identification banner."""
    display_name = f"Patient #{patient_id}" if privacy_mode else f"{patient_name} ({patient_id})"
    age = patient_dict.get("age", "--")
    sex_label = "Male" if patient_dict.get("sex") == 1 else "Female"
    bmi = patient_dict.get("bmi", "--")
    
    st.markdown(f"""
    <div class="patient-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.3rem; font-weight: 700; color: #0f172a;">{display_name}</span>
                    <span class="badge badge-blue">Active Patient</span>
                </div>
                <div style="margin-top: 4px; font-size: 0.88rem; color: #475569;">
                    <strong>Demographics:</strong> {age} yrs | {sex_label} | BMI: {bmi} kg/m² &nbsp;|&nbsp; 
                    <strong>Record Status:</strong> <span style="color: #059669; font-weight: 600;">Complete (17 Parameters)</span>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="timeline-time">Latest Assessment</span><br>
                <span style="font-size: 0.85rem; font-weight: 600; color: #1e293b;">Live Clinical Session</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(
    label: str,
    value: str,
    unit: str = "",
    status_text: str = "",
    status_type: str = "normal",
    context: str = ""
):
    """Renders a structured vital/lab summary card."""
    badge_class = {
        "normal": "badge-green",
        "caution": "badge-amber",
        "warning": "badge-red",
        "info": "badge-blue"
    }.get(status_type, "badge-gray")
    
    st.markdown(f"""
    <div class="metric-tile">
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-label">{label}</span>
                <span class="badge {badge_class}">{status_text}</span>
            </div>
            <div style="margin-top: 6px;">
                <span class="metric-val">{value}</span>
                <span class="metric-unit">{unit}</span>
            </div>
        </div>
        <div class="metric-sub">{context}</div>
    </div>
    """, unsafe_allow_html=True)
