"""Patient worklist landing page styled for a clinical SaaS workflow."""

import streamlit as st


def _open_profile(profile):
    """Streamlit callback: switch active profile before the next script run."""
    from backend.patient_repository import generate_patient_history
    previous_id = str(st.session_state.get("patient_id", ""))
    new_id = str(profile.get("id", "PT-DEMO"))
    if previous_id and previous_id != new_id:
        for key in list(st.session_state.keys()):
            if str(key).startswith(f"patient_{previous_id}_"):
                st.session_state.pop(key, None)
    st.session_state.current_patient = dict(profile.get("data", {}))
    st.session_state.patient_id = new_id
    st.session_state.patient_name = profile.get("name", "Demo patient")
    st.session_state.scenario = profile.get("scenario", "stable_healthy")
    st.session_state.history_df = generate_patient_history(
        st.session_state.current_patient,
        scenario=st.session_state.scenario,
        n_days=45,
        n_logs=24,
    )
    st.session_state.history_patient_id = st.session_state.patient_id
    st.session_state.loaded_profile_id = st.session_state.patient_id
    st.session_state.active_patient_id = st.session_state.patient_id
    for key in ("form_patient_data", "form_patient_id"):
        st.session_state.pop(key, None)
    st.session_state.nav_selection = "📊 Dashboard"


def render_patient_worklist(profiles, risk_predictor):
    st.markdown("## Patient worklist")
    st.caption("Synthetic demonstration profiles sorted by the current model estimate. Open a record to review measurements, trends and explanations.")

    rows = []
    for profile in profiles:
        data = profile.get("data", {})
        result = risk_predictor.predict_patient_risk(data)
        rows.append({
            "id": profile.get("id", "N/A"),
            "name": profile.get("name", "Unnamed profile"),
            "age": data.get("age", "—"),
            "sex": "M" if data.get("sex") == 1 else "F",
            "risk_percent": float(result.get("composite_hazard_percent", 0.0)),
            "risk_tier": result.get("risk_tier", "Not available"),
            "tier_code": result.get("tier_code", "YELLOW"),
            "sbp": data.get("systolic_bp", "—"),
            "dbp": data.get("diastolic_bp", "—"),
            "scenario": profile.get("scenario", "demo"),
            "profile": profile,
        })

    high_count = sum(r["tier_code"] == "RED" for r in rows)
    adverse_count = sum(r["scenario"] in {"acute_drift", "diabetic_decompensation"} for r in rows)
    completeness = 100.0 if rows else 0.0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patients monitored", len(rows), help="Built-in synthetic demonstration profiles")
    k2.metric("High model risk", high_count, help="Project-defined tier; not a diagnosis")
    k3.metric("Profiles with simulated shifts", adverse_count, help="Based on the scenario fixture labels, not observed real-world events")
    k4.metric("Average data completeness", f"{completeness:.0f}%", help="Completeness of the predefined demonstration profiles")

    st.markdown("---")
    f1, f2 = st.columns([1.4, 1])
    with f1:
        search = st.text_input("Search patients", placeholder="Name or patient ID", label_visibility="visible")
    with f2:
        risk_filter = st.selectbox("Model-risk filter", ["All", "High risk", "Moderate risk", "Low risk", "Simulated shifts"])

    filtered = rows
    if search.strip():
        q = search.strip().casefold()
        filtered = [r for r in filtered if q in r["name"].casefold() or q in r["id"].casefold()]
    if risk_filter == "High risk":
        filtered = [r for r in filtered if r["tier_code"] == "RED"]
    elif risk_filter == "Moderate risk":
        filtered = [r for r in filtered if r["tier_code"] == "YELLOW"]
    elif risk_filter == "Low risk":
        filtered = [r for r in filtered if r["tier_code"] == "GREEN"]
    elif risk_filter == "Simulated shifts":
        filtered = [r for r in filtered if r["scenario"] in {"acute_drift", "diabetic_decompensation"}]

    filtered.sort(key=lambda r: r["risk_percent"], reverse=True)
    st.caption(f"Showing {len(filtered)} of {len(rows)} synthetic demo profiles")

    if not filtered:
        st.info("No profiles match those filters.")
        return

    header = st.columns([2.3, 1.1, 1.4, 1.6, 1.4])
    header[0].markdown("**PATIENT**")
    header[1].markdown("**MODEL RISK**")
    header[2].markdown("**BLOOD PRESSURE**")
    header[3].markdown("**SCENARIO**")
    header[4].markdown("**ACTION**")
    st.divider()

    for i, row in enumerate(filtered):
        cols = st.columns([2.3, 1.1, 1.4, 1.6, 1.4])
        with cols[0]:
            initials = "".join(part[0] for part in row["name"].split()[:2]).upper()
            st.markdown(f"**{row['name']}**")
            st.caption(f"{row['id']} · {row['age']} y · {row['sex']}")
        with cols[1]:
            st.markdown(f"**{row['risk_percent']:.1f}%**")
            st.caption(row["risk_tier"].replace(" (Green)", "").replace(" (Yellow)", "").replace(" (Red)", ""))
        with cols[2]:
            st.markdown(f"{row['sbp']}/{row['dbp']} mmHg")
            st.caption("Systolic / diastolic")
        with cols[3]:
            scenario_label = {
                "stable_healthy": "Stable demo",
                "diabetic_decompensation": "Metabolic shift",
                "acute_drift": "Acute-drift scenario",
            }.get(row["scenario"], "Demo scenario")
            st.caption(scenario_label)
        with cols[4]:
            st.button(
                "Open record",
                key=f"open_profile_{row['id']}",
                use_container_width=True,
                type="primary",
                on_click=_open_profile,
                args=(row["profile"],),
            )
        st.divider()

    st.caption("All profiles on this page are predefined synthetic examples. Model estimates are for research/UI demonstration only.")
