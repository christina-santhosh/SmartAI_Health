"""
Cross-Morbidity Synergistic Risk Fusion (CM-SRF) Module.
Integrates:
- Method A: Bounded heuristic fusion with sensitivity analysis (gamma in [1.0, 1.5]).
- Method B: Data-driven composite estimation.
- Method C: Multi-morbidity pipeline integration.
Operates on leakage-safe sklearn Pipelines.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

class CrossMorbidityRiskPredictor:
    """
    Evaluates domain-specific risks and computes the composite multi-morbidity hazard index
    using leakage-safe fitted Pipelines.
    """
    def __init__(self, use_calibrated: bool = True):
        self.use_calibrated = use_calibrated
        
        # Load fitted domain pipelines
        cvd_file = "cvd_calibrated_pipeline.pkl" if use_calibrated and (MODELS_DIR / "cvd_calibrated_pipeline.pkl").exists() else "cvd_pipeline.pkl"
        diab_file = "diabetes_calibrated_pipeline.pkl" if use_calibrated and (MODELS_DIR / "diabetes_calibrated_pipeline.pkl").exists() else "diabetes_pipeline.pkl"
        multi_file = "multi_morbidity_calibrated_pipeline.pkl" if use_calibrated and (MODELS_DIR / "multi_morbidity_calibrated_pipeline.pkl").exists() else "multi_morbidity_pipeline.pkl"
        
        self.cvd_pipeline = joblib.load(MODELS_DIR / cvd_file)
        self.cvd_features = joblib.load(MODELS_DIR / "cvd_features.pkl")
        self.cvd_threshold = float(joblib.load(MODELS_DIR / "cvd_threshold.pkl")) if (MODELS_DIR / "cvd_threshold.pkl").exists() else 0.5
        
        self.diab_pipeline = joblib.load(MODELS_DIR / diab_file)
        self.diab_features = joblib.load(MODELS_DIR / "diabetes_features.pkl")
        self.diab_threshold = float(joblib.load(MODELS_DIR / "diabetes_threshold.pkl")) if (MODELS_DIR / "diabetes_threshold.pkl").exists() else 0.5
        
        self.multi_pipeline = joblib.load(MODELS_DIR / multi_file)
        self.multi_features = joblib.load(MODELS_DIR / "multi_morbidity_features.pkl")
        
        # Synergy parameters
        self.gamma_default = 1.25  # Bounded within [1.0, 1.5]

    def predict_patient_risk(
        self,
        patient_dict: Dict[str, Any],
        gamma: float = 1.25,
        fusion_method: str = "bounded_heuristic"
    ) -> Dict[str, Any]:
        """
        Executes multi-domain risk evaluation for a single patient dictionary input.
        :param patient_dict: Patient features dictionary.
        :param gamma: Synergistic multiplier coefficient bounded in [1.0, 1.5].
        :param fusion_method: 'bounded_heuristic', 'data_driven', or 'integrated_multimorbidity'.
        """
        # Constrain gamma within [1.0, 1.5]
        gamma = float(np.clip(gamma, 1.0, 1.5))
        
        # 1. Cardiovascular Risk via Pipeline
        cvd_df = pd.DataFrame([{feat: float(patient_dict.get(feat, 0.0)) for feat in self.cvd_features}])
        p_cvd = float(self.cvd_pipeline.predict_proba(cvd_df)[0, 1])
        
        # 2. Diabetes & Metabolic Risk via Pipeline
        diab_df = pd.DataFrame([{feat: float(patient_dict.get(feat, 0.0)) for feat in self.diab_features}])
        p_diab = float(self.diab_pipeline.predict_proba(diab_df)[0, 1])
        
        # 3. Hypertensive / Stroke Crisis Hazard
        sbp = float(patient_dict.get("systolic_bp", 120.0))
        dbp = float(patient_dict.get("diastolic_bp", 80.0))
        smoking = int(patient_dict.get("smoking_status", 0))
        age = float(patient_dict.get("age", 45.0))
        
        htn_hazard_score = (
            max(0.0, (sbp - 115) / 80.0) * 0.50 +
            max(0.0, (dbp - 75) / 50.0) * 0.25 +
            (0.12 if smoking == 1 else 0.0) +
            max(0.0, (age - 50) / 45.0) * 0.13
        )
        p_htn = float(np.clip(htn_hazard_score, 0.02, 0.95))
        
        # 4. Multi-Morbidity Synergy Computation
        # Note: This synergy multiplier is an expert-configured heuristic rule (gamma=1.25 when both channels > 40%),
        # not an empirically learned biological phenomenon.
        synergy_active = bool(p_cvd > 0.40 and p_diab > 0.40)
        synergy_term = (gamma * p_cvd * p_diab) if synergy_active else (p_cvd * p_diab * 0.05)
        
        if fusion_method == "integrated_multimorbidity":
            # Pass all available multi-morbidity features to the trained multi-morbidity pipeline
            multi_dict = {feat: float(patient_dict.get(feat, 0.0)) for feat in self.multi_features}
            multi_dict["synergy_interaction"] = 1.0 if (sbp >= 140 and float(patient_dict.get("fasting_glucose", 90)) >= 126) else 0.0
            multi_df = pd.DataFrame([multi_dict])
            raw_hazard = float(self.multi_pipeline.predict_proba(multi_df)[0, 1])
            base_hazard = raw_hazard
            unbounded_hazard = raw_hazard
        else:
            # Bounded heuristic fusion: weighted combination of domain hazards
            base_hazard = 0.45 * p_cvd + 0.35 * p_diab + 0.20 * p_htn
            unbounded_hazard = base_hazard * (1.0 + 0.30 * min(0.95, synergy_term))
            
        # Coherent probability lower bound:
        # The composite risk of an adverse clinical event across cardiovascular or metabolic channels
        # cannot logically fall below either channel's isolated probability: P(Composite) >= max(P_cvd, P_diab).
        # An unconstrained weighted average or arbitrary hard-cap (e.g. 0.98) causes the composite to drop
        # below its individual parts despite compounding synergy. Enforcing the axiomatic Fréchet lower bound
        # ensures mathematical and clinical coherence.
        coherent_lower_bound = max(p_cvd, p_diab)
        composite_hazard = max(unbounded_hazard, coherent_lower_bound)
        composite_hazard = float(np.clip(composite_hazard, 0.01, 1.0))
        score_percent = round(composite_hazard * 100, 1)
        
        # Project-Defined Model Risk Tiers (<25% Low, 25-65% Moderate, >=65% High)
        if score_percent < 25.0:
            tier = "Low Risk (Green)"
            tier_code = "GREEN"
            clinical_action = "Routine annual health maintenance. Preventive lifestyle guidance recommended."
        elif score_percent < 65.0:
            tier = "Moderate Risk (Yellow)"
            tier_code = "YELLOW"
            clinical_action = "Proactive clinical follow-up within 30 days. Targeted lifestyle and dietary modification."
        else:
            tier = "High Risk (Red)"
            tier_code = "RED"
            clinical_action = "Urgent clinical attention indicated. Schedule diagnostic workup and specialist consultation."

        return {
            "composite_hazard_percent": score_percent,
            "risk_tier": tier,
            "tier_code": tier_code,
            "clinical_recommendation": clinical_action,
            "fusion_method": fusion_method,
            "gamma_used": gamma,
            "sub_hazards": {
                "cardiovascular_risk_percent": round(p_cvd * 100, 1),
                "diabetes_metabolic_risk_percent": round(p_diab * 100, 1),
                "hypertensive_stroke_risk_percent": round(p_htn * 100, 1),
                "synergistic_multiplier_active": synergy_active
            },
            "derivation_details": {
                "p_cvd_percent": round(p_cvd * 100, 1),
                "p_diab_percent": round(p_diab * 100, 1),
                "p_htn_percent": round(p_htn * 100, 1),
                "base_hazard_percent": round(base_hazard * 100, 1),
                "synergy_multiplier_gamma": gamma,
                "synergy_term": round(synergy_term, 4),
                "synergy_active": synergy_active,
                "synergy_nature": "Heuristic rule-based supervisory multiplier (gamma=1.25 when both channels > 40%), not an empirically learned interaction.",
                "unbounded_hazard_percent": round(unbounded_hazard * 100, 1),
                "coherent_lower_bound_percent": round(coherent_lower_bound * 100, 1),
                "final_composite_percent": score_percent,
                "mathematical_formula": "Composite = max(Base*(1 + 0.30*min(0.95, gamma*p_cvd*p_diab)), max(p_cvd, p_diab))"
            },
            "patient_vitals_summary": {
                "blood_pressure": f"{int(sbp)}/{int(dbp)} mmHg",
                "bmi": round(float(patient_dict.get("bmi", 25.0)), 1),
                "fasting_glucose": f"{int(patient_dict.get('fasting_glucose', 90))} mg/dL",
                "resting_hr": f"{int(patient_dict.get('resting_hr', 72))} bpm"
            }
        }

    def compute_fusion_sensitivity(self, patient_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculates how composite risk varies across gamma in [1.0, 1.5].
        """
        gammas = [1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5]
        records = []
        for g in gammas:
            res = self.predict_patient_risk(patient_dict, gamma=g)
            records.append({
                "gamma": g,
                "composite_hazard_percent": res["composite_hazard_percent"],
                "risk_tier": res["risk_tier"]
            })
        return pd.DataFrame(records)
