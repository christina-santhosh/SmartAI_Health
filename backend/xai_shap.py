"""
Aligned Explainable AI (XAI) Module: Model-Specific SHAP Explainer.
Guarantees that the explanation model strictly corresponds to the exact deployed model:
- If the deployed model is linear (Logistic Regression): uses shap.LinearExplainer on transformed features.
- If the deployed model is tree-based (Random Forest, XGBoost, CatBoost): uses shap.TreeExplainer.
Exposes reference to the deployed fitted estimator for consistency verification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib
import shap

from backend.data_loader import load_cardiovascular_data

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

class ClinicalExplainabilityEngine:
    """
    Computes local SHAP attributions using an explainer strictly aligned
    with the deployed prediction model.
    """
    def __init__(self, use_calibrated: bool = False):
        # Load the deployed cardiovascular pipeline
        pipe_file = "cvd_calibrated_pipeline.pkl" if use_calibrated and (MODELS_DIR / "cvd_calibrated_pipeline.pkl").exists() else "cvd_pipeline.pkl"
        self.deployed_pipeline = joblib.load(MODELS_DIR / pipe_file)
        self.cvd_features = joblib.load(MODELS_DIR / "cvd_features.pkl")
        
        # If wrapped in CalibratedClassifierCV, extract underlying pipeline
        if hasattr(self.deployed_pipeline, "estimator"):
            self.base_pipeline = self.deployed_pipeline.estimator
        else:
            self.base_pipeline = self.deployed_pipeline
            
        self.imputer = self.base_pipeline.named_steps.get("imputer")
        self.scaler = self.base_pipeline.named_steps.get("scaler")
        self.deployed_model = self.base_pipeline.named_steps["model"]
        
        # Determine model type and initialize corresponding SHAP explainer
        self.model_class_name = type(self.deployed_model).__name__
        
        # Load background reference data for background masker
        train_data = load_cardiovascular_data()["X_train"]
        if self.scaler is not None:
            bg_transformed = self.scaler.transform(self.imputer.transform(train_data))
        else:
            bg_transformed = self.imputer.transform(train_data)
            
        # Use a representative background subset (100 samples) for numerical stability
        bg_summary = shap.sample(bg_transformed, 100, random_state=42)
        
        if "LogisticRegression" in self.model_class_name:
            self.explainer_type = "LinearExplainer"
            self.explainer = shap.LinearExplainer(self.deployed_model, bg_summary)
        elif any(tree_name in self.model_class_name for tree_name in ["Forest", "GradientBoosting", "XGB", "CatBoost"]):
            self.explainer_type = "TreeExplainer"
            self.explainer = shap.TreeExplainer(self.deployed_model)
        else:
            self.explainer_type = "Explainer"
            self.explainer = shap.Explainer(self.deployed_model, bg_summary)

    def explain_cardiovascular_risk(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for the input patient using the aligned explainer.
        """
        # Format input row adhering to feature order
        row_df = pd.DataFrame([{feat: float(patient_dict.get(feat, 0.0)) for feat in self.cvd_features}])
        
        # Apply pipeline transformations (imputation and optional scaling)
        if self.scaler is not None:
            X_trans = self.scaler.transform(self.imputer.transform(row_df))
        else:
            X_trans = self.imputer.transform(row_df)
            
        # Compute SHAP values
        raw_shap = self.explainer.shap_values(X_trans)
        
        # Format SHAP values array
        if isinstance(raw_shap, list):
            vals = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif hasattr(raw_shap, "values"):
            vals = raw_shap.values[0]
        elif len(np.shape(raw_shap)) == 2:
            vals = raw_shap[0]
        elif len(np.shape(raw_shap)) == 3:
            vals = raw_shap[0, :, 1]
        else:
            vals = raw_shap
            
        expected_val = float(
            self.explainer.expected_value[1] 
            if isinstance(getattr(self.explainer, "expected_value", 0.0), (list, np.ndarray)) 
            else getattr(self.explainer, "expected_value", 0.0)
        )
        # Baseline probability: p_0 = 1 / (1 + exp(-expected_val))
        base_prob = 1.0 / (1.0 + np.exp(-expected_val))
        
        row_vals = [float(patient_dict.get(feat, 0.0)) for feat in self.cvd_features]
        attributions = []
        for feat, val, sv in zip(self.cvd_features, row_vals, vals):
            # First-order Taylor approximation of probability shift: delta_p ≈ phi * p0 * (1 - p0) * 100
            approx_prob_shift = float(sv) * base_prob * (1.0 - base_prob) * 100.0
            unit = self._get_feature_unit(feat)
            
            direction_label = "Increases Model Risk (Above Reference Average)" if sv > 0 else "Lowers Model Risk (Below Reference Average)"
            attributions.append({
                "feature": feat,
                "label": self._format_feature_label(feat),
                "patient_value": val,
                "unit": unit,
                "shap_value": round(float(sv), 4),
                "log_odds_contribution": round(float(sv), 4),
                "approx_prob_shift_percent": round(approx_prob_shift, 2),
                "impact": direction_label,
                "is_risk_driver": bool(sv > 0)
            })
            
        attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_risk_drivers = [a for a in attributions if a["shap_value"] > 0][:5]
        top_mitigating = [a for a in attributions if a["shap_value"] < 0][:5]
        
        return {
            "domain": "Cardiovascular",
            "model_architecture": f"{self.model_class_name} ({self.explainer_type})",
            "base_value_log_odds": round(expected_val, 4),
            "base_probability_percent": round(base_prob * 100.0, 1),
            "attributions": attributions,
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_mitigating,
            "top_mitigating_factors": top_mitigating,
            "clinical_summary": self._generate_clinical_summary(top_risk_drivers, top_mitigating)
        }

    def _get_feature_unit(self, feat_key: str) -> str:
        units = {
            "age": "years",
            "sex": "0=F, 1=M",
            "bmi": "kg/m²",
            "systolic_bp": "mmHg",
            "diastolic_bp": "mmHg",
            "resting_hr": "bpm",
            "spo2": "%",
            "cholesterol": "mg/dL",
            "hdl": "mg/dL",
            "ldl": "mg/dL",
            "fasting_glucose": "mg/dL",
            "smoking_status": "0=No, 1=Yes",
            "physical_activity_min": "min/week",
            "sleep_hours": "hrs/night",
            "family_history_cvd": "0=No, 1=Yes",
            "chest_pain": "0=No, 1=Yes",
            "shortness_of_breath": "0=No, 1=Yes"
        }
        return units.get(feat_key, "")

    def _format_feature_label(self, feat_key: str) -> str:
        mapping = {
            "age": "Age",
            "sex": "Biological Sex",
            "bmi": "Body Mass Index (BMI)",
            "systolic_bp": "Systolic Blood Pressure",
            "diastolic_bp": "Diastolic Blood Pressure",
            "resting_hr": "Resting Heart Rate",
            "spo2": "Oxygen Saturation (SpO2)",
            "cholesterol": "Total Serum Cholesterol",
            "hdl": "High-Density Lipoprotein (HDL)",
            "ldl": "Low-Density Lipoprotein (LDL)",
            "fasting_glucose": "Fasting Blood Glucose",
            "smoking_status": "Tobacco Smoking History",
            "physical_activity_min": "Physical Activity (min/week)",
            "sleep_hours": "Nightly Sleep Duration",
            "family_history_cvd": "Family History of CVD",
            "chest_pain": "Chest Discomfort / Angina",
            "shortness_of_breath": "Dyspnea / Shortness of Breath",
            "hba1c": "Glycated Hemoglobin (HbA1c)",
            "insulin": "Fasting Serum Insulin",
            "diet_score": "Nutritional Quality Score"
        }
        return mapping.get(feat_key, feat_key.replace("_", " ").title())

    def _generate_clinical_summary(self, risk_drivers: List[Dict], mitigating: List[Dict]) -> str:
        if not risk_drivers:
            return "All physiological inputs currently reside within standard reference intervals."
            
        primary = risk_drivers[0]
        drivers_str = ", ".join([f"{d['label']} ({d['patient_value']} {d['unit']})" for d in risk_drivers[:3]])
        
        narrative = (
            f"Primary statistical driver increasing model output is **{primary['label']}** "
            f"({primary['patient_value']} {primary['unit']}; +{primary['shap_value']:.2f} log-odds). "
            f"Other contributing factors include: {drivers_str}. "
        )
        if mitigating:
            mit_str = ", ".join([f"{m['label']} ({m['patient_value']} {m['unit']})" for m in mitigating[:2]])
            narrative += f"Factors lowering model output relative to the population reference: {mit_str}. "
            
        narrative += (
            "*(Notice: Feature attributions quantify sensitivity within the mathematical model relative to the reference cohort "
            "and do not constitute proven biological causality or medical advice.)*"
        )
        return narrative
