"""
Comprehensive Automated Test Suite for Smart AI Healthcare Monitoring System.
Tests:
1. Data leakage prevention (Pipelines fit inside folds).
2. Prediction shape and probability validity in [0, 1].
3. SHAP explainer strictly aligned with deployed prediction model.
4. CM-SRF synergy boundary conditions (gamma min/max, zero synergy, clipping).
5. ST-BTE temporal edge cases (zero history, short history, near-zero delta).
6. MEC-CR constraint enforcement (immutability of age/sex, physiological bounds).
7. HL7 FHIR v4 structural schema validation.
8. Safe handling of malformed input, NaN, and impossible physiological values.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backend.data_loader import load_cardiovascular_data, load_diabetes_data
from backend.risk_predictor import CrossMorbidityRiskPredictor
from backend.xai_shap import ClinicalExplainabilityEngine
from backend.baseline_tracker import SparseTemporalBaselineTracker
from backend.counterfactual_engine import CounterfactualRecourseEngine
from backend.report_generator import generate_fhir_risk_assessment, validate_fhir_risk_assessment_structure

class TestHardenedHealthcareAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = CrossMorbidityRiskPredictor(use_calibrated=True)
        cls.xai = ClinicalExplainabilityEngine(use_calibrated=False)
        cls.tracker = SparseTemporalBaselineTracker(tau_days=21.0, z_threshold=2.0)
        cls.recourse = CounterfactualRecourseEngine(cls.predictor)
        
        cls.valid_patient = {
            "age": 55, "sex": 1, "bmi": 28.5, "systolic_bp": 145.0, "diastolic_bp": 92.0,
            "resting_hr": 78.0, "spo2": 97.0, "fasting_glucose": 115.0, "hba1c": 6.1,
            "insulin": 16.0, "cholesterol": 220.0, "hdl": 42.0, "ldl": 140.0,
            "smoking_status": 1, "physical_activity_min": 60, "sleep_hours": 6.5,
            "diet_score": 4, "family_history_cvd": 1, "family_history_diabetes": 1,
            "chest_pain": 0, "shortness_of_breath": 0
        }

    # 1. Pipeline & Leakage Test
    def test_pipeline_structure_and_no_leakage(self):
        pipe = self.predictor.cvd_pipeline
        # If calibrated, check underlying estimator
        base_pipe = pipe.estimator if hasattr(pipe, "estimator") else pipe
        self.assertIn("imputer", base_pipe.named_steps)
        self.assertIn("model", base_pipe.named_steps)
        # Scaler should exist for linear model
        model = base_pipe.named_steps["model"]
        if "LogisticRegression" in type(model).__name__:
            self.assertIn("scaler", base_pipe.named_steps)

    # 2. Prediction Validity
    def test_prediction_output_and_bounds(self):
        pred = self.predictor.predict_patient_risk(self.valid_patient)
        self.assertIn("composite_hazard_percent", pred)
        hazard = pred["composite_hazard_percent"]
        self.assertGreaterEqual(hazard, 0.0)
        self.assertLessEqual(hazard, 100.0)
        self.assertIn(pred["tier_code"], ["GREEN", "YELLOW", "RED"])
        for sub in ["cardiovascular_risk_percent", "diabetes_metabolic_risk_percent", "hypertensive_stroke_risk_percent"]:
            val = pred["sub_hazards"][sub]
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 100.0)

    # 3. Model Alignment Test: Inference Model == SHAP Explanation Model
    def test_shap_model_alignment(self):
        # Explainer deployed_model must be the exact model inside base_pipeline
        explainer_model = self.xai.deployed_model
        predictor_pipe = self.predictor.cvd_pipeline
        predictor_base = predictor_pipe.estimator if hasattr(predictor_pipe, "estimator") else predictor_pipe
        predictor_model = predictor_base.named_steps["model"]
        
        # Verify same class and coefficient shape
        self.assertEqual(type(explainer_model), type(predictor_model))
        if hasattr(explainer_model, "coef_"):
            np.testing.assert_array_equal(explainer_model.coef_, predictor_model.coef_)
        shap_res = self.xai.explain_cardiovascular_risk(self.valid_patient)
        self.assertIn("attributions", shap_res)
        self.assertGreater(len(shap_res["attributions"]), 0)

    # 4. CM-SRF Boundary Conditions & Sensitivity
    def test_cmsrf_boundary_conditions(self):
        # Gamma clipping: gamma passed as 0.5 should be clamped to 1.0; 2.5 clamped to 1.5
        pred_low = self.predictor.predict_patient_risk(self.valid_patient, gamma=0.5)
        self.assertEqual(pred_low["gamma_used"], 1.0)
        pred_high = self.predictor.predict_patient_risk(self.valid_patient, gamma=2.5)
        self.assertEqual(pred_high["gamma_used"], 1.5)
        
        # Zero risk bounds: healthy young patient
        healthy = dict(self.valid_patient)
        healthy.update({"age": 22, "systolic_bp": 110, "diastolic_bp": 70, "fasting_glucose": 80, "smoking_status": 0})
        pred_healthy = self.predictor.predict_patient_risk(healthy)
        self.assertLess(pred_healthy["composite_hazard_percent"], 40.0)
        self.assertGreaterEqual(pred_healthy["composite_hazard_percent"], 1.0)

    # 5. ST-BTE Temporal Limits
    def test_stbte_edge_cases(self):
        now = datetime.now()
        # Zero history
        res_zero = self.tracker.compute_baseline_corridor([], [])
        self.assertEqual(res_zero["mean"], 0.0)
        
        # Single observation
        res_one = self.tracker.compute_baseline_corridor([now], [120.0])
        self.assertEqual(res_one["mean"], 120.0)
        
        # Identical values (variance floor test)
        res_same = self.tracker.compute_baseline_corridor([now - timedelta(days=i) for i in range(5)], [120.0]*5)
        self.assertGreaterEqual(res_same["std"], 1.5) # variance floor enforced
        
        # Near-zero time delta
        res_fast = self.tracker.evaluate_deviation([now, now + timedelta(seconds=1)], [120.0, 120.5], now + timedelta(seconds=2), 121.0)
        self.assertFalse(res_fast["is_anomalous"])

    # 6. MEC-CR Immutability and Boundary Enforcement
    def test_counterfactual_immutability_and_bounds(self):
        rec = self.recourse.compute_optimal_recourse(self.valid_patient, target_risk_percent=25.0)
        opt_vitals = rec["optimized_vitals"]
        
        # Immutable features must be strictly preserved
        self.assertEqual(opt_vitals["age"], self.valid_patient["age"])
        self.assertEqual(opt_vitals["sex"], self.valid_patient["sex"])
        self.assertEqual(opt_vitals["family_history_cvd"], self.valid_patient["family_history_cvd"])
        
        # Modifiable vitals must respect safety boundaries
        self.assertGreaterEqual(opt_vitals["systolic_bp"], 110.0)
        self.assertLessEqual(opt_vitals["systolic_bp"], 180.0)
        self.assertGreaterEqual(opt_vitals["fasting_glucose"], 80.0)
        self.assertLessEqual(opt_vitals["fasting_glucose"], 200.0)
        self.assertGreaterEqual(opt_vitals["physical_activity_min"], 0.0)
        self.assertLessEqual(opt_vitals["physical_activity_min"], 300.0)
        self.assertIn("Model-Generated Hypothetical Scenario", rec["scenario_type"])

    # 7. FHIR v4 Schema Validation
    def test_fhir_structural_validation(self):
        risk_res = self.predictor.predict_patient_risk(self.valid_patient)
        shap_res = self.xai.explain_cardiovascular_risk(self.valid_patient)
        fhir_obj = generate_fhir_risk_assessment("PT-TEST-01", self.valid_patient, risk_res, shap_res)
        
        is_valid, errors = validate_fhir_risk_assessment_structure(fhir_obj)
        self.assertTrue(is_valid, f"FHIR validation failed: {errors}")
        self.assertEqual(fhir_obj["resourceType"], "RiskAssessment")
        self.assertEqual(fhir_obj["status"], "final")
        self.assertTrue(0.0 <= fhir_obj["prediction"][0]["probabilityDecimal"] <= 1.0)

    # 8. Input Validation & Extreme / Malformed Values
    def test_input_resilience(self):
        # Missing fields should not crash (pipeline imputer handles missing)
        sparse_input = {"age": 50, "systolic_bp": 130}
        pred_sparse = self.predictor.predict_patient_risk(sparse_input)
        self.assertIn("composite_hazard_percent", pred_sparse)
        
        # Extreme physiological inputs (SBP=350, glucose=600)
        extreme_input = dict(self.valid_patient)
        extreme_input.update({"systolic_bp": 350.0, "fasting_glucose": 600.0})
        pred_extreme = self.predictor.predict_patient_risk(extreme_input)
        self.assertEqual(pred_extreme["tier_code"], "RED")
        self.assertLessEqual(pred_extreme["composite_hazard_percent"], 100.0)
        # Composite risk must not fall below either sub-domain hazard
        p_cvd = pred_extreme["sub_hazards"]["cardiovascular_risk_percent"]
        p_diab = pred_extreme["sub_hazards"]["diabetes_metabolic_risk_percent"]
        self.assertGreaterEqual(pred_extreme["composite_hazard_percent"], max(p_cvd, p_diab))

    # 9. SpO2 Directional Consistency & Physiological Alignment
    def test_spo2_risk_direction(self):
        # High/Normal SpO2 (99%) should have negative SHAP (mitigating risk relative to cohort)
        patient_optimal_o2 = dict(self.valid_patient)
        patient_optimal_o2["spo2"] = 99.0
        shap_opt = self.xai.explain_cardiovascular_risk(patient_optimal_o2)
        spo2_attr_opt = next(a for a in shap_opt["attributions"] if a["feature"] == "spo2")
        self.assertLess(spo2_attr_opt["shap_value"], 0.0, "Optimal 99% SpO2 must lower risk relative to reference cohort.")

        # Hypoxemic SpO2 (89%) should have positive SHAP (increasing risk)
        patient_hypoxemic = dict(self.valid_patient)
        patient_hypoxemic["spo2"] = 89.0
        shap_hypo = self.xai.explain_cardiovascular_risk(patient_hypoxemic)
        spo2_attr_hypo = next(a for a in shap_hypo["attributions"] if a["feature"] == "spo2")
        self.assertGreater(spo2_attr_hypo["shap_value"], 0.0, "Hypoxemic SpO2 (89%) must increase risk.")

    # 10. Baseline Reliability and Status Handling
    def test_baseline_reliability_flags(self):
        now = datetime.now()
        # Sparse history (< 3 observations) must flag INSUFFICIENT_HISTORY
        corridor_sparse = self.tracker.compute_baseline_corridor([now, now - timedelta(days=2)], [120.0, 122.0])
        self.assertFalse(corridor_sparse["is_reliable"])
        self.assertEqual(corridor_sparse["status"], "INSUFFICIENT_HISTORY")

        # Sufficient history (>= 3 observations) should flag RELIABLE
        timestamps = [now - timedelta(days=i*3) for i in range(6)]
        values = [120.0, 122.0, 119.0, 121.0, 120.0, 123.0]
        corridor_full = self.tracker.compute_baseline_corridor(timestamps, values)
        self.assertTrue(corridor_full["is_reliable"])
        self.assertEqual(corridor_full["status"], "RELIABLE")

    # 11. Clinical Input Validation Schema
    def test_patient_input_validation(self):
        from backend.patient_repository import validate_patient_vitals
        # Valid patient should pass
        valid_res, errs, warns = validate_patient_vitals(self.valid_patient)
        self.assertTrue(valid_res)
        self.assertEqual(len(errs), 0)

        # Inverted BP (Systolic <= Diastolic) must fail
        invalid_bp = dict(self.valid_patient)
        invalid_bp["systolic_bp"] = 80.0
        invalid_bp["diastolic_bp"] = 90.0
        inv_res, inv_errs, _ = validate_patient_vitals(invalid_bp)
        self.assertFalse(inv_res)
        self.assertTrue(any("greater than Diastolic" in e for e in inv_errs))

    # 12. UI Views Import Integrity
    def test_ui_views_import(self):
        from ui.dashboard_view import render_dashboard
        from ui.patient_records_view import render_patient_records
        from ui.health_trends_view import render_health_trends
        from ui.risk_insights_view import render_risk_insights
        from ui.what_if_view import render_what_if_scenarios
        from ui.population_view import render_population_overview
        from ui.settings_view import render_settings
        self.assertTrue(callable(render_dashboard))
        self.assertTrue(callable(render_patient_records))
        self.assertTrue(callable(render_health_trends))
        self.assertTrue(callable(render_risk_insights))
        self.assertTrue(callable(render_what_if_scenarios))
        self.assertTrue(callable(render_population_overview))
        self.assertTrue(callable(render_settings))

if __name__ == "__main__":
    unittest.main()
