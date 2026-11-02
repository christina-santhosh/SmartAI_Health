"""Small smoke tests for downloadable report generation."""

import json
import unittest
from pathlib import Path

from backend.report_generator import (
    generate_fhir_risk_assessment,
    generate_patient_record_pdf,
    validate_fhir_risk_assessment_structure,
)


class TestReportExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        profile_path = Path(__file__).resolve().parent.parent / "data" / "patient_profiles.json"
        cls.profile = json.loads(profile_path.read_text(encoding="utf-8"))["profiles"][0]
        cls.patient = cls.profile["data"]
        cls.risk = {
            "composite_hazard_percent": 18.4,
            "risk_tier": "Low Risk (Green)",
            "tier_code": "GREEN",
            "clinical_recommendation": "Routine follow-up.",
            "sub_hazards": {
                "cardiovascular_risk_percent": 12.0,
                "diabetes_metabolic_risk_percent": 9.0,
                "hypertensive_stroke_risk_percent": 15.0,
                "synergistic_multiplier_active": False,
            },
        }
        cls.shap = {
            "clinical_summary": "Example explanation summary.",
            "top_risk_drivers": [{"label": "Age", "patient_value": 32, "unit": "years", "shap_value": 0.12}],
        }
        cls.recourse = {"initial_risk_percent": 18.4, "achieved_risk_percent": 16.2, "prescriptions": []}

    def test_pdf_signature(self):
        pdf = generate_patient_record_pdf(
            self.profile["id"], self.profile["name"], self.patient,
            self.risk, self.shap, self.recourse,
        )
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertGreater(len(pdf), 1000)

    def test_fhir_structure(self):
        fhir = generate_fhir_risk_assessment(
            self.profile["id"], self.patient, self.risk, self.shap
        )
        valid, errors = validate_fhir_risk_assessment_structure(fhir)
        self.assertTrue(valid, errors)
        self.assertEqual(fhir["resourceType"], "RiskAssessment")


if __name__ == "__main__":
    unittest.main()
