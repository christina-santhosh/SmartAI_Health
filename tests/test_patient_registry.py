import json
import tempfile
import unittest
from pathlib import Path

import backend.patient_repository as repository
from backend.data_loader import load_real_pima_dataset


class PatientRegistryTests(unittest.TestCase):
    def test_saved_profiles_overlay_defaults_and_new_profiles_persist(self):
        original_dir = repository.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                repository.DATA_DIR = Path(temp_dir)
                (Path(temp_dir) / "patient_profiles.json").write_text(
                    json.dumps({"profiles": [{"id": "PT-101", "name": "Demo", "scenario": "stable_healthy", "data": {"age": 32}}]}),
                    encoding="utf-8",
                )
                repository.save_patient_profile({"id": "PT-101", "name": "Demo", "scenario": "stable_healthy", "data": {"age": 33}})
                repository.save_patient_profile({"id": "PT-601", "name": "New Patient", "scenario": "user_added", "data": {"age": 45}})
                profiles = repository.load_patient_profiles()
                self.assertEqual(len(profiles), 2)
                self.assertEqual(next(p for p in profiles if p["id"] == "PT-101")["data"]["age"], 33)
                self.assertEqual(next(p for p in profiles if p["id"] == "PT-601")["name"], "New Patient")
        finally:
            repository.DATA_DIR = original_dir

    def test_pima_zero_placeholders_are_missing_in_modeling_copy(self):
        dataset = load_real_pima_dataset()
        self.assertEqual(len(dataset["X_train"]) + len(dataset["X_test"]), 768)
        columns = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
        missing = dataset["X_train"][columns].isna().sum().sum() + dataset["X_test"][columns].isna().sum().sum()
        self.assertGreater(missing, 0)


if __name__ == "__main__":
    unittest.main()
