"""
Patient Repository and Health Profile Management Module.
Manages patient demographic profiles, vital sign observation logs,
data quality validation, and longitudinal trajectory generation.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Clinical bounds for validation
PHYSIOLOGICAL_RANGES = {
    "age": (18, 110, "years", "Age must be between 18 and 110."),
    "bmi": (12.0, 65.0, "kg/m²", "BMI must be between 12.0 and 65.0 kg/m²."),
    "systolic_bp": (70.0, 260.0, "mmHg", "Systolic BP must be between 70 and 260 mmHg."),
    "diastolic_bp": (40.0, 160.0, "mmHg", "Diastolic BP must be between 40 and 160 mmHg."),
    "resting_hr": (35.0, 200.0, "bpm", "Resting HR must be between 35 and 200 bpm."),
    "spo2": (70.0, 100.0, "%", "SpO2 must be between 70% and 100%."),
    "fasting_glucose": (40.0, 500.0, "mg/dL", "Fasting Glucose must be between 40 and 500 mg/dL."),
    "hba1c": (3.5, 18.0, "%", "HbA1c must be between 3.5% and 18.0%."),
    "cholesterol": (80.0, 500.0, "mg/dL", "Total Cholesterol must be between 80 and 500 mg/dL."),
    "hdl": (10.0, 150.0, "mg/dL", "HDL must be between 10 and 150 mg/dL."),
    "ldl": (30.0, 350.0, "mg/dL", "LDL must be between 30 and 350 mg/dL."),
    "sleep_hours": (2.0, 16.0, "hrs", "Sleep hours must be between 2.0 and 16.0 hrs."),
    "physical_activity_min": (0, 1200, "min/wk", "Activity must be between 0 and 1200 min/wk.")
}

def validate_patient_vitals(patient_dict: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validates patient dictionary against physiological ranges.
    Returns: (is_valid, list_of_errors, list_of_clinical_warnings)
    """
    errors = []
    warnings = []
    
    # Range validations
    for field, (low, high, unit, err_msg) in PHYSIOLOGICAL_RANGES.items():
        if field in patient_dict and patient_dict[field] is not None:
            try:
                val = float(patient_dict[field])
                if val < low or val > high:
                    errors.append(f"{err_msg} Observed: {val} {unit}.")
            except (ValueError, TypeError):
                errors.append(f"Invalid non-numeric value for {field}: {patient_dict[field]}")
                
    # Clinical sanity checks
    sbp = float(patient_dict.get("systolic_bp", 120))
    dbp = float(patient_dict.get("diastolic_bp", 80))
    if sbp <= dbp:
        errors.append(f"Systolic BP ({sbp} mmHg) must be strictly greater than Diastolic BP ({dbp} mmHg).")
        
    chol = float(patient_dict.get("cholesterol", 200))
    hdl = float(patient_dict.get("hdl", 50))
    ldl = float(patient_dict.get("ldl", 100))
    if chol < (hdl + ldl - 30):
        warnings.append(f"Total Cholesterol ({chol}) appears lower than HDL ({hdl}) + LDL ({ldl}). Verify lab values.")
        
    if float(patient_dict.get("spo2", 98)) < 90.0:
        warnings.append("Severe hypoxemia detected (SpO2 < 90%). Immediate clinical attention advised.")
        
    if sbp >= 180.0 or dbp >= 120.0:
        warnings.append("Hypertensive Crisis range detected (BP ≥ 180/120 mmHg). Requires urgent medical evaluation.")
        
    return (len(errors) == 0, errors, warnings)

def _profile_registry_path() -> Path:
    """Writable local registry for edits/new profiles; fixture data remains untouched."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "patient_registry.json"


def load_patient_profiles() -> List[Dict[str, Any]]:
    """Load built-in demo profiles, then overlay persistent user-created/edited profiles."""
    prof_path = DATA_DIR / "patient_profiles.json"
    defaults = []
    if prof_path.exists():
        with open(prof_path, "r", encoding="utf-8") as f:
            defaults = json.load(f).get("profiles", [])

    registry_path = _profile_registry_path()
    saved = []
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                saved = json.load(f).get("profiles", [])
        except (json.JSONDecodeError, OSError):
            saved = []

    merged = {str(p.get("id")): p for p in defaults if p.get("id")}
    for profile in saved:
        if profile.get("id"):
            merged[str(profile["id"])] = profile
    return list(merged.values())


def save_patient_profile(profile: Dict[str, Any]) -> None:
    """Persist a created or edited patient profile in the local JSON registry."""
    if not profile.get("id") or not profile.get("name") or not isinstance(profile.get("data"), dict):
        raise ValueError("A patient profile requires an id, name, and data dictionary.")
    registry_path = _profile_registry_path()
    try:
        existing = json.loads(registry_path.read_text(encoding="utf-8")).get("profiles", []) if registry_path.exists() else []
    except (json.JSONDecodeError, OSError):
        existing = []
    by_id = {str(item.get("id")): item for item in existing if item.get("id")}
    safe_profile = dict(profile)
    safe_profile["data"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in profile["data"].items()}
    safe_profile["updated_at"] = datetime.now().isoformat(timespec="seconds")
    by_id[str(safe_profile["id"])] = safe_profile
    registry_path.write_text(json.dumps({"profiles": list(by_id.values())}, indent=2, ensure_ascii=False), encoding="utf-8")

def generate_patient_history(
    patient_data: Dict[str, Any],
    scenario: str = "stable_healthy",
    n_days: int = 45,
    n_logs: int = 24
) -> pd.DataFrame:
    """
    Generates a realistic historical timeline of sparse vital readings ending at the current patient state.
    """
    rng = np.random.default_rng(hash(patient_data.get("id", "PT-101")) % 10000)
    now = datetime.now()
    base_date = now - timedelta(days=n_days)
    
    # Irregular log timestamps
    offsets = np.sort(rng.uniform(0, n_days, size=n_logs))
    # Make sure the last log is current
    offsets[-1] = n_days
    
    dates = [base_date + timedelta(days=float(o)) for o in offsets]
    
    cur_sbp = float(patient_data.get("systolic_bp", 120.0))
    cur_dbp = float(patient_data.get("diastolic_bp", 80.0))
    cur_hr = float(patient_data.get("resting_hr", 72.0))
    cur_spo2 = float(patient_data.get("spo2", 98.0))
    cur_glucose = float(patient_data.get("fasting_glucose", 95.0))
    
    # Trajectory behavior based on scenario
    records = []
    for idx, d in enumerate(dates):
        progress = idx / max(1, n_logs - 1)
        
        if scenario == "acute_drift":
            # Baseline was lower, recently drifted upward
            sbp = (cur_sbp - 24.0) + (progress ** 2) * 24.0 + rng.normal(0, 3.0)
            dbp = (cur_dbp - 14.0) + (progress ** 2) * 14.0 + rng.normal(0, 2.0)
            hr = (cur_hr - 16.0) + (progress ** 2) * 16.0 + rng.normal(0, 2.5)
            glucose = cur_glucose + rng.normal(0, 3.5)
            spo2 = cur_spo2 + rng.normal(0, 0.4)
        elif scenario == "diabetic_decompensation":
            # Glucose has progressively spiked
            glucose = (cur_glucose - 60.0) + (progress ** 1.8) * 60.0 + rng.normal(0, 6.0)
            sbp = cur_sbp + rng.normal(0, 4.0)
            dbp = cur_dbp + rng.normal(0, 3.0)
            hr = cur_hr + rng.normal(0, 3.0)
            spo2 = cur_spo2 + rng.normal(0, 0.4)
        else: # stable_healthy
            sbp = cur_sbp + rng.normal(0, 3.0)
            dbp = cur_dbp + rng.normal(0, 2.0)
            hr = cur_hr + rng.normal(0, 2.5)
            glucose = cur_glucose + rng.normal(0, 4.0)
            spo2 = cur_spo2 + rng.normal(0, 0.3)
            
        records.append({
            "timestamp": d,
            "date_str": d.strftime("%Y-%m-%d %H:%M"),
            "systolic_bp": round(float(np.clip(sbp, 85, 230)), 1),
            "diastolic_bp": round(float(np.clip(dbp, 50, 135)), 1),
            "resting_hr": round(float(np.clip(hr, 45, 140)), 1),
            "spo2": round(float(np.clip(spo2, 85.0, 100.0)), 1),
            "fasting_glucose": round(float(np.clip(glucose, 60, 320)), 1)
        })
        
    df = pd.DataFrame(records)
    # Ensure final row matches patient's current readings
    df.loc[df.index[-1], "systolic_bp"] = cur_sbp
    df.loc[df.index[-1], "diastolic_bp"] = cur_dbp
    df.loc[df.index[-1], "resting_hr"] = cur_hr
    df.loc[df.index[-1], "spo2"] = cur_spo2
    df.loc[df.index[-1], "fasting_glucose"] = cur_glucose
    return df
