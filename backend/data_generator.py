"""
Data generation module for benchmark clinical datasets.
Generates clinically grounded cohorts calibrated to NHANES, Framingham, and Pima distributions
with realistic correlations, non-linear disease risk mappings, and realistic noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_cardiovascular_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic cardiovascular risk dataset.
    Features include demographics, vitals, lab values, lifestyle, and symptoms.
    Target: cvd_risk (0 = low/no risk, 1 = high risk of adverse cardiovascular event).
    """
    rng = np.random.default_rng(random_state)
    
    age = rng.integers(25, 82, size=n_samples)
    sex = rng.binomial(1, 0.52, size=n_samples)  # 1 = Male, 0 = Female
    
    # Body Mass Index (BMI)
    bmi_base = 22 + (age - 25) * 0.1
    bmi = np.clip(rng.normal(bmi_base, 4.8, size=n_samples), 17.0, 48.0)
    
    # Systolic & Diastolic BP (correlated with age, BMI, sex)
    sbp_mean = 110 + (age * 0.38) + (bmi * 0.6) + (sex * 3.5)
    systolic_bp = np.clip(rng.normal(sbp_mean, 14.5), 90, 210)
    diastolic_bp = np.clip(systolic_bp * 0.62 + rng.normal(5, 7, size=n_samples), 55, 125)
    
    # Heart Rate & SpO2
    resting_hr = np.clip(rng.normal(72 + (bmi - 25) * 0.35, 9.8), 45, 125)
    spo2 = np.clip(99.0 - (age * 0.015) - (bmi * 0.02) - rng.exponential(0.6, size=n_samples), 88.0, 100.0)
    
    # Lipids & Blood Chemistry
    cholesterol = np.clip(rng.normal(185 + (age * 0.5) + (bmi * 0.8), 35.0), 110, 340)
    hdl = np.clip(rng.normal(52 - (bmi * 0.3) - (sex * 6.0), 12.0), 22, 95)
    ldl = np.clip(cholesterol - hdl - rng.normal(28, 8, size=n_samples), 50, 240)
    fasting_glucose = np.clip(rng.normal(90 + (bmi * 0.65) + (age * 0.2), 22.0), 65, 260)
    
    # Lifestyle factors
    smoking_status = rng.binomial(1, 0.22, size=n_samples)
    physical_activity = np.clip(rng.normal(150 - (age * 0.8) - (bmi * 1.5), 65), 0, 420)
    sleep_hours = np.clip(rng.normal(7.0 - (age * 0.01), 1.2), 3.5, 10.5)
    family_history_cvd = rng.binomial(1, 0.28, size=n_samples)
    
    # Symptoms
    # Typical/atypical chest discomfort
    cp_prob = 0.05 + 0.15 * (systolic_bp > 140) + 0.15 * (smoking_status == 1) + 0.10 * (age > 55)
    chest_pain = rng.binomial(1, np.clip(cp_prob, 0.02, 0.75))
    sob_prob = 0.04 + 0.12 * (bmi > 30) + 0.10 * (resting_hr > 85)
    shortness_of_breath = rng.binomial(1, np.clip(sob_prob, 0.02, 0.65))

    # Pathophysiological Risk Probability (Log-odds calculation)
    # Clinically grounded: hypoxemia (low SpO2), tachycardia (elevated HR),
    # and chronic sleep restriction directly increase cardiovascular risk.
    log_odds = (
        -4.2
        + 0.045 * (age - 45)
        + 0.45 * sex
        + 0.032 * (systolic_bp - 120)
        + 0.018 * (resting_hr - 72)
        - 0.220 * (spo2 - 97.5)
        + 0.022 * (cholesterol - 190)
        - 0.035 * (hdl - 50)
        + 0.015 * (fasting_glucose - 100)
        + 0.045 * (bmi - 25)
        + 0.750 * smoking_status
        + 0.650 * family_history_cvd
        + 0.850 * chest_pain
        + 0.550 * shortness_of_breath
        - 0.120 * (sleep_hours - 7.0)
        - 0.005 * (physical_activity - 120)
    )
    risk_prob = 1 / (1 + np.exp(-log_odds))
    target_cvd = rng.binomial(1, risk_prob)
    
    df = pd.DataFrame({
        "age": np.round(age, 0).astype(int),
        "sex": sex.astype(int),
        "bmi": np.round(bmi, 1),
        "systolic_bp": np.round(systolic_bp, 1),
        "diastolic_bp": np.round(diastolic_bp, 1),
        "resting_hr": np.round(resting_hr, 1),
        "spo2": np.round(spo2, 1),
        "cholesterol": np.round(cholesterol, 1),
        "hdl": np.round(hdl, 1),
        "ldl": np.round(ldl, 1),
        "fasting_glucose": np.round(fasting_glucose, 1),
        "smoking_status": smoking_status.astype(int),
        "physical_activity_min": np.round(physical_activity, 0).astype(int),
        "sleep_hours": np.round(sleep_hours, 1),
        "family_history_cvd": family_history_cvd.astype(int),
        "chest_pain": chest_pain.astype(int),
        "shortness_of_breath": shortness_of_breath.astype(int),
        "target_cvd": target_cvd.astype(int)
    })
    return df

def generate_diabetes_data(n_samples: int = 3000, random_state: int = 123) -> pd.DataFrame:
    """
    Generates realistic Type 2 Diabetes & Metabolic syndrome dataset.
    Target: target_diabetes (0 = Normal/Pre-diabetes well-managed, 1 = High Diabetes Risk/Uncontrolled).
    """
    rng = np.random.default_rng(random_state)
    
    age = rng.integers(21, 80, size=n_samples)
    sex = rng.binomial(1, 0.49, size=n_samples)
    bmi = np.clip(rng.normal(27.5 + (age * 0.05), 5.5), 18.0, 50.0)
    
    # Blood Glucose & HbA1c
    glucose_mean = 85 + (bmi * 0.9) + (age * 0.25)
    fasting_glucose = np.clip(rng.normal(glucose_mean, 28.0), 65, 310)
    hba1c = np.clip(4.2 + (fasting_glucose * 0.024) + rng.normal(0, 0.4, size=n_samples), 4.4, 13.5)
    
    # Insulin & Blood Pressure
    insulin = np.clip(rng.normal(12 + (bmi - 22) * 0.9 + (fasting_glucose - 90) * 0.15, 8.5), 2.0, 95.0)
    systolic_bp = np.clip(115 + (age * 0.3) + (bmi * 0.5) + rng.normal(0, 12, size=n_samples), 95, 195)
    diastolic_bp = np.clip(systolic_bp * 0.63 + rng.normal(4, 6, size=n_samples), 58, 115)
    
    # Lifestyle & Genetics
    family_history_diabetes = rng.binomial(1, 0.32, size=n_samples)
    physical_activity = np.clip(rng.normal(140 - (bmi * 1.8), 60), 0, 360)
    diet_score = rng.integers(1, 11, size=n_samples) # 1=very poor, 10=excellent Mediterranean
    
    # Calculate Diabetes Risk
    log_odds = (
        -4.6
        + 0.038 * (age - 40)
        + 0.08 * (bmi - 24)
        + 0.042 * (fasting_glucose - 95)
        + 1.1 * (hba1c - 5.6)
        + 0.02 * (insulin - 10)
        + 0.8 * family_history_diabetes
        - 0.006 * physical_activity
        - 0.15 * (diet_score - 5)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    target_diabetes = rng.binomial(1, prob)
    
    df = pd.DataFrame({
        "age": np.round(age, 0).astype(int),
        "sex": sex.astype(int),
        "bmi": np.round(bmi, 1),
        "fasting_glucose": np.round(fasting_glucose, 1),
        "hba1c": np.round(hba1c, 1),
        "insulin": np.round(insulin, 1),
        "systolic_bp": np.round(systolic_bp, 1),
        "diastolic_bp": np.round(diastolic_bp, 1),
        "physical_activity_min": np.round(physical_activity, 0).astype(int),
        "family_history_diabetes": family_history_diabetes.astype(int),
        "diet_score": diet_score.astype(int),
        "target_diabetes": target_diabetes.astype(int)
    })
    return df

def generate_multi_morbidity_cohort(n_samples: int = 4000, random_state: int = 789) -> pd.DataFrame:
    """
    Unified multi-condition cohort capturing synergistic hazards between
    cardiovascular, diabetic, and hypertensive acute decompensation.
    """
    rng = np.random.default_rng(random_state)
    
    patient_ids = [f"PT-{1000 + i}" for i in range(n_samples)]
    age = rng.integers(24, 85, size=n_samples)
    sex = rng.binomial(1, 0.51, size=n_samples)
    bmi = np.clip(rng.normal(26.5 + (age * 0.04), 5.2), 17.5, 49.0)
    
    sbp = np.clip(rng.normal(118 + (age * 0.35) + (bmi * 0.5), 16.0), 92, 215)
    dbp = np.clip(sbp * 0.63 + rng.normal(4, 6.5, size=n_samples), 55, 125)
    resting_hr = np.clip(rng.normal(72 + (bmi - 25) * 0.3, 10.5), 48, 120)
    spo2 = np.clip(99 - (age * 0.025) - (bmi * 0.04) - rng.exponential(0.5, size=n_samples), 88.0, 100.0)
    
    fasting_glucose = np.clip(rng.normal(92 + (bmi * 0.8) + (age * 0.2), 26.0), 68, 300)
    hba1c = np.clip(4.3 + (fasting_glucose * 0.023) + rng.normal(0, 0.35, size=n_samples), 4.5, 13.0)
    cholesterol = np.clip(rng.normal(188 + (age * 0.4) + (bmi * 0.7), 36.0), 115, 335)
    hdl = np.clip(rng.normal(51 - (bmi * 0.28) - (sex * 5.0), 11.5), 24, 90)
    ldl = np.clip(cholesterol - hdl - 25, 52, 230)
    
    smoking = rng.binomial(1, 0.21, size=n_samples)
    activity = np.clip(rng.normal(145 - (age * 0.7) - (bmi * 1.3), 62), 0, 400)
    sleep = np.clip(rng.normal(7.1, 1.15), 4.0, 10.0)
    
    fam_cvd = rng.binomial(1, 0.27, size=n_samples)
    fam_diab = rng.binomial(1, 0.30, size=n_samples)
    
    # Synergistic interaction term: Co-occurrence of Hypertension (SBP > 140) and Hyperglycemia (Glucose > 125)
    htn_flag = (sbp >= 140).astype(int)
    diab_flag = (fasting_glucose >= 126).astype(int)
    synergy_interaction = htn_flag * diab_flag
    
    # Multi-Morbidity Composite Hazard Score (0 to 1 continuous probability)
    raw_hazard = (
        -4.5
        + 0.04 * (age - 45)
        + 0.35 * sex
        + 0.03 * (sbp - 120)
        + 0.012 * (resting_hr - 72)
        - 0.070 * (spo2 - 97.5)
        + 0.045 * (bmi - 25)
        + 0.022 * (fasting_glucose - 100)
        + 0.015 * (ldl - 100)
        + 0.65 * smoking
        + 0.55 * fam_cvd
        + 0.50 * fam_diab
        + 0.85 * synergy_interaction  # Synergistic Non-linear multiplier
        - 0.005 * (activity - 120)
    )
    hazard_prob = 1 / (1 + np.exp(-raw_hazard))
    
    # Categorize clinical risk tier
    risk_tier = np.where(hazard_prob < 0.25, "Low Risk", np.where(hazard_prob < 0.65, "Moderate Risk", "High Risk"))
    composite_acute_event = rng.binomial(1, hazard_prob)
    
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": np.round(age, 0).astype(int),
        "sex": sex.astype(int),
        "bmi": np.round(bmi, 1),
        "systolic_bp": np.round(sbp, 1),
        "diastolic_bp": np.round(dbp, 1),
        "resting_hr": np.round(resting_hr, 1),
        "spo2": np.round(spo2, 1),
        "fasting_glucose": np.round(fasting_glucose, 1),
        "hba1c": np.round(hba1c, 1),
        "cholesterol": np.round(cholesterol, 1),
        "hdl": np.round(hdl, 1),
        "ldl": np.round(ldl, 1),
        "smoking_status": smoking.astype(int),
        "physical_activity_min": np.round(activity, 0).astype(int),
        "sleep_hours": np.round(sleep, 1),
        "family_history_cvd": fam_cvd.astype(int),
        "family_history_diabetes": fam_diab.astype(int),
        "synergy_interaction": synergy_interaction.astype(int),
        "hazard_score": np.round(hazard_prob, 4),
        "risk_tier": risk_tier,
        "target_acute_event": composite_acute_event.astype(int)
    })
    return df

def generate_and_save_all():
    print("Generating benchmark clinical datasets...")
    cv_df = generate_cardiovascular_data()
    cv_path = DATA_DIR / "cardiovascular_data.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"Saved {len(cv_df)} rows to {cv_path}")
    
    diab_df = generate_diabetes_data()
    diab_path = DATA_DIR / "diabetes_data.csv"
    diab_df.to_csv(diab_path, index=False)
    print(f"Saved {len(diab_df)} rows to {diab_path}")
    
    multi_df = generate_multi_morbidity_cohort()
    multi_path = DATA_DIR / "multi_morbidity_cohort.csv"
    multi_df.to_csv(multi_path, index=False)
    print(f"Saved {len(multi_df)} rows to {multi_path}")

if __name__ == "__main__":
    generate_and_save_all()
