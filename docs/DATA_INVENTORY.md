# Data Inventory and Provenance Documentation

This document provides a transparent, scientifically honest account of all datasets utilized within the Smart AI Healthcare Monitoring System project. In accordance with clinical AI research standards, all datasets are explicitly categorized as either **Synthetic Benchmark Clinical Data** or **Public Benchmark Clinical Data**.

---

## 1. Summary Overview

| Dataset File | Category | Sample Count | Features | Target Variable | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cardiovascular_data.csv` | **Synthetic Benchmark** | 3,000 | 17 features | `target_cvd` (binary) | UI demonstration, algorithm benchmarking, SHAP evaluation |
| `diabetes_data.csv` | **Synthetic Benchmark** | 3,000 | 11 features | `target_diabetes` (binary) | UI demonstration, metabolic model training |
| `multi_morbidity_cohort.csv`| **Synthetic Benchmark** | 4,000 | 18 features | `target_acute_event` (binary), `hazard_score` | Cross-morbidity synergy modeling, clinician cohort triage |
| `patient_profiles.json` | **Synthetic Reference** | 5 profiles | Full clinical profiles | Pre-set scenarios | Interactive UI scenario testing |
| `real_pima_diabetes_formatted.csv` | **Public Benchmark** | 768 | 8 features | `target_diabetes` (binary) | External model generalization & pipeline validation |
| `real_cleveland_heart_formatted.csv`| **Public Benchmark** | 303 | 13 features | `target_cvd` (binary) | External clinical model benchmarking |

---

## 2. Synthetic Benchmark Datasets

### 2.1 `cardiovascular_data.csv`
- **Source & Generation Method:** Procedurally generated using `backend/data_generator.py` via calibrated logistic equations simulating realistic demographic and hemodynamic distributions.
- **Licensing:** Open Research / MIT (Project-Generated).
- **Number of Samples:** 3,000 records.
- **Features (17):**
  - `age` (years, [25, 81])
  - `sex` (binary, 0 = Female, 1 = Male)
  - `bmi` (kg/m², [17.0, 48.0])
  - `systolic_bp` (mmHg, [90.0, 210.0])
  - `diastolic_bp` (mmHg, [55.0, 125.0])
  - `resting_hr` (bpm, [45.0, 125.0])
  - `spo2` (%, [88.0, 100.0])
  - `cholesterol` (mg/dL, [110.0, 340.0])
  - `hdl` (mg/dL, [22.0, 95.0])
  - `ldl` (mg/dL, [50.0, 240.0])
  - `fasting_glucose` (mg/dL, [65.0, 260.0])
  - `smoking_status` (binary, 0 = Non-smoker, 1 = Smoker)
  - `physical_activity_min` (minutes/week, [0, 420])
  - `sleep_hours` (hours/night, [3.5, 10.5])
  - `family_history_cvd` (binary)
  - `chest_pain` (binary, exertional angina indicator)
  - `shortness_of_breath` (binary, dyspnea indicator)
- **Target Variable:** `target_cvd` (binary: 0 = Low Risk [55.73%], 1 = High Cardiovascular Risk [44.27%]).
- **Limitations:** Data is synthetically constructed and cannot establish real-world epidemiological causality or clinical safety.
- **Intended Purpose:** Algorithm benchmarking, UI interaction testing, and reproducible unit/integration tests.

### 2.2 `diabetes_data.csv`
- **Source & Generation Method:** Procedurally generated via `backend/data_generator.py`.
- **Licensing:** Open Research / MIT.
- **Number of Samples:** 3,000 records.
- **Features (11):** `age`, `sex`, `bmi`, `fasting_glucose`, `hba1c`, `insulin`, `systolic_bp`, `diastolic_bp`, `physical_activity_min`, `family_history_diabetes`, `diet_score`.
- **Target Variable:** `target_diabetes` (binary: 0 = 59.97%, 1 = 40.03%).
- **Limitations:** Synthetic distributions based on literature-approximated physiological correlation matrices.

### 2.3 `multi_morbidity_cohort.csv`
- **Source & Generation Method:** Generated via `backend/data_generator.py` with an explicit non-linear interaction term modeling the compounding co-occurrence of hypertension ($SBP \ge 140$) and hyperglycemia ($Glucose \ge 126$).
- **Licensing:** Open Research / MIT.
- **Number of Samples:** 4,000 records.
- **Features (18):** Patient ID, demographic vitals, blood chemistry, lifestyle parameters, and `synergy_interaction`.
- **Target Variables:** `hazard_score` (continuous probability $[0.0028, 0.9931]$), `risk_tier` (categorical: Low, Moderate, High), and `target_acute_event` (binary).

---

## 3. Public Benchmark Clinical Datasets

### 3.1 Pima Indians Diabetes Dataset (`real_pima_diabetes_formatted.csv`)
- **Original Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK); hosted by the UCI Machine Learning Repository.
- **Licensing:** Public Domain / CC0.
- **Sample Count:** 768 female patients of Pima Indian heritage ($\ge 21$ years).
- **Features (8):** `pregnancies`, `glucose`, `blood_pressure`, `skin_thickness`, `insulin`, `bmi`, `diabetes_pedigree`, `age`.
- **Target:** `target_diabetes` (binary: 0 = 500 [65.1%], 1 = 268 [34.9%]).
- **Scientific Caveats & Feature Compatibility:** The Pima dataset is restricted to female patients and includes pregnancy counts, whereas the synthetic clinical dataset models both sexes and includes lifestyle/sleep parameters. When used for external validation, models are evaluated either on mutually compatible feature subsets or via domain-specific adaptation, as documented in the benchmark scripts.

### 3.1.1 Data-quality handling for Pima measurements

The formatted Pima CSV contains no literal null cells, but zero values occur in physiological measurement columns where zero is not plausible. The current loader converts zero values in `glucose`, `blood_pressure`, `skin_thickness`, `insulin`, and `bmi` to missing values in the modeling copy. The original CSV is not overwritten. The training pipeline's median imputer then learns imputation values within each training fold, reducing preprocessing leakage. This affects future retraining; existing saved model artifacts and their metadata are not automatically retrained by this code change.

### 3.2 Cleveland Heart Disease Dataset (`real_cleveland_heart_formatted.csv`)
- **Original Source:** Cleveland Clinic Foundation; UCI Machine Learning Repository.
- **Licensing:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Sample Count:** 303 patients (contains missing values in `ca` and `thal`, imputed via median during pipeline execution).
- **Features (13):** `age`, `sex`, `cp` (chest pain type), `trestbps` (resting BP), `chol` (cholesterol), `fbs` (fasting blood sugar > 120), `restecg`, `thalach` (max HR achieved), `exang` (exercise angina), `oldpeak`, `slope`, `ca`, `thal`.
- **Target:** `target_cvd` (binary: presence of heart disease [diagnosis $> 0$] = 139 [45.87%], absence [0] = 164 [54.13%]).
- **Scientific Caveats & Feature Compatibility:** Features differ in format from continuous wearable logs (e.g. categorical ECG, peak ST depression), and are used strictly to benchmark standard classifier pipelines on authentic patient records.

---

## 3.3 Data-cleaning summary

A live audit of the supplied files at project update time found:

- Synthetic cardiovascular: 3,000 rows, 0 missing cells, 0 exact duplicate rows.
- Synthetic diabetes: 3,000 rows, 0 missing cells, 0 exact duplicate rows.
- Synthetic multi-morbidity: 4,000 rows, 0 missing cells, 0 exact duplicate rows.
- Formatted Pima: 768 rows, 0 literal missing cells and 0 exact duplicate rows; physiological zero placeholders require semantic cleaning as described above.
- Formatted Cleveland: 303 rows, 6 missing cells and 0 exact duplicate rows; median imputation is performed in the modeling pipeline.

These counts are file-level observations, not a guarantee of overall data quality. Additional checks should cover units, impossible values, category encoding, target construction, subgroup representation, synthetic-data artifacts, and potential target leakage.

## 4. Methodological Statement

> **Notice on Clinical Validity:**  
> The internal models powering the interactive Streamlit demonstration are trained on synthetic benchmark datasets designed to reflect physiological boundaries. While public clinical datasets are included for external pipeline verification, **no claim is made that this system is clinically validated for medical diagnostic use.**
