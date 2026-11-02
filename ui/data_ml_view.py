"""Dataset inventory, data-quality audit, and machine-learning method overview."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

DATASETS = [
    ("cardiovascular_data.csv", "Synthetic", "3,000", "Cardiovascular risk model training; 17 input features + target."),
    ("diabetes_data.csv", "Synthetic", "3,000", "Diabetes/metabolic risk model training; 11 input features + target."),
    ("multi_morbidity_cohort.csv", "Synthetic", "4,000", "Multi-morbidity acute-event model training; includes engineered synergy interaction."),
    ("real_pima_diabetes_formatted.csv", "Public benchmark", "768", "Pima diabetes benchmark; 8 input features + target."),
    ("real_cleveland_heart_formatted.csv", "Public benchmark", "303", "Cleveland heart disease benchmark; 13 model input features + target plus source columns."),
]
MODELS_INFO = [
    ("Logistic Regression", "Linear probabilistic classifier; selected for CVD, diabetes and multi-morbidity synthetic domains, and Cleveland benchmark."),
    ("Random Forest", "Ensemble of decision trees; non-linear benchmark candidate."),
    ("Histogram Gradient Boosting", "Boosted tree ensemble; benchmark candidate."),
    ("XGBoost", "Gradient-boosted trees; benchmark candidate."),
    ("CatBoost", "Gradient-boosted trees; selected for the public Pima diabetes benchmark."),
]


def _audit(path):
    if not path.exists():
        return {"File": path.name, "Rows": "Missing", "Columns": "—", "Missing cells": "—", "Duplicate rows": "—", "Zero placeholders": "—"}
    df = pd.read_csv(path)
    zero_note = "—"
    if path.name == "real_pima_diabetes_formatted.csv":
        cols = [c for c in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"] if c in df]
        zero_note = int((df[cols] == 0).sum().sum())
    return {"File": path.name, "Rows": len(df), "Columns": len(df.columns), "Missing cells": int(df.isna().sum().sum()), "Duplicate rows": int(df.duplicated().sum()), "Zero placeholders": zero_note}


def render_data_ml_view():
    st.markdown("## Datasets & Machine Learning")
    st.caption("Transparent inventory of the files and methods in this project. Training data are not patient records.")
    st.warning("Research prototype only. Synthetic-data performance does not establish clinical validity, and benchmark performance should not be interpreted as real-world diagnostic accuracy.")

    with st.expander("⚠️ Critical Methodological Limitations & Dataset Truths", expanded=True):
        st.markdown("""
- **All Three Primary Training Cohorts Are Synthetic:**
  - `cardiovascular_data.csv` (3,000 rows), `diabetes_data.csv` (3,000 rows), and `multi_morbidity_cohort.csv` (4,000 rows) are completely synthetically generated fixtures.
  - The only real patient cohorts in the repository are **Cleveland Heart Disease** (303 rows) and **Pima Indians Diabetes** (768 rows). Crucially, these real datasets validate **single isolated diseases only**.
  - There is **no real clinical multi-morbidity dataset** in this repository. Reported AUCs of 0.86 to 0.95 on synthetic datasets reflect mathematical model fitting to synthetic generative rules and **cannot support claims about real-world multi-morbidity or patient outcomes**.

- **Synergy is Hand-Set and Engineered, Not Learned:**
  - The cross-morbidity synergy is governed by a hand-set heuristic multiplier ($\gamma = 1.25$, bounded in $[1.0, 1.5]$) activated whenever both channels exceed 40%.
  - The synthetic multi-morbidity dataset contains an engineered feature `synergy_interaction = (SBP ≥ 140) * (Glucose ≥ 126)`, which was used directly in `data_generator.py` to create the synthetic outcome label.
  - Training a machine learning classifier on synthetic labels built from the developer's own rule confirms rule recovery, but proves nothing about biological synergy in human patients.

- **Composite Hazard Derivation & Axiomatic Probability Bounds:**
  - The composite risk index combines domain channels ($0.45 \cdot P_{cvd} + 0.35 \cdot P_{diab} + 0.20 \cdot P_{htn}$) scaled by the synergy factor.
  - Axiomatically, the composite risk of adverse multi-morbidity event cannot fall below either single channel: $P(A \cup B) \ge \max(P_A, P_B)$.
  - In earlier versions, a linear weighted average combined with an arbitrary 0.98 hard ceiling caused composite risk to register 98.0% even when CVD was 99.8% and Diabetes was 99.5%. Enforcing the coherent lower bound and removing the arbitrary cap ensures the composite score accurately reflects compounding risk.
        """)
    st.markdown("### Dataset inventory")
    st.write("The project contains **5 tabular ML datasets** (3 synthetic training cohorts and 2 public benchmark datasets), plus a JSON file of 5 predefined synthetic patient scenarios. The five CSVs total 11,071 rows, but they are separate task-specific datasets and must not be combined into one cohort.")
    inv_rows = []
    for filename, kind, count, purpose in DATASETS:
        p = DATA / filename
        if p.exists():
            df = pd.read_csv(p)
            inv_rows.append({"Dataset": filename, "Type": kind, "Rows": len(df), "Columns": len(df.columns), "Purpose": purpose})
        else:
            inv_rows.append({"Dataset": filename, "Type": kind, "Rows": count, "Columns": "—", "Purpose": purpose + " File not found."})
    st.dataframe(pd.DataFrame(inv_rows), use_container_width=True, hide_index=True)

    st.markdown("### Live data-quality audit")
    audit = pd.DataFrame([_audit(DATA / row[0]) for row in DATASETS])
    st.dataframe(audit, use_container_width=True, hide_index=True)
    st.caption("Audit values are computed from the CSV files in this project folder. Zero placeholders are counted only for the Pima benchmark's physiological measurement columns.")

    with st.expander("Does the data need cleaning?", expanded=True):
        st.markdown("""
- **Synthetic cardiovascular (3,000 rows):** the supplied CSV currently has no missing cells and no duplicate rows. Still validate ranges, target balance, feature distributions, and whether synthetic generation creates unrealistic correlations.
- **Synthetic diabetes (3,000 rows):** currently no missing cells or duplicate rows. Check physiological bounds and class balance before any retraining.
- **Synthetic multi-morbidity (4,000 rows):** currently no missing cells or duplicate rows. Review the engineered `synergy_interaction`, target definition, and leakage risk; do not feed target-derived fields into predictors.
- **Pima diabetes (768 rows):** the formatted CSV has no literal missing cells, but zero values occur in physiological measurement columns where zero is not plausible. The updated loader treats zero in glucose, blood pressure, skin thickness, insulin, and BMI as missing in the modeling copy; the pipeline's median imputer then handles those values inside each training fold. The source CSV remains unchanged.
- **Cleveland heart disease (303 rows):** the formatted file has 6 missing cells and no exact duplicate rows. The modeling pipeline uses median imputation inside each training fold. Confirm source encoding, categorical-feature treatment, and the original target conversion.

**Recommended cleaning workflow:** verify schema and units; normalize impossible sentinel values to missing; inspect missingness and duplicates; validate ranges and target labels; split train/test before fitting imputers/scalers; fit all preprocessing only on training folds; then evaluate once on the untouched test set. Do not delete outliers automatically: clinically unusual values may be meaningful.
        """)

    st.markdown("### Algorithms benchmarked")
    st.dataframe(pd.DataFrame(MODELS_INFO, columns=["Algorithm", "Role in this project"]), use_container_width=True, hide_index=True)
    st.write("The training code benchmarks five classifiers with stratified 5-fold cross-validation: Logistic Regression, Random Forest, HistGradientBoosting, XGBoost and CatBoost. Candidate pipelines use median imputation; Logistic Regression additionally uses StandardScaler. Threshold selection uses out-of-fold predictions, and the trainer applies sigmoid probability calibration.")

    st.markdown("### Saved model selection and reported metrics")
    metadata_rows = []
    for fname in ["cvd_metadata.json", "diabetes_metadata.json", "multi_morbidity_metadata.json", "public_cleveland_heart_metadata.json", "public_pima_diabetes_metadata.json"]:
        path = MODELS / fname
        if path.exists():
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                metadata_rows.append({
                    "Domain": meta.get("domain", fname.replace("_metadata.json", "")),
                    "Dataset": meta.get("dataset_name", "—"),
                    "Saved selected model": meta.get("selected_model", "—"),
                    "5-fold CV ROC-AUC": meta.get("cv_auc_mean", "—"),
                    "Test ROC-AUC": meta.get("test_roc_auc", "—"),
                    "Test PR-AUC": meta.get("test_pr_auc", "—"),
                    "Test Brier score (calibrated)": meta.get("test_brier_calibrated", "—"),
                })
            except (OSError, json.JSONDecodeError):
                pass
    if metadata_rows:
        st.dataframe(pd.DataFrame(metadata_rows), use_container_width=True, hide_index=True)
    st.caption("Metrics are those recorded in the supplied model metadata. They describe the saved experiments, not a guarantee of performance on newly entered patients.")
    st.markdown("### Additional methods")
    st.markdown("""- **SHAP:** feature-attribution explanations for the cardiovascular model.
- **Probability calibration:** sigmoid/Platt calibration is included in calibrated model pipelines.
- **Personalized temporal baseline:** exponentially weighted baseline corridor and deviation detection over generated/supplied time series.
- **Counterfactual/what-if search:** constrained hypothetical changes; not a treatment recommendation.
- **Evaluation:** ROC-AUC, PR-AUC, sensitivity, specificity, F1, Brier score, calibration curves, subgroup fairness metrics, fusion sensitivity, and temporal alert comparisons.""")
    st.info("The built-in patient profiles and generated history trajectories are synthetic examples. A new patient entered into the UI is a locally stored demo record, not a real EHR connection.")
