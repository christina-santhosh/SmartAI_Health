"""
Fairness, Robustness, and Subgroup Evaluation Module.
Evaluates model performance across demographic subgroups:
- Biological Sex (Male vs Female)
- Age Cohorts (<45 years, 45-65 years, >65 years)
Calculates subgroup ROC-AUC, Sensitivity, Specificity, Brier score, and Demographic Parity Delta.
Outputs results to results/fairness_evaluation.csv and docs/FAIRNESS_EVALUATION.md.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, brier_score_loss, confusion_matrix

from backend.data_loader import load_cardiovascular_data
from backend.risk_predictor import CrossMorbidityRiskPredictor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

MIN_SAMPLE_THRESHOLD = 30

def evaluate_subgroup_slice(
    name: str,
    X_sub: pd.DataFrame,
    y_sub: np.ndarray,
    probs: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Evaluates classification and calibration metrics on a demographic slice.
    """
    n_samples = len(y_sub)
    n_positives = int(np.sum(y_sub))
    n_negatives = n_samples - n_positives
    
    if n_samples < MIN_SAMPLE_THRESHOLD or n_positives < 5 or n_negatives < 5:
        return {
            "Subgroup": name,
            "Sample Count": n_samples,
            "Positive Count": n_positives,
            "ROC-AUC": "Insufficient sample size",
            "Sensitivity": "Insufficient sample size",
            "Specificity": "Insufficient sample size",
            "Brier Score": "Insufficient sample size",
            "Positive Selection Rate": "Insufficient sample size"
        }
        
    auc = float(roc_auc_score(y_sub, probs))
    preds = (probs >= threshold).astype(int)
    sens = float(recall_score(y_sub, preds))
    cm = confusion_matrix(y_sub, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    brier = float(brier_score_loss(y_sub, probs))
    selection_rate = float(np.mean(preds))
    
    return {
        "Subgroup": name,
        "Sample Count": n_samples,
        "Positive Count": n_positives,
        "ROC-AUC": round(auc, 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "Brier Score": round(brier, 4),
        "Positive Selection Rate": round(selection_rate, 4)
    }

def run_fairness_evaluation():
    print("Executing Demographic Fairness & Subgroup Robustness Evaluation...")
    predictor = CrossMorbidityRiskPredictor()
    cvd_data = load_cardiovascular_data()
    X_test = cvd_data["X_test"].copy().reset_index(drop=True)
    y_test = cvd_data["y_test"].values
    threshold = predictor.cvd_threshold
    
    # Predict probabilities using deployed CVD pipeline
    test_probs = predictor.cvd_pipeline.predict_proba(X_test)[:, 1]
    
    records = []
    
    # Overall cohort baseline
    overall = evaluate_subgroup_slice("Overall Test Cohort", X_test, y_test, test_probs, threshold)
    records.append(overall)
    
    # 1. Sex Slices (0 = Female, 1 = Male)
    female_mask = (X_test["sex"] == 0).values
    male_mask = (X_test["sex"] == 1).values
    
    female_eval = evaluate_subgroup_slice("Female (sex=0)", X_test[female_mask], y_test[female_mask], test_probs[female_mask], threshold)
    male_eval = evaluate_subgroup_slice("Male (sex=1)", X_test[male_mask], y_test[male_mask], test_probs[male_mask], threshold)
    records.append(female_eval)
    records.append(male_eval)
    
    # 2. Age Slices (<45, 45-65, >65)
    age_young_mask = (X_test["age"] < 45).values
    age_mid_mask = ((X_test["age"] >= 45) & (X_test["age"] <= 65)).values
    age_senior_mask = (X_test["age"] > 65).values
    
    young_eval = evaluate_subgroup_slice("Age < 45 years", X_test[age_young_mask], y_test[age_young_mask], test_probs[age_young_mask], threshold)
    mid_eval = evaluate_subgroup_slice("Age 45-65 years", X_test[age_mid_mask], y_test[age_mid_mask], test_probs[age_mid_mask], threshold)
    senior_eval = evaluate_subgroup_slice("Age > 65 years", X_test[age_senior_mask], y_test[age_senior_mask], test_probs[age_senior_mask], threshold)
    records.append(young_eval)
    records.append(mid_eval)
    records.append(senior_eval)
    
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "fairness_evaluation.csv", index=False)
    print("\n=== Demographic Fairness Evaluation Metrics ===")
    print(df.to_string(index=False))
    
    # Calculate Demographic Parity Difference between sexes
    if isinstance(male_eval["Positive Selection Rate"], float) and isinstance(female_eval["Positive Selection Rate"], float):
        dp_diff = abs(male_eval["Positive Selection Rate"] - female_eval["Positive Selection Rate"])
    else:
        dp_diff = "N/A"
        
    generate_fairness_markdown(df, dp_diff)
    return df

def generate_fairness_markdown(df: pd.DataFrame, dp_diff: Any):
    md = f"""# Demographic Fairness and Subgroup Robustness Report

**Evaluation Date:** September 2026  
**Evaluated Pipeline:** Deployed Cardiovascular Risk Estimator  
**Subgroups Inspected:** Biological Sex and Age Brackets  
**Sample Threshold:** Subgroups with $N < {MIN_SAMPLE_THRESHOLD}$ are explicitly flagged as insufficient.

---

## 1. Subgroup Performance Table

| Subgroup | Sample Count | Positives | ROC-AUC | Sensitivity | Specificity | Brier Score | Positive Selection Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in df.iterrows():
        md += f"| **{row['Subgroup']}** | {row['Sample Count']} | {row['Positive Count']} | {row['ROC-AUC']} | {row['Sensitivity']} | {row['Specificity']} | {row['Brier Score']} | {row['Positive Selection Rate']} |\n"

    md += f"""
---

## 2. Demographic Parity & Fairness Analysis

- **Demographic Parity Delta (Sex):** `{dp_diff if dp_diff == 'N/A' else f'{dp_diff*100:.2f}%'}` difference in positive alert selection rate between male and female cohorts.
- **Sensitivity Balance:** Sensitivity is evaluated across sex slices to ensure the model does not disproportionately under-diagnose female patients.
- **Methodological Disclaimer:** Demographic parity difference reflects base epidemiological risk prevalence differences and must not be interpreted in isolation as proof of algorithmic bias.
"""
    with open(DOCS_DIR / "FAIRNESS_EVALUATION.md", "w") as f:
        f.write(md)
    print(f"Saved fairness markdown report to {DOCS_DIR / 'FAIRNESS_EVALUATION.md'}")

if __name__ == "__main__":
    run_fairness_evaluation()
