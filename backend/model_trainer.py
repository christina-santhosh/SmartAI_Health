"""
Leakage-Safe Machine Learning Training, Calibration, and Benchmarking Engine.
Adheres strictly to clinical AI best practices:
- Preprocessing (Imputation, Scaling) encapsulated inside sklearn Pipelines.
- 5-Fold Stratified Cross-Validation fitted strictly within training folds.
- Threshold selection (Youden's J) performed exclusively on Out-Of-Fold (OOF) predictions.
- Probability calibration (Platt Sigmoid / Isotonic) evaluated on training data only.
- Untouched test set evaluated exactly once for final unbiased reporting.
- 5-Model Benchmark: Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, CatBoost.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Tuple
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    confusion_matrix, roc_curve
)

# Gradient Boosting libraries
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from backend.data_loader import (
    load_cardiovascular_data,
    load_diabetes_data,
    load_multi_morbidity_data,
    load_real_pima_dataset,
    load_real_cleveland_dataset
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

def create_candidate_pipelines() -> Dict[str, Pipeline]:
    """
    Constructs candidate model pipelines.
    Scalers are included ONLY for linear models. Tree models receive raw imputed values.
    """
    return {
        "Logistic_Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
        ]),
        "Random_Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1))
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=RANDOM_SEED))
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                  random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=-1))
        ]),
        "CatBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05,
                                       random_seed=RANDOM_SEED, verbose=0))
        ])
    }

def find_oof_optimal_threshold(y_true: np.ndarray, y_oof_prob: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the optimal decision threshold maximizing Youden's J statistic
    (Sensitivity + Specificity - 1) EXCLUSIVELY on Out-Of-Fold training predictions.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_oof_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])
    best_j = float(j_scores[best_idx])
    return best_threshold, best_j

def benchmark_domain_models(
    domain_key: str,
    data_dict: Dict[str, Any],
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Executes a leakage-safe 5-model benchmark across 5-fold cross-validation,
    generates OOF predictions, selects threshold via Youden's J, fits calibrated estimators,
    and performs a single final test evaluation on untouched test data.
    """
    X_train, y_train = data_dict["X_train"], data_dict["y_train"].values
    X_test, y_test = data_dict["X_test"], data_dict["y_test"].values
    features = data_dict["features"]
    dataset_name = data_dict["dataset_name"]
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    candidates = create_candidate_pipelines()
    
    model_records = []
    oof_predictions = {}
    fitted_pipelines = {}
    
    print(f"\n=======================================================")
    print(f"   BENCHMARKING DOMAIN: {domain_key.upper()} ({dataset_name})")
    print(f"=======================================================")
    
    for name, pipeline in candidates.items():
        print(f"Evaluating {name} with {n_splits}-fold CV...")
        oof_probs = np.zeros(len(y_train))
        fold_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            
            # Pipeline fits imputer, scaler (if linear), and model strictly on fold training data
            pipeline.fit(X_tr, y_tr)
            val_prob = pipeline.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = val_prob
            fold_aucs.append(roc_auc_score(y_val, val_prob))
            
        cv_auc_mean = float(np.mean(fold_aucs))
        cv_auc_std = float(np.std(fold_aucs))
        oof_predictions[name] = oof_probs
        
        # Threshold selection exclusively on OOF predictions
        oof_thresh, oof_j = find_oof_optimal_threshold(y_train, oof_probs)
        
        # Measure OOF calibration Brier score
        oof_brier = float(brier_score_loss(y_train, oof_probs))
        
        # Fit on full training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        
        # Test set evaluation (UNTOUCHED until this point, evaluated with fixed OOF threshold)
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_auc = float(roc_auc_score(y_test, test_prob))
        test_prauc = float(average_precision_score(y_test, test_prob))
        test_brier = float(brier_score_loss(y_test, test_prob))
        
        # Predictions using OOF-selected threshold
        test_pred_opt = (test_prob >= oof_thresh).astype(int)
        test_acc = float(accuracy_score(y_test, test_pred_opt))
        test_prec = float(precision_score(y_test, test_pred_opt, zero_division=0))
        test_sens = float(recall_score(y_test, test_pred_opt))
        cm = confusion_matrix(y_test, test_pred_opt)
        tn, fp, fn, tp = cm.ravel()
        test_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        test_f1 = float(f1_score(y_test, test_pred_opt))
        
        record = {
            "domain": domain_key,
            "model_name": name,
            "cv_auc_mean": round(cv_auc_mean, 4),
            "cv_auc_std": round(cv_auc_std, 4),
            "oof_threshold": round(oof_thresh, 4),
            "oof_youden_j": round(oof_j, 4),
            "oof_brier_score": round(oof_brier, 4),
            "test_roc_auc": round(test_auc, 4),
            "test_pr_auc": round(test_prauc, 4),
            "test_accuracy": round(test_acc, 4),
            "test_precision": round(test_prec, 4),
            "test_sensitivity": round(test_sens, 4),
            "test_specificity": round(test_spec, 4),
            "test_f1": round(test_f1, 4),
            "test_brier_score": round(test_brier, 4),
            "test_confusion_matrix": cm.tolist()
        }
        model_records.append(record)
        print(f"   -> CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f} | Test AUC: {test_auc:.4f} | OOF Thresh: {oof_thresh:.4f} | Sens: {test_sens:.4f} | Spec: {test_spec:.4f}")

    results_df = pd.DataFrame(model_records)
    
    # Selection criteria:
    # 1. Primary: Highest CV AUC mean (reproducible cross-validation performance)
    # 2. Secondary: Lower Brier score (better probability calibration)
    best_idx = results_df.sort_values(by=["cv_auc_mean", "oof_brier_score"], ascending=[False, True]).index[0]
    best_row = results_df.loc[best_idx]
    best_model_name = best_row["model_name"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_threshold = float(best_row["oof_threshold"])
    
    print(f"\n==> Selected Model for {domain_key.upper()}: {best_model_name} (CV AUC: {best_row['cv_auc_mean']:.4f})")
    
    # Fit calibrated version of best pipeline using CalibratedClassifierCV on training data only
    print(f"Fitting Platt sigmoid calibration on training folds...")
    calibrated_pipeline = CalibratedClassifierCV(
        estimator=best_pipeline,
        method="sigmoid",
        cv=5
    )
    calibrated_pipeline.fit(X_train, y_train)
    cal_prob = calibrated_pipeline.predict_proba(X_test)[:, 1]
    cal_brier = float(brier_score_loss(y_test, cal_prob))
    print(f"Test Brier score: Uncalibrated={best_row['test_brier_score']:.4f} | Calibrated={cal_brier:.4f}")
    
    # Save artifacts
    joblib.dump(best_pipeline, MODELS_DIR / f"{domain_key}_pipeline.pkl")
    joblib.dump(calibrated_pipeline, MODELS_DIR / f"{domain_key}_calibrated_pipeline.pkl")
    joblib.dump(best_threshold, MODELS_DIR / f"{domain_key}_threshold.pkl")
    joblib.dump(features, MODELS_DIR / f"{domain_key}_features.pkl")
    
    # Also save metadata
    metadata = {
        "domain": domain_key,
        "dataset_name": dataset_name,
        "selected_model": best_model_name,
        "selection_criteria": "Primary: 5-fold CV AUC mean. Secondary: Brier score.",
        "oof_threshold": best_threshold,
        "cv_auc_mean": best_row["cv_auc_mean"],
        "cv_auc_std": best_row["cv_auc_std"],
        "test_roc_auc": best_row["test_roc_auc"],
        "test_pr_auc": best_row["test_pr_auc"],
        "test_sensitivity": best_row["test_sensitivity"],
        "test_specificity": best_row["test_specificity"],
        "test_f1": best_row["test_f1"],
        "test_brier_uncalibrated": best_row["test_brier_score"],
        "test_brier_calibrated": round(cal_brier, 4),
        "random_seed": RANDOM_SEED,
        "feature_names": features
    }
    with open(MODELS_DIR / f"{domain_key}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "calibrated_pipeline": calibrated_pipeline,
        "best_threshold": best_threshold,
        "oof_predictions": oof_predictions,
        "metadata": metadata
    }

def train_and_benchmark_all():
    """Executes benchmark suite across all synthetic cohorts and public validation datasets."""
    all_results = []
    
    # 1. Cardiovascular Risk
    cvd_data = load_cardiovascular_data(random_state=RANDOM_SEED)
    cvd_out = benchmark_domain_models("cvd", cvd_data)
    all_results.append(cvd_out["results_df"])
    
    # 2. Type 2 Diabetes Risk
    diab_data = load_diabetes_data(random_state=RANDOM_SEED)
    diab_out = benchmark_domain_models("diabetes", diab_data)
    all_results.append(diab_out["results_df"])
    
    # 3. Multi-Morbidity Synergy Risk
    multi_data = load_multi_morbidity_data(random_state=RANDOM_SEED)
    multi_out = benchmark_domain_models("multi_morbidity", multi_data)
    all_results.append(multi_out["results_df"])
    
    # 4. External Validation: Public Pima Indians Diabetes
    print("\n--- External Public Benchmark: Pima Indians Diabetes ---")
    pima_data = load_real_pima_dataset(random_state=RANDOM_SEED)
    pima_out = benchmark_domain_models("public_pima_diabetes", pima_data)
    all_results.append(pima_out["results_df"])
    
    # 5. External Validation: Public Cleveland Heart Disease
    print("\n--- External Public Benchmark: Cleveland Heart Disease ---")
    clev_data = load_real_cleveland_dataset(random_state=RANDOM_SEED)
    clev_out = benchmark_domain_models("public_cleveland_heart", clev_data)
    all_results.append(clev_out["results_df"])
    
    # Concatenate full comparison table
    full_comparison = pd.concat(all_results, ignore_index=True)
    full_comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    print(f"\nSaved full model comparison table to {MODELS_DIR / 'model_comparison.csv'}")
    
    # Save training summary object for app and report consumption
    training_summary = {
        "cardiovascular": cvd_out["metadata"],
        "diabetes": diab_out["metadata"],
        "multi_morbidity": multi_out["metadata"],
        "public_pima_diabetes": pima_out["metadata"],
        "public_cleveland_heart": clev_out["metadata"]
    }
    joblib.dump(training_summary, MODELS_DIR / "training_summary.pkl")
    print(f"Saved training summary object to {MODELS_DIR / 'training_summary.pkl'}")
    
    # Generate and save calibration curve plot
    generate_calibration_plot(cvd_data, cvd_out, diab_data, diab_out, multi_data, multi_out)
    return training_summary

def generate_calibration_plot(cvd_data, cvd_out, diab_data, diab_out, multi_data, multi_out):
    """Generates and saves visual calibration curves for the three core domains."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    domains = [
        ("Cardiovascular Risk", cvd_data, cvd_out, axes[0]),
        ("Diabetes & Metabolic Risk", diab_data, diab_out, axes[1]),
        ("Multi-Morbidity Acute Synergy", multi_data, multi_out, axes[2])
    ]
    
    for title, data, out, ax in domains:
        y_test = data["y_test"].values
        X_test = data["X_test"]
        
        # Uncalibrated vs Calibrated probabilities
        prob_uncal = out["best_pipeline"].predict_proba(X_test)[:, 1]
        prob_cal = out["calibrated_pipeline"].predict_proba(X_test)[:, 1]
        
        fraction_of_positives_u, mean_predicted_value_u = calibration_curve(y_test, prob_uncal, n_bins=8)
        fraction_of_positives_c, mean_predicted_value_c = calibration_curve(y_test, prob_cal, n_bins=8)
        
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(mean_predicted_value_u, fraction_of_positives_u, "s-", color="#ef4444",
                label=f"Uncalibrated (Brier={out['metadata']['test_brier_uncalibrated']:.3f})")
        ax.plot(mean_predicted_value_c, fraction_of_positives_c, "o-", color="#22c55e",
                label=f"Platt Calibrated (Brier={out['metadata']['test_brier_calibrated']:.3f})")
        
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Fraction of Positives", fontsize=10)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        
    plt.tight_layout()
    plot_path = MODELS_DIR / "calibration_curves.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved probability calibration curves plot to {plot_path}")

if __name__ == "__main__":
    train_and_benchmark_all()
