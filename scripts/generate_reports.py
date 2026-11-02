"""Verify the presence of the project's shipped model and evaluation artifacts."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def check_file(path: Path, label: str) -> bool:
    if path.exists():
        print(f"  [OK] {label} ({path.stat().st_size:,} bytes)")
        return True
    print(f"  [MISSING] {label}: {path.relative_to(ROOT_DIR)}")
    return False


def main() -> int:
    print("Smart AI Health — Artifact Verification")
    models = ROOT_DIR / "models"
    data = ROOT_DIR / "data"
    results = ROOT_DIR / "results"
    docs = ROOT_DIR / "docs"

    checks = [
        (models / "cvd_calibrated_pipeline.pkl", "Calibrated cardiovascular pipeline"),
        (models / "diabetes_calibrated_pipeline.pkl", "Calibrated diabetes pipeline"),
        (models / "multi_morbidity_calibrated_pipeline.pkl", "Calibrated multi-morbidity pipeline"),
        (models / "cvd_features.pkl", "Cardiovascular feature list"),
        (models / "diabetes_features.pkl", "Diabetes feature list"),
        (models / "multi_morbidity_features.pkl", "Multi-morbidity feature list"),
        (models / "training_summary.pkl", "Training summary"),
        (models / "model_comparison.csv", "Model comparison table"),
        (models / "calibration_curves.png", "Calibration curves"),
        (data / "cardiovascular_data.csv", "Synthetic cardiovascular cohort"),
        (data / "diabetes_data.csv", "Synthetic diabetes cohort"),
        (data / "multi_morbidity_cohort.csv", "Synthetic multi-morbidity cohort"),
        (data / "patient_profiles.json", "Demonstration patient profiles"),
        (data / "real_pima_diabetes_formatted.csv", "Formatted public Pima dataset"),
        (data / "real_cleveland_heart_formatted.csv", "Formatted public Cleveland dataset"),
        (results / "stbte_evaluation.json", "ST-BTE evaluation metrics"),
        (results / "recourse_metrics.json", "Counterfactual recourse metrics"),
        (results / "fusion_sensitivity.csv", "Fusion sensitivity results"),
        (results / "fairness_evaluation.csv", "Fairness evaluation results"),
        (docs / "DATA_INVENTORY.md", "Dataset provenance documentation"),
        (ROOT_DIR / "ui" / "reports_view.py", "Downloadable reports view"),
    ]
    outcomes = [check_file(path, label) for path, label in checks]
    passed = sum(outcomes)
    print(f"\nVerification complete: {passed}/{len(outcomes)} required artifacts present.")
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
