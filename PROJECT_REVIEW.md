# Project Review and Changes

## Applied changes

1. **Restored the expected folder structure.** Source modules now live in `backend/` and `ui/`; datasets are in `data/`; serialized estimators and metadata are in `models/`; benchmark outputs are in `results/`; scripts and tests have their own folders. The main entry point is now `app.py`.
2. **Added a patient worklist landing page.** It lists the five supplied demonstration profiles, allows name/ID search and model-tier filtering, and opens a profile into the dashboard. The supplied reference screenshots were used for the clean teal/white clinical SaaS styling direction.
3. **Added a Reports section.** Users can download a PDF snapshot, Markdown consultation summary, or FHIR-style RiskAssessment JSON for the active profile.
4. **Added PDF report generation.** PDF bytes are generated in memory with ReportLab, so the report does not need to be written as a patient-specific file on the server.
5. **Recovered the Pima dataset formatting.** The uploaded `real_pima_diabetes.csv` was parsed with its first observation mistakenly used as column headers. `real_pima_diabetes_formatted.csv` now restores that first row and has the expected 768 rows and named columns.
6. **Updated the visual theme and launch instructions.** The theme uses a calm teal/white palette and `run.bat` now checks for a project-local virtual environment and runs `app.py` from the project root.
7. **Removed obsolete manuscript/patent checks** from the report-verification script.

## Validation performed

- All Python source files parse successfully.
- Required dataset schemas and row counts checked: 3,000 cardiovascular rows, 3,000 diabetes rows, 4,000 multi-morbidity rows, 768 Pima rows, and 303 Cleveland rows.
- Artifact verification script: **21/21 required files present**.
- Report export tests: PDF signature and FHIR structural validation passed.

## Runtime limitation to check on the target machine

The execution environment used for this review does not have Streamlit installed and has scikit-learn 1.8.0, while the supplied model artifacts were serialized with scikit-learn 1.7.2. Attempting inference in the mismatched environment raises a `SimpleImputer` compatibility error. `requirements.txt` retains scikit-learn 1.7.2 to match the uploaded model artifacts; install the project dependencies in a clean virtual environment before running the full app and end-to-end tests.

## Model choice note

The current task is structured clinical-tabular prediction, so the existing fitted scikit-learn pipelines are retained. A Hugging Face text-generation model is not a drop-in replacement for these risk models. If a language model is added later, it should be a separate, optional note-assistance feature with privacy controls and clinical review—not the source of risk scores.


## Follow-up implementation update (role access, patient persistence, and data transparency)

- Added a separate Doctor and Patient/Caregiver demo login screen with role-specific navigation. The patient/caregiver account is linked to one configured profile and gets read-only record access. Demo credentials are not suitable for production use.
- Added persistent local patient registry (`data/patient_registry.json`) for doctor-created profiles and saved edits; the original synthetic fixture file remains unchanged.
- Reworked active-patient switching to use patient IDs, update the current measurements and generated history together, and clear patient-specific form state.
- Added a doctor-only Add Patient form with measurement validation.
- Added a Datasets & ML page with live row/column/missing/duplicate/zero-placeholder audit, model algorithm inventory, saved model metadata and cleaning guidance.
- Updated Pima data loading to treat zero placeholders in five physiological measurement fields as missing in the modeling copy; preprocessing imputation remains inside the ML pipeline. Existing saved models are not automatically retrained.
- Static Python compilation and local registry save/load smoke checks passed. Full Streamlit browser interaction and model inference still need to be checked on the user's machine with the pinned dependencies.

## Methodological Corrections and Clinical Coherence Update

1. **Synergy Transparency (Rule-Based vs Learned):**
   - Explicitly clarified that cross-morbidity synergy is not an empirically learned biological phenomenon; it is a hand-set heuristic multiplier ($\gamma = 1.25$) triggered when cardiovascular and metabolic channels both exceed 40%.
   - Documented that the synthetic multi-morbidity training dataset contains an engineered interaction feature (`synergy_interaction = SBP ≥ 140 × Glucose ≥ 126`) directly tied to the synthetic label generation formula. A model trained on synthetic labels derived from this rule confirms rule capture, not clinical multi-morbidity proof.

2. **Validation Cohort Realism & Single-Disease Limitations:**
   - Clearly documented across the UI and codebase that all three primary training cohorts (`cardiovascular_data.csv`, `diabetes_data.csv`, `multi_morbidity_cohort.csv`) are 100% synthetic.
   - Clarified that the only real clinical cohorts are public Cleveland Heart Disease (N=303) and Pima Indians Diabetes (N=768), which validate isolated single diseases only. There is no real-world multi-morbidity dataset; reported synthetic AUCs (0.86 to 0.95) cannot support claims about real-world multi-morbidity diagnosis.

3. **Composite Hazard Derivation & Axiomatic Probability Bounds:**
   - Explained step-by-step how composite hazard is derived: weighted linear combination ($0.45 \cdot P_{\text{cvd}} + 0.35 \cdot P_{\text{diab}} + 0.20 \cdot P_{\text{htn}}$) compounded by the heuristic synergy multiplier ($\gamma = 1.25$).
   - Resolved the mathematical inconsistency where Eleanor Vance displayed 98.0% composite hazard despite having 99.8% cardiovascular and 99.5% diabetes risks. Enforced the axiomatic Fréchet probability lower bound $P(\text{Composite}) \ge \max(P_{\text{cvd}}, P_{\text{diab}})$ and removed the arbitrary 0.98 hard ceiling that caused the flat plateau.

4. **Counterfactual Recourse Feasibility & Locked Variable Dominance:**
   - Fixed the recourse engine and report generators so that when greedy coordinate search cannot reach the target risk threshold ($\le 25\%$), it explicitly reports failure/infeasibility rather than declaring success or advising patients to "maintain current practices" at 98%+ risk.
   - Reports the nearest achievable risk and explains that residual risk is heavily dominated and constrained by locked/unmodifiable clinical variables (such as Age 67, biological sex, genetic family history, and established chronic severity).
   - Added locked-variable analysis in the interactive what-if view explaining why sliders only shift risk incrementally.

5. **Accurate Model Metric Attribution:**
   - Fixed the Model Audit card in Risk Insights: attributed the Composite Multi-Morbidity model metrics (ROC-AUC 0.8624, PR-AUC 0.8414, Brier 0.1511, Threshold 0.4381) to the composite prediction, and separated the Cardiovascular model metrics (ROC-AUC 0.8740, PR-AUC 0.8577, Brier 0.1420) to align specifically with the SHAP attributions.
   - Added distinct tabs for Multi-Morbidity, CVD, Diabetes, and Real Benchmarks.

6. **Baseline Disease Normalization Override & Cold-Start Fix:**
   - Overcame the disease normalization failure mode: when a patient's vital signs or baseline breach absolute diagnostic thresholds (e.g. Eleanor's SBP 165 mmHg and chronic baseline ~153 mmHg), the system triggers an Absolute Clinical Threshold Override, preventing false clinical reassurance ("Stable Baseline Concordance").
   - Fixed the cold-start bug in `baseline_tracker.py` by anchoring the initial baseline at the patient's first recorded measurement rather than 0.0 mmHg.
