# Smart AI Health — Clinical Decision-Support Prototype

A Streamlit demonstration app for patient-profile review, longitudinal vital-sign trends, model-based cardiovascular/metabolic risk estimation, SHAP explanation highlights, what-if scenarios, and downloadable reports.

> **Important:** This is a research and educational prototype. Model scores and risk tiers are not a medical diagnosis, do not replace a licensed clinician, and must not be used as the sole basis for care. Several built-in profiles and benchmark cohorts are synthetic. Confirm any values against an authorized clinical record before use.

## Project layout

```text
SmartAIHealth_Project/
├── app.py
├── backend/                 # prediction, data, baseline, XAI, report services
├── ui/                      # Streamlit views and shared components
├── data/                    # CSV datasets and demonstration patient profiles
├── models/                  # fitted model pipelines and metadata
├── results/                 # evaluation outputs
├── scripts/                 # model training, evaluation, report checks
├── tests/                   # automated pipeline tests
├── docs/                    # dataset provenance
├── reference_screenshots/   # supplied UI references
├── requirements.txt
└── run.bat
```

## Run on Windows

Use Python 3.11 or 3.12 in a fresh environment. The fitted model files were serialized with specific Python/Scikit-learn and ML-library versions, so use the versions in `requirements.txt` where possible.

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

Or double-click `run.bat` after creating `.venv` and installing dependencies. The app normally opens at `http://localhost:8501`.

## Demo login

The login screen has separate Doctor and Patient/Caregiver roles. Default local demonstration credentials are:

- Doctor: `doctor` / `doctor123`
- Patient/Caregiver: `patient` / `patient123` (linked to `PT-101` by default)

These are intentionally simple **demo-only credentials**. Set `SMARTAI_DOCTOR_USERNAME`, `SMARTAI_DOCTOR_PASSWORD`, `SMARTAI_PATIENT_USERNAME`, `SMARTAI_PATIENT_PASSWORD`, and optionally `SMARTAI_PATIENT_PROFILE_ID` environment variables to change the local demo values. This is not production-grade authentication; use an established identity provider, hashed passwords, HTTPS, and audited authorization before handling real health information.

Doctor users can manage patient profiles and see research/administrative pages. Patient/Caregiver users are limited to a single linked profile and read-only records, trends, and report exports.

New and edited patient profiles are persisted in `data/patient_registry.json`; the built-in synthetic fixtures in `data/patient_profiles.json` are kept unchanged. This local JSON store is for the demonstration only and is not a multi-user database.

## Main sections

- **Patient Worklist:** search/filter and open a profile; active measurements and history follow the selected profile.
- **Dashboard:** active profile, key measurements, model estimate, baseline deviation and event timeline.
- **Patient Records:** doctors can edit and save validated measurements; patient/caregiver role is read-only.
- **Health Trends:** inspect generated longitudinal trajectories and personalized baselines.
- **Risk Insights:** view model-domain estimates and feature attribution summaries.
- **What-If Scenarios:** explore hypothetical changes without modifying the saved profile.
- **Population Overview:** filter the synthetic cohort and export FHIR-style JSON.
- **Reports:** download a patient report as PDF, a consultation summary as Markdown, or a FHIR RiskAssessment JSON file.
- **Settings:** adjust research parameters and privacy display mode.

## Data and model notes

See [`docs/DATA_INVENTORY.md`](docs/DATA_INVENTORY.md) for dataset provenance and limitations. The app's Datasets & ML page also runs a live quality audit. In the Pima dataset loader, zero placeholders for glucose, blood pressure, skin thickness, insulin and BMI are converted to missing values in the modeling copy and median-imputed within the training pipeline; the original CSV is preserved. If you retrain models after this preprocessing change, metrics may differ from the currently saved model metadata. Public benchmark datasets are used for model-pipeline evaluation; the main interactive models use synthetic benchmark cohorts and are not clinically validated. Model artifacts are loaded from `models/`; do not retrain unless you intend to replace the supplied artifacts.

## Training, evaluation and tests

From the project root, after installing requirements:

```bash
python scripts/train_all.py
python scripts/evaluate_all.py
python scripts/generate_reports.py
python -m unittest discover -s tests -v
```

Training can overwrite artifacts in `models/`. Back up the supplied models before running it if you need to preserve the original experiment.

## Privacy note

The included profiles are synthetic demonstration profiles. Do not put identifiable real patient information into this demo or expose it publicly without appropriate security, access controls, consent and governance.
