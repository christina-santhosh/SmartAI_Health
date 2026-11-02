"""
Data loading module for synthetic and public benchmark datasets.
Strictly separates raw data loading and stratified splitting from preprocessing.
Preprocessing, imputation, and scaling are encapsulated exclusively inside sklearn Pipelines.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Feature definitions for synthetic benchmark cohorts
CVD_FEATURES: List[str] = [
    "age", "sex", "bmi", "systolic_bp", "diastolic_bp", "resting_hr", 
    "spo2", "cholesterol", "hdl", "ldl", "fasting_glucose", 
    "smoking_status", "physical_activity_min", "sleep_hours", 
    "family_history_cvd", "chest_pain", "shortness_of_breath"
]

DIABETES_FEATURES: List[str] = [
    "age", "sex", "bmi", "fasting_glucose", "hba1c", "insulin", 
    "systolic_bp", "diastolic_bp", "physical_activity_min", 
    "family_history_diabetes", "diet_score"
]

MULTI_MORBIDITY_FEATURES: List[str] = [
    "age", "sex", "bmi", "systolic_bp", "diastolic_bp", "resting_hr",
    "spo2", "fasting_glucose", "hba1c", "cholesterol", "hdl", "ldl",
    "smoking_status", "physical_activity_min", "sleep_hours",
    "family_history_cvd", "family_history_diabetes", "synergy_interaction"
]

def load_cardiovascular_data(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Loads raw synthetic cardiovascular dataset and returns an untouched stratified split.
    Zero preprocessing is applied here to guarantee zero data leakage.
    """
    df = pd.read_csv(DATA_DIR / "cardiovascular_data.csv")
    X = df[CVD_FEATURES].copy()
    y = df["target_cvd"].copy().astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "features": CVD_FEATURES,
        "dataset_name": "Synthetic Cardiovascular Cohort",
        "sample_count": len(df)
    }

def load_diabetes_data(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Loads raw synthetic diabetes dataset and returns an untouched stratified split.
    """
    df = pd.read_csv(DATA_DIR / "diabetes_data.csv")
    X = df[DIABETES_FEATURES].copy()
    y = df["target_diabetes"].copy().astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "features": DIABETES_FEATURES,
        "dataset_name": "Synthetic Diabetes Cohort",
        "sample_count": len(df)
    }

def load_multi_morbidity_data(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Loads raw multi-morbidity cohort dataset and returns an untouched stratified split.
    """
    df = pd.read_csv(DATA_DIR / "multi_morbidity_cohort.csv")
    X = df[MULTI_MORBIDITY_FEATURES].copy()
    y = df["target_acute_event"].copy().astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "features": MULTI_MORBIDITY_FEATURES,
        "dataset_name": "Synthetic Multi-Morbidity Cohort",
        "sample_count": len(df),
        "full_df": df
    }

def load_real_pima_dataset(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Loads real public Pima Indians Diabetes dataset.
    """
    df = pd.read_csv(DATA_DIR / "real_pima_diabetes_formatted.csv")
    # In the Pima benchmark, zero is not a plausible measurement for these
    # physiological variables and is commonly used as a missing-value placeholder.
    # Keep the raw file unchanged; convert placeholders to NaN in the modeling copy.
    zero_as_missing = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    for column in zero_as_missing:
        if column in df.columns:
            df.loc[df[column] == 0, column] = np.nan
    features = [c for c in df.columns if c != "target_diabetes"]
    X = df[features].copy()
    y = df["target_diabetes"].copy().astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "features": features,
        "dataset_name": "Public NIDDK Pima Indians Diabetes",
        "sample_count": len(df)
    }

def load_real_cleveland_dataset(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Loads real public Cleveland Heart Disease dataset.
    """
    df = pd.read_csv(DATA_DIR / "real_cleveland_heart_formatted.csv")
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = df[features].copy()
    y = df["target_cvd"].copy().astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "features": features,
        "dataset_name": "Public UCI Cleveland Heart Disease",
        "sample_count": len(df)
    }
