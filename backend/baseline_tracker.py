"""
Invention 1: Sparse Temporal Baseline Tracking Engine (ST-BTE)
Patented Algorithmic Methodology for reconstructing personalized physiological trajectories
and detecting sub-clinical drift from irregular, asynchronous patient health logs.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

# Standard diagnostic clinical reference limits for absolute overrides
CLINICAL_ABSOLUTE_LIMITS = {
    "systolic_bp": [
        {"threshold": 160.0, "condition": "gte", "label": "Severe Hypertension / Crisis (>= 160 mmHg)", "severity": "CRITICAL"},
        {"threshold": 140.0, "condition": "gte", "label": "Stage 2 Hypertension (>= 140 mmHg)", "severity": "HIGH"},
        {"threshold": 130.0, "condition": "gte", "label": "Stage 1 Hypertension (>= 130 mmHg)", "severity": "ELEVATED"},
    ],
    "diastolic_bp": [
        {"threshold": 100.0, "condition": "gte", "label": "Severe Diastolic HTN / Crisis (>= 100 mmHg)", "severity": "CRITICAL"},
        {"threshold": 90.0, "condition": "gte", "label": "Stage 2 Diastolic HTN (>= 90 mmHg)", "severity": "HIGH"},
        {"threshold": 80.0, "condition": "gte", "label": "Stage 1 Diastolic HTN (>= 80 mmHg)", "severity": "ELEVATED"},
    ],
    "fasting_glucose": [
        {"threshold": 126.0, "condition": "gte", "label": "Diabetes Diagnostic Threshold (>= 126 mg/dL)", "severity": "HIGH"},
        {"threshold": 100.0, "condition": "gte", "label": "Impaired Fasting Glucose (>= 100 mg/dL)", "severity": "ELEVATED"},
    ],
    "resting_hr": [
        {"threshold": 100.0, "condition": "gte", "label": "Tachycardia (> 100 bpm)", "severity": "ELEVATED"},
        {"threshold": 60.0, "condition": "lte", "label": "Bradycardia (< 60 bpm)", "severity": "ELEVATED"},
    ],
    "spo2": [
        {"threshold": 90.0, "condition": "lte", "label": "Severe Hypoxemia (<= 90%)", "severity": "CRITICAL"},
        {"threshold": 95.0, "condition": "lte", "label": "Hypoxemia Alert (< 95%)", "severity": "HIGH"},
    ]
}

def _lookup_clinical_breach(metric_key: str, val: float, base_mean: float) -> Optional[Dict[str, Any]]:
    """Checks whether current value or chronic baseline breaches absolute clinical limits."""
    clean_key = metric_key.lower().replace(" ", "_").replace("systolic_blood_pressure", "systolic_bp").replace("diastolic_blood_pressure", "diastolic_bp").replace("resting_heart_rate", "resting_hr").replace("blood_glucose", "fasting_glucose")
    limits = CLINICAL_ABSOLUTE_LIMITS.get(clean_key, [])
    for lim in limits:
        thresh = lim["threshold"]
        cond = lim["condition"]
        if cond == "gte" and (val >= thresh or base_mean >= thresh):
            return lim
        elif cond == "lte" and (val <= thresh or base_mean <= thresh):
            return lim
    return None

class SparseTemporalBaselineTracker:
    """
    Self-Calibrating Temporal Baseline Engine for sparse, asynchronous health check-ins.
    Models patient-specific physiological envelopes using exponentially weighted
    temporal decay kernels and dynamic standard deviation corridors.
    """
    def __init__(self, tau_days: float = 21.0, z_threshold: float = 2.0, epsilon: float = 1e-4):
        """
        :param tau_days: Temporal half-life / memory decay parameter in days.
        :param z_threshold: Number of standard deviations indicating sub-clinical deviation.
        :param epsilon: Numerical stability regularizer.
        """
        self.tau = tau_days
        self.z_threshold = z_threshold
        self.epsilon = epsilon
        
    def compute_baseline_corridor(
        self,
        timestamps: List[datetime],
        values: List[float],
        target_time: Optional[datetime] = None,
        robust_damping: bool = True
    ) -> Dict[str, Any]:
        """
        Computes the personalized baseline mean, standard deviation, and dynamic corridor
        at a target evaluation timestamp based on sparse historical logs.
        Includes robust variance estimation, outlier damping, and minimum history checks.
        """
        if not timestamps or not values or len(timestamps) != len(values):
            return {
                "mean": 0.0,
                "std": 0.0,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "effective_sample_size": 0.0,
                "is_reliable": False,
                "status": "INSUFFICIENT_HISTORY",
                "status_message": "No historical observations available to establish baseline."
            }
            
        if target_time is None:
            target_time = max(timestamps)
            
        t_target_days = target_time.timestamp() / 86400.0
        t_obs_days = np.array([t.timestamp() / 86400.0 for t in timestamps])
        deltas = np.maximum(0.0, t_target_days - t_obs_days)
        
        # Exponential temporal decay weighting kernel: w_i = exp(-delta / tau)
        weights = np.exp(-deltas / self.tau)
        sum_weights = float(np.sum(weights))
        n_obs = len(values)
        vals = np.array(values, dtype=float)
        
        if n_obs < 3 or sum_weights < 1e-4:
            mean = float(np.mean(vals))
            std = float(np.std(vals)) if n_obs > 1 else 2.0
            std = max(std, 1.5)
            return {
                "mean": round(mean, 2),
                "std": round(std, 2),
                "lower_bound": round(mean - self.z_threshold * std, 2),
                "upper_bound": round(mean + self.z_threshold * std, 2),
                "effective_sample_size": round(sum_weights, 2),
                "is_reliable": False,
                "status": "INSUFFICIENT_HISTORY",
                "status_message": f"Only {n_obs} observation(s) recorded; minimum of 3 required for a reliable baseline."
            }
            
        # First-pass weighted mean and variance
        mean = float(np.sum(weights * vals) / sum_weights)
        raw_variance = float(np.sum(weights * ((vals - mean) ** 2)) / sum_weights)
        raw_std = max(float(np.sqrt(raw_variance)), 1.5)
        
        # Robust damping: prevent isolated acute abnormal episodes from corrupting baseline
        if robust_damping and n_obs >= 5:
            z_raw = np.abs(vals - mean) / (raw_std + self.epsilon)
            # Huber-style down-weighting for extreme spikes (|z| > 2.5)
            damp_mask = z_raw > 2.5
            if np.any(damp_mask):
                damp_factors = np.ones_like(weights)
                damp_factors[damp_mask] = 2.5 / (z_raw[damp_mask] + 1e-6)
                adj_weights = weights * damp_factors
                sum_adj = float(np.sum(adj_weights))
                if sum_adj > 1e-4:
                    mean = float(np.sum(adj_weights * vals) / sum_adj)
                    # Unbiased weighted variance estimation
                    sum_sq = float(np.sum(adj_weights ** 2))
                    denom = sum_adj - (sum_sq / sum_adj)
                    if denom > 1e-4:
                        variance = float(np.sum(adj_weights * ((vals - mean) ** 2)) / denom)
                    else:
                        variance = float(np.sum(adj_weights * ((vals - mean) ** 2)) / sum_adj)
                    raw_std = max(float(np.sqrt(variance)), 1.5)
                    sum_weights = sum_adj
        else:
            sum_sq = float(np.sum(weights ** 2))
            denom = sum_weights - (sum_sq / sum_weights)
            if denom > 1e-4:
                variance = float(np.sum(weights * ((vals - mean) ** 2)) / denom)
            else:
                variance = raw_variance
            raw_std = max(float(np.sqrt(variance)), 1.5)

        is_reliable = (n_obs >= 3) and (sum_weights >= 2.0)
        status = "RELIABLE" if is_reliable else "LOW_CONFIDENCE"
        status_msg = "Personal baseline established." if is_reliable else "Sparse temporal weight; baseline has low statistical confidence."

        return {
            "mean": round(mean, 2),
            "std": round(raw_std, 2),
            "lower_bound": round(mean - self.z_threshold * raw_std, 2),
            "upper_bound": round(mean + self.z_threshold * raw_std, 2),
            "effective_sample_size": round(float(sum_weights), 2),
            "is_reliable": is_reliable,
            "status": status,
            "status_message": status_msg
        }

    def evaluate_deviation(
        self,
        historical_timestamps: List[datetime],
        historical_values: List[float],
        current_timestamp: datetime,
        current_value: float,
        metric_name: str = "vital_sign",
        unit: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluates whether a newly logged vital sign constitutes a statistically significant
        deviation from the patient's self-calibrated personal baseline.
        Applies absolute clinical threshold overrides to prevent normalizing chronic pathology.
        """
        if not historical_timestamps or not historical_values:
            # Cold-start anchor: anchor baseline at current observation to prevent 0.0 plotting artifact
            mean = float(current_value)
            std = 3.0
            corridor = {
                "mean": round(mean, 2),
                "std": round(std, 2),
                "lower_bound": round(mean - self.z_threshold * std, 2),
                "upper_bound": round(mean + self.z_threshold * std, 2),
                "effective_sample_size": 1.0,
                "is_reliable": False,
                "status": "INSUFFICIENT_HISTORY",
                "status_message": "Baseline anchored at initial observation; accumulating longitudinal logs."
            }
        else:
            corridor = self.compute_baseline_corridor(historical_timestamps, historical_values, current_timestamp)
            mean = corridor["mean"]
            std = corridor["std"]
            
        is_reliable = corridor.get("is_reliable", True)
        
        # Check absolute clinical threshold override
        breached = _lookup_clinical_breach(metric_name, current_value, mean)
        clinical_override = breached is not None
        breached_label = breached["label"] if breached else ""
        
        # Compute normalized z-score excursion
        if not is_reliable and corridor.get("status") == "INSUFFICIENT_HISTORY" and not historical_timestamps:
            z_score = 0.0
            is_anomalous = False
            severity = "Cold Start (Anchored Initial Vital)"
            narrative = f"Initial baseline establishing: 1 initial observation recorded ({current_value} {unit})."
        else:
            z_score = (current_value - mean) / (std + self.epsilon)
            is_anomalous = abs(z_score) >= self.z_threshold
            
            if is_anomalous:
                direction = "elevated above" if z_score > 0 else "depressed below"
                severity = f"Baseline Deviation (|z| = {abs(z_score):.1f})"
                delta = current_value - mean
                narrative = (
                    f"Observation ({current_value} {unit}) is {direction} personal baseline "
                    f"({mean:.1f} ± {std:.1f} {unit}, Δ = {delta:+.1f}, z = {z_score:+.2f}). "
                    f"Statistical excursion from expected corridor; evaluate alongside clinical symptoms."
                )
            else:
                if clinical_override:
                    severity = f"Clinical Override: Pathological Baseline"
                    narrative = (
                        f"CLINICAL THRESHOLD OVERRIDE: Observation ({current_value} {unit}) falls within personal baseline corridor "
                        f"({corridor['lower_bound']} to {corridor['upper_bound']} {unit}, z = {z_score:+.2f}), but breaches absolute clinical safety thresholds "
                        f"({breached_label}). Chronically elevated baselines must NOT normalize disease or delay clinical intervention."
                    )
                else:
                    severity = "Within Baseline Corridor"
                    narrative = (
                        f"Observation ({current_value} {unit}) is within personal baseline corridor "
                        f"({corridor['lower_bound']} to {corridor['upper_bound']} {unit}, z = {z_score:+.2f})."
                    )
            
        return {
            "metric": metric_name,
            "unit": unit,
            "current_value": current_value,
            "baseline_mean": mean,
            "baseline_std": std,
            "corridor": (corridor["lower_bound"], corridor["upper_bound"]),
            "z_score": round(float(z_score), 2),
            "is_anomalous": is_anomalous,
            "is_reliable": is_reliable,
            "clinical_override_active": clinical_override,
            "breached_clinical_limit": breached_label if clinical_override else None,
            "clinical_status": severity,
            "narrative": narrative
        }

    def generate_patient_longitudinal_trajectory(
        self,
        scenario: str = "acute_drift",
        days: int = 60,
        n_observations: int = 25
    ) -> pd.DataFrame:
        """
        Generates rich, realistic longitudinal time-series data of asynchronous patient logs
        demonstrating baseline tracking vs acute drift.
        """
        rng = np.random.default_rng(42)
        base_date = datetime.now() - timedelta(days=days)
        
        # Irregular log timestamps (some intervals 1 day, some 4 days)
        interval_offsets = np.sort(rng.uniform(0, days, size=n_observations))
        dates = [base_date + timedelta(days=float(o)) for o in interval_offsets]
        
        # Baseline vital values
        if scenario == "stable_healthy":
            base_sbp = 118.0
            base_hr = 68.0
            base_glucose = 88.0
            drift_onset = 999 # No drift
        elif scenario == "acute_drift":
            base_sbp = 124.0
            base_hr = 70.0
            base_glucose = 92.0
            drift_onset = days - 14 # Drift starts 14 days ago
        elif scenario == "diabetic_decompensation":
            base_sbp = 130.0
            base_hr = 75.0
            base_glucose = 115.0
            drift_onset = days - 20
        else:
            base_sbp = 120.0
            base_hr = 72.0
            base_glucose = 90.0
            drift_onset = 999
            
        records = []
        for d in dates:
            days_from_start = (d - base_date).days
            
            # Normal physiological circadian/random jitter
            sbp = rng.normal(base_sbp, 3.5)
            hr = rng.normal(base_hr, 3.0)
            glucose = rng.normal(base_glucose, 4.0)
            
            # Apply progressive drift if past drift onset
            if days_from_start > drift_onset:
                progression = (days_from_start - drift_onset) / (days - drift_onset)
                if scenario == "acute_drift":
                    sbp += progression * 32.0 # SBP climbs towards 156
                    hr += progression * 22.0  # HR climbs towards 92
                elif scenario == "diabetic_decompensation":
                    glucose += progression * 85.0 # Glucose spikes to 200
                    sbp += progression * 18.0
                    
            records.append({
                "timestamp": d,
                "date_str": d.strftime("%Y-%m-%d %H:%M"),
                "systolic_bp": round(float(sbp), 1),
                "resting_hr": round(float(hr), 1),
                "fasting_glucose": round(float(glucose), 1)
            })
            
        df = pd.DataFrame(records)
        return df
