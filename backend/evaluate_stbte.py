"""
Empirical Evaluation Experiment for Sparse Temporal Baseline Tracking Engine (ST-BTE).
Benchmarks ST-BTE against three baseline alerting methods:
1. Fixed Population Threshold (SBP >= 140 mmHg)
2. Rolling Window Mean/Std (7-point window)
3. Exponentially Weighted Moving Average (EWMA, alpha=0.2)
Simulates N=500 deterministic patient trajectories across 4 cohorts:
Stable, Gradual Drift, Abrupt Spike, High Noise.
Saves results to results/stbte_evaluation.csv, results/stbte_evaluation.json, and docs/STBTE_EVALUATION.md.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from backend.baseline_tracker import SparseTemporalBaselineTracker

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

def generate_trajectory(patient_id: int, cohort: str, rng: np.random.Generator) -> Dict[str, Any]:
    """
    Generates a single patient's 60-day asynchronous observation trajectory.
    """
    base_date = datetime(2026, 1, 1, 8, 0, 0)
    n_obs = rng.integers(24, 32)
    # Irregular observation timestamps over 60 days
    days_offsets = np.sort(rng.uniform(1.0, 59.0, size=n_obs))
    timestamps = [base_date + timedelta(days=float(d)) for d in days_offsets]
    
    # Baseline personal vitals
    base_sbp = rng.normal(120.0, 5.0)
    base_hr = rng.normal(70.0, 4.0)
    
    noise_scale = 2.5 if cohort == "high_noise" else 1.0
    sbp_noise = rng.normal(0.0, 3.0 * noise_scale, size=n_obs)
    
    sbp_values = base_sbp + sbp_noise
    
    has_event = cohort in ["gradual_drift", "abrupt_spike"]
    event_onset_day = None
    clinical_event_day = None
    
    if cohort == "gradual_drift":
        event_onset_day = float(rng.uniform(28.0, 38.0))
        clinical_event_day = event_onset_day + 14.0  # Clinical crisis at day onset + 14
        for i, d in enumerate(days_offsets):
            if d >= event_onset_day:
                # Progressive rise: +1.6 mmHg per day
                sbp_values[i] += (d - event_onset_day) * 1.6
                
    elif cohort == "abrupt_spike":
        event_onset_day = float(rng.uniform(32.0, 42.0))
        clinical_event_day = event_onset_day  # Acute crisis on onset day
        for i, d in enumerate(days_offsets):
            if d >= event_onset_day:
                sbp_values[i] += 28.0  # Acute step jump
                
    return {
        "patient_id": patient_id,
        "cohort": cohort,
        "timestamps": timestamps,
        "days_offsets": days_offsets,
        "sbp_values": sbp_values,
        "has_event": has_event,
        "event_onset_day": event_onset_day,
        "clinical_event_day": clinical_event_day
    }

def run_detector(
    detector_name: str,
    traj: Dict[str, Any],
    tracker: SparseTemporalBaselineTracker
) -> Dict[str, Any]:
    """
    Evaluates an alerting method on an individual patient trajectory.
    Returns whether an alarm occurred, alarm timestamp/day, and lead time.
    """
    timestamps = traj["timestamps"]
    days = traj["days_offsets"]
    values = traj["sbp_values"]
    has_event = traj["has_event"]
    event_onset = traj["event_onset_day"]
    clinical_event = traj["clinical_event_day"]
    
    alarm_triggered = False
    first_alarm_day = None
    
    if detector_name == "ST-BTE":
        for i in range(3, len(values)):
            hist_t = timestamps[:i]
            hist_v = list(values[:i])
            eval_res = tracker.evaluate_deviation(hist_t, hist_v, timestamps[i], values[i], "systolic_bp")
            if eval_res["is_anomalous"] and eval_res["z_score"] >= 2.0:
                alarm_triggered = True
                first_alarm_day = days[i]
                break
                
    elif detector_name == "Fixed Population Threshold":
        # SBP >= 140 mmHg
        for i in range(len(values)):
            if values[i] >= 140.0:
                alarm_triggered = True
                first_alarm_day = days[i]
                break
                
    elif detector_name == "Rolling Mean/Std (7-pt)":
        window_size = 7
        for i in range(window_size, len(values)):
            sub_window = values[i-window_size:i]
            m = np.mean(sub_window)
            s = max(np.std(sub_window), 2.0)
            if values[i] >= m + 2.0 * s:
                alarm_triggered = True
                first_alarm_day = days[i]
                break
                
    elif detector_name == "EWMA (alpha=0.2)":
        alpha = 0.2
        s_ewma = values[0]
        for i in range(1, len(values)):
            s_ewma = alpha * values[i-1] + (1 - alpha) * s_ewma
            resids = values[:i] - s_ewma
            est_std = max(np.std(resids), 2.0)
            if values[i] >= s_ewma + 2.0 * est_std:
                alarm_triggered = True
                first_alarm_day = days[i]
                break

    # Determine classification correctness
    is_tp = False
    is_fp = False
    is_fn = False
    is_tn = False
    lead_time = None
    
    if has_event:
        if alarm_triggered:
            # Alarm triggered after onset and before/at event (or up to 2 days after crisis)
            if first_alarm_day >= (event_onset - 1.0) and first_alarm_day <= (clinical_event + 2.0):
                is_tp = True
                lead_time = clinical_event - first_alarm_day
            else:
                is_fp = True  # Premature alarm before deterioration started
        else:
            is_fn = True  # Missed event
    else:
        if alarm_triggered:
            is_fp = True  # False alarm in stable/noise cohort
        else:
            is_tn = True  # Correct negative
            
    return {
        "detector": detector_name,
        "patient_id": traj["patient_id"],
        "cohort": traj["cohort"],
        "has_event": has_event,
        "alarm_triggered": alarm_triggered,
        "first_alarm_day": first_alarm_day,
        "is_tp": is_tp,
        "is_fp": is_fp,
        "is_fn": is_fn,
        "is_tn": is_tn,
        "lead_time": lead_time
    }

def run_stbte_experiment(n_patients: int = 500) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs the full empirical comparison on N=500 patients.
    """
    print(f"Executing ST-BTE Empirical Benchmark on N={n_patients} trajectories...")
    rng = np.random.default_rng(RANDOM_SEED)
    cohorts = ["stable", "gradual_drift", "abrupt_spike", "high_noise"]
    n_per_cohort = n_patients // 4
    
    tracker = SparseTemporalBaselineTracker(tau_days=21.0, z_threshold=2.0)
    detectors = [
        "ST-BTE",
        "Fixed Population Threshold",
        "Rolling Mean/Std (7-pt)",
        "EWMA (alpha=0.2)"
    ]
    
    all_trajectories = []
    pid = 1
    for c in cohorts:
        for _ in range(n_per_cohort):
            traj = generate_trajectory(pid, c, rng)
            all_trajectories.append(traj)
            pid += 1
            
    results = []
    for d in detectors:
        for traj in all_trajectories:
            eval_record = run_detector(d, traj, tracker)
            results.append(eval_record)
            
    df = pd.DataFrame(results)
    
    # Compute aggregate metrics per detector
    summary_metrics = {}
    table_rows = []
    
    for d in detectors:
        sub = df[df["detector"] == d]
        
        tp = sub["is_tp"].sum()
        fp = sub["is_fp"].sum()
        fn = sub["is_fn"].sum()
        tn = sub["is_tn"].sum()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        lead_times = sub["lead_time"].dropna().values
        if len(lead_times) > 0:
            lt_mean = float(np.mean(lead_times))
            lt_median = float(np.median(lead_times))
            lt_std = float(np.std(lead_times))
            sem = stats.sem(lead_times)
            ci95 = float(1.96 * sem) if sem > 0 else 0.0
            lt_ci = (round(lt_mean - ci95, 2), round(lt_mean + ci95, 2))
        else:
            lt_mean, lt_median, lt_std, ci95 = 0.0, 0.0, 0.0, 0.0
            lt_ci = (0.0, 0.0)
            
        summary_metrics[d] = {
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "lead_time_mean_days": round(lt_mean, 2),
            "lead_time_median_days": round(lt_median, 2),
            "lead_time_std_days": round(lt_std, 2),
            "lead_time_95ci_days": lt_ci,
            "evaluated_patients": len(sub)
        }
        
        table_rows.append({
            "Method": d,
            "Sensitivity": f"{sensitivity*100:.1f}%",
            "Specificity": f"{specificity*100:.1f}%",
            "False Positive Rate": f"{fpr*100:.1f}%",
            "Mean Lead Time (Days)": f"{lt_mean:.2f} ± {lt_std:.2f}",
            "Median Lead Time (Days)": f"{lt_median:.2f}",
            "95% CI (Days)": f"[{lt_ci[0]}, {lt_ci[1]}]"
        })
        
    summary_df = pd.DataFrame(table_rows)
    print("\n=== Empirical Benchmark Results for Longitudinal Detection ===")
    print(summary_df.to_string(index=False))
    
    # Save CSV and JSON
    df.to_csv(RESULTS_DIR / "stbte_evaluation.csv", index=False)
    with open(RESULTS_DIR / "stbte_evaluation.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)
        
    # Write docs/STBTE_EVALUATION.md
    generate_markdown_report(summary_df, summary_metrics, n_patients)
    return df, summary_metrics

def generate_markdown_report(summary_df: pd.DataFrame, summary_metrics: Dict[str, Any], n_patients: int):
    stbte = summary_metrics["ST-BTE"]
    fixed = summary_metrics["Fixed Population Threshold"]
    
    md_content = f"""# ST-BTE Empirical Evaluation Report

**Evaluation Date:** September 2026  
**Experimental Design:** Deterministic Monte Carlo simulation across $N={n_patients}$ synthetic patient trajectories.  
**Cohorts Simulated ($N={n_patients//4}$ per cohort):**
1. **Stable:** Homeostatic baseline with normal physiological jitter.
2. **Gradual Drift:** Progressive sub-clinical SBP deterioration ($+1.6\\text{{ mmHg/day}}$ over 14 days).
3. **Abrupt Spike:** Acute hemodynamic step-elevation ($+28\\text{{ mmHg}}$).
4. **High Noise:** Elevated physiological variance without systemic pathology.

---

## 1. Comparative Performance Benchmark

| Method | Sensitivity | Specificity | False Positive Rate | Mean Lead Time (Days) | Median Lead Time (Days) | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in summary_df.iterrows():
        md_content += f"| **{row['Method']}** | {row['Sensitivity']} | {row['Specificity']} | {row['False Positive Rate']} | {row['Mean Lead Time (Days)']} | {row['Median Lead Time (Days)']} | {row['95% CI (Days)']} |\n"

    md_content += f"""
---

## 2. Key Empirical Findings

1. **Lead Time Advantage:**
   - **ST-BTE** achieved an empirical mean detection lead time of **{stbte['lead_time_mean_days']} days** (median: {stbte['lead_time_median_days']} days, 95% CI: [{stbte['lead_time_95ci_days'][0]}, {stbte['lead_time_95ci_days'][1]}]) prior to acute decompensation.
   - In comparison, the **Fixed Population Threshold (SBP $\\ge 140\\text{{ mmHg}}$)** achieved a mean lead time of **{fixed['lead_time_mean_days']} days** (median: {fixed['lead_time_median_days']} days).
   - This provides an empirical lead time advantage of **{stbte['lead_time_mean_days'] - fixed['lead_time_mean_days']:.2f} days** over population thresholds on this benchmark cohort.

2. **Detection Sensitivity & False Alarms:**
   - ST-BTE achieved **{stbte['sensitivity']*100:.1f}% sensitivity** vs **{fixed['sensitivity']*100:.1f}%** for fixed threshold.
   - Specificity in high-noise/stable cohorts was **{stbte['specificity']*100:.1f}%** for ST-BTE (False Positive Rate: {stbte['false_positive_rate']*100:.1f}%).

3. **Methodological Note:**
   - Previous informal claims of a fixed "12.4 days earlier" have been replaced with these mathematically reproduced simulation figures.
"""
    with open(DOCS_DIR / "STBTE_EVALUATION.md", "w") as f:
        f.write(md_content)
    print(f"Saved ST-BTE evaluation markdown to {DOCS_DIR / 'STBTE_EVALUATION.md'}")

if __name__ == "__main__":
    run_stbte_experiment()
