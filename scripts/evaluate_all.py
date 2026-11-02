"""
Master Evaluation Script for Smart AI Healthcare Monitoring System.
Usage: python scripts/evaluate_all.py
Executes:
1. ST-BTE Empirical Evaluation against 3 baselines (N=500)
2. Counterfactual Recourse Benchmark across high-risk patients (N=300)
3. CM-SRF Fusion Sensitivity Analysis
4. Demographic Fairness & Subgroup Robustness Evaluation
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.evaluate_stbte import run_stbte_experiment
from backend.counterfactual_engine import evaluate_recourse_cohort
from backend.evaluate_fusion import run_fusion_sensitivity_analysis
from backend.evaluate_fairness import run_fairness_evaluation

def main():
    print("==================================================================")
    print("   MASTER EVALUATION PIPELINE: REPRODUCIBLE BENCHMARK RUNNER      ")
    print("==================================================================")
    
    # 1. ST-BTE Evaluation
    print("\n[1/4] Running ST-BTE Longitudinal Evaluation (N=500)...")
    run_stbte_experiment(n_patients=500)
    
    # 2. Counterfactual Recourse Evaluation
    print("\n[2/4] Running Counterfactual Recourse Evaluation (N=300)...")
    evaluate_recourse_cohort(n_eval=300)
    
    # 3. Fusion Sensitivity Analysis
    print("\n[3/4] Running CM-SRF Fusion Sensitivity Analysis...")
    run_fusion_sensitivity_analysis()
    
    # 4. Fairness & Subgroup Evaluation
    print("\n[4/4] Running Fairness & Subgroup Evaluation...")
    run_fairness_evaluation()
    
    print("\nAll evaluation benchmarks successfully executed.")
    print(f"Results recorded in {ROOT_DIR / 'results'} and {ROOT_DIR / 'docs'}")

if __name__ == "__main__":
    main()
