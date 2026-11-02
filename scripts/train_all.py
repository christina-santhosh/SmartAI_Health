"""
Master Training Script for Smart AI Healthcare Monitoring System.
Usage: python scripts/train_all.py
Executes:
1. Benchmark model training across all clinical domains
2. Out-of-fold threshold tuning via Youden's J
3. Probability calibration (Platt Sigmoid)
4. Model artifacts and metadata serialization
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.model_trainer import train_and_benchmark_all

def main():
    print("==================================================================")
    print("   MASTER TRAINING PIPELINE: SMART HEALTHCARE AI (HARDENED)       ")
    print("==================================================================")
    summary = train_and_benchmark_all()
    print("\nMaster training successfully completed.")
    print(f"Artifacts saved to {ROOT_DIR / 'models'}")

if __name__ == "__main__":
    main()
