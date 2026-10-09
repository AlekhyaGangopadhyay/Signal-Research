"""
Master Pipeline Entry Point for Few-Shot EEG Direction Detection.
Runs:
1. Data Conversion (if needed)
2. Meta-Learning Model Training
3. Leave-One-Subject-Out Cross-Validation (LOSO-CV)
4. Plot Generation (Few-Shot curve and Confusion Matrix)
"""

import os
import argparse

from src.convert_raw_data import run_conversion
from src.evaluate import run_loso_cross_validation
from src.config import TENSORS_DIR

def main():
    parser = argparse.ArgumentParser(description="Master Few-Shot EEG Pipeline")
    parser.add_argument("--convert", action="store_true", help="Force re-run raw data conversion")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs per meta-training fold")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5], help="K-shots to evaluate")
    args = parser.parse_args()

    # Step 1: Check if conversion is needed
    npz_count = len([f for f in os.listdir(TENSORS_DIR) if f.endswith('.npz')]) if os.path.exists(TENSORS_DIR) else 0
    if args.convert or npz_count < 2:
        print("[Step 1] Running Raw EEG Data Conversion & Preprocessing...")
        run_conversion()
    else:
        print(f"[Step 1] Found {npz_count} preprocessed subject tensor files. Skipping conversion.")

    # Step 2 & 3: Run Cross-Validation & Metric Evaluation
    print("\n[Step 2 & 3] Starting Leave-One-Subject-Out (LOSO) Meta-Learning & Evaluation...")
    results = run_loso_cross_validation(k_shots=args.shots, epochs_per_fold=args.epochs)
    print("\nPipeline execution successfully completed!")

if __name__ == "__main__":
    main()
