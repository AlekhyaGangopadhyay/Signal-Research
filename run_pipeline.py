"""
Master Pipeline Entry Point for Few-Shot EEG Direction Detection.
Runs:
1. Data-validity checks (src.sanity)
2. Data conversion (if needed): all pages, 256 Hz, stimulus-locked windows
3. Classical baselines (src.baselines)
4. Leave-One-Subject-Out few-shot evaluation of the ProtoNet, with shuffled-label control (src.evaluate)
"""

import argparse
import glob
import os
import subprocess
import sys

from src.config import TENSORS_DIR, NUM_EPOCHS, ensure_dirs
from src.convert_raw_data import run_conversion
from src.evaluate import run_loso_cross_validation


def run_module(module, *args):
    print(f"\n$ python -m {module} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-u", "-m", module, *args], check=True)


def main():
    parser = argparse.ArgumentParser(description="Master Few-Shot EEG Pipeline")
    parser.add_argument("--convert", action="store_true", help="Force re-run of raw data conversion")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip the data-validity checks")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Max epochs per meta-training fold")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5, 10], help="K-shots to evaluate")
    parser.add_argument("--model", default="spatial_temporal", choices=["spatial_temporal", "eegnet"])
    args = parser.parse_args()
    ensure_dirs()

    if not args.skip_sanity:
        print("[Step 1] Data-validity checks", flush=True)
        run_module("src.sanity")

    npz = glob.glob(os.path.join(TENSORS_DIR, "subject_*.npz"))
    if args.convert or len(npz) < 2:
        print("\n[Step 2] Converting raw data", flush=True)
        run_conversion()
    else:
        print(f"\n[Step 2] Found {len(npz)} subject files; skipping conversion (use --convert to rebuild)", flush=True)

    if not args.skip_baselines:
        print("\n[Step 3] Classical baselines", flush=True)
        run_module("src.baselines")

    print("\n[Step 4] LOSO few-shot evaluation", flush=True)
    run_loso_cross_validation(k_shots=args.shots, epochs_per_fold=args.epochs, model_type=args.model)
    print("\nPipeline finished. See results/.", flush=True)


if __name__ == "__main__":
    main()
