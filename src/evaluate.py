"""
Evaluation and Leave-One-Subject-Out Cross-Validation (LOSO-CV) Engine.
Generates metrics (Accuracy, Cohen's Kappa, ITR) and publication plots.
"""

import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from src.config import (
    TENSORS_DIR, RESULTS_DIR, CHECKPOINTS_DIR,
    N_WAY, DIRECTIONS, TRIAL_SAMPLES, SAMPLING_RATE
)
from src.dataset import EEGSubjectDataset
from src.models import SpatialTemporalProtoNet
from src.train_fewshot import train_meta_model

def calculate_itr(n_classes, accuracy_ratio, trial_duration_sec):
    """
    Computes Information Transfer Rate (ITR) in bits per minute (Wolpaw et al.).
    """
    p = np.clip(accuracy_ratio, 1e-4, 0.9999)
    n = n_classes
    if p <= (1.0 / n):
        b = 0.0
    else:
        b = np.log2(n) + p * np.log2(p) + (1.0 - p) * np.log2((1.0 - p) / (n - 1.0))
    trials_per_min = 60.0 / trial_duration_sec
    return max(0.0, b * trials_per_min)

def evaluate_fewshot_on_target(model, target_dataset, k_shot=5, n_eval_episodes=50, device="cpu"):
    """
    Evaluates trained meta-model on unseen target subject using k-shot calibration.
    """
    model.eval()
    all_preds = []
    all_targets = []
    accs = []

    with torch.no_grad():
        for _ in range(n_eval_episodes):
            s_x, s_y, q_x, q_y = target_dataset.sample_fewshot_episode(n_way=4, k_shot=k_shot, q_query=5)
            s_x, q_x = s_x.to(device), q_x.to(device)

            # 1. Compute target subject prototypes from support shots
            z_s = model(s_x).view(4, k_shot, -1).mean(dim=1) # [4, D]
            z_q = model(q_x)                                 # [N_query, D]

            # 2. Distance-based classification
            dists = torch.cdist(z_q, z_s, p=2) ** 2
            preds = torch.argmin(dists, dim=1).cpu().numpy()
            targets = q_y.numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)
            accs.append(np.mean(preds == targets))

    mean_acc = np.mean(accs) * 100
    kappa = cohen_kappa_score(all_targets, all_preds)
    itr = calculate_itr(4, mean_acc / 100.0, TRIAL_SAMPLES / SAMPLING_RATE)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3])

    return {
        'accuracy': mean_acc,
        'kappa': kappa,
        'itr': itr,
        'confusion_matrix': cm,
        'all_preds': all_preds,
        'all_targets': all_targets
    }

def run_loso_cross_validation(k_shots=[1, 3, 5, 10], epochs_per_fold=60):
    """
    Performs Leave-One-Subject-Out Cross-Validation across all subjects in processed_data/tensors.
    """
    npz_files = sorted(glob.glob(os.path.join(TENSORS_DIR, "*.npz")))
    if len(npz_files) < 2:
        print("[Error] Need at least 2 subject .npz files for LOSO-CV.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print(f"Starting Leave-One-Subject-Out (LOSO) Cross-Validation on {len(npz_files)} Subjects")
    print("=" * 75)

    loso_results = {}

    for target_idx, target_file in enumerate(npz_files):
        target_name = os.path.basename(target_file).replace('.npz', '')
        train_files = [f for i, f in enumerate(npz_files) if i != target_idx]

        print(f"\n[Fold {target_idx + 1}/{len(npz_files)}] Target Unseen Subject: {target_name.upper()}")
        print(f"  --> Base Training Subjects: {[os.path.basename(f) for f in train_files]}")

        train_ds = EEGSubjectDataset(train_files)
        target_ds = EEGSubjectDataset([target_file])

        # Train meta-model on base subjects
        model = train_meta_model(
            train_dataset=train_ds,
            val_dataset=None,
            k_shot=5,
            epochs=epochs_per_fold,
            device=device
        )

        fold_shot_results = {}
        for k in k_shots:
            res = evaluate_fewshot_on_target(model, target_ds, k_shot=k, n_eval_episodes=50, device=device)
            fold_shot_results[k] = res
            print(f"    - Few-Shot (K={k:2d}): Accuracy = {res['accuracy']:.2f}% | Kappa = {res['kappa']:.3f} | ITR = {res['itr']:.2f} bpm")

        loso_results[target_name] = fold_shot_results

    # Plot Average Few-Shot Scaling Curve
    plot_fewshot_curve(loso_results, k_shots)

    # Plot Aggregate Confusion Matrix at K=5
    plot_aggregate_confusion_matrix(loso_results, k_target=5)

    print("\n" + "=" * 75)
    print(f"LOSO-CV Complete! Results and plots saved to: {RESULTS_DIR}")
    print("=" * 75)
    return loso_results

def plot_fewshot_curve(loso_results, k_shots):
    plt.figure(figsize=(8, 5))
    for sub, res in loso_results.items():
        accs = [res[k]['accuracy'] for k in k_shots]
        plt.plot(k_shots, accs, marker='o', label=f'Target: {sub}')

    avg_accs = [np.mean([loso_results[sub][k]['accuracy'] for sub in loso_results]) for k in k_shots]
    plt.plot(k_shots, avg_accs, 'k--', linewidth=2.5, marker='s', label='Mean Across Subjects')

    plt.axhline(25.0, color='r', linestyle=':', label='Chance Level (25%)')
    plt.title('Few-Shot Adaptation Performance across Unseen Subjects', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Support Calibration Trials per Direction (K-Shot)', fontsize=11)
    plt.ylabel('4-Way Direction Decoding Accuracy (%)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "fewshot_scaling_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Few-Shot Curve Plot: {out_path}")

def plot_aggregate_confusion_matrix(loso_results, k_target=5):
    total_cm = np.zeros((4, 4), dtype=np.int32)
    for sub in loso_results:
        if k_target in loso_results[sub]:
            total_cm += loso_results[sub][k_target]['confusion_matrix']

    cm_norm = total_cm.astype('float') / (total_cm.sum(axis=1)[:, np.newaxis] + 1e-6) * 100.0

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=[d.capitalize() for d in DIRECTIONS],
                yticklabels=[d.capitalize() for d in DIRECTIONS])
    plt.title(f'Aggregate Direction Confusion Matrix (K={k_target} Shot)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Direction', fontsize=11)
    plt.ylabel('True Direction', fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, f"confusion_matrix_k{k_target}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix Plot: {out_path}")

if __name__ == "__main__":
    run_loso_cross_validation()
