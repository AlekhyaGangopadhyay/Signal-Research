"""
Evaluation and Leave-One-Subject-Out Cross-Validation (LOSO-CV) Engine.

For each target subject:
  1. Meta-train on the other subjects (early stopping on their held-out recordings).
  2. Optionally re-estimate BatchNorm statistics on the target's unlabelled data (AdaBN).
  3. For each K: every target recording takes a turn as the calibration session (K windows per class),
     and all windows of the target's other recordings are classified by nearest prototype.
  4. Report accuracy with a block-bootstrap 95% CI and a block-permutation p-value, plus the same
     pipeline trained on shuffled labels (control).
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.config import (
    RESULTS_DIR, CHECKPOINTS_DIR, DIRECTIONS, NUM_CLASSES, TRIAL_SAMPLES, SAMPLING_RATE, NUM_EPOCHS, SEED,
    ensure_dirs
)
from src.dataset import EEGSubjectDataset, calibration_splits, find_subject_files, shuffled_block_labels
from src.models import adapt_batchnorm
from src.stats import summarise, fmt_result
from src.train_fewshot import train_meta_model


def calculate_itr(n_classes, accuracy_ratio, trial_duration_sec):
    """
    Information Transfer Rate in bits/min (Wolpaw et al.). Only meaningful for accuracies that are
    significantly above chance; 0 at or below chance.
    """
    p = np.clip(accuracy_ratio, 1e-4, 0.9999)
    n = n_classes
    if p <= (1.0 / n):
        return 0.0
    b = np.log2(n) + p * np.log2(p) + (1.0 - p) * np.log2((1.0 - p) / (n - 1.0))
    return max(0.0, b * 60.0 / trial_duration_sec)


@torch.no_grad()
def embed(model, X, device="cpu", batch_size=256):
    model.eval()
    return torch.cat([model(torch.as_tensor(X[i:i + batch_size], device=device)).cpu()
                      for i in range(0, len(X), batch_size)])


def evaluate_fewshot_on_target(model, target_ds, k_shot, rng, device="cpu"):
    """Nearest-prototype classification over all calibration splits of the target subject."""
    Z = embed(model, target_ds.X, device)
    preds, labels, groups = [], [], []
    for split_i, (s, q, _) in enumerate(calibration_splits(target_ds, k_shot, rng)):
        s_t = torch.as_tensor(s)
        protos = torch.stack([Z[s_t][torch.as_tensor(target_ds.y[s] == c)].mean(0) for c in range(NUM_CLASSES)])
        d = torch.cdist(Z[torch.as_tensor(q)], protos) ** 2
        preds.append(d.argmin(1).numpy())
        labels.append(target_ds.y[q])
        groups.append(target_ds.group[q] + 100_000 * split_i)   # each split is its own set of blocks
    if not preds:
        return None
    preds, labels, groups = np.concatenate(preds), np.concatenate(labels), np.concatenate(groups)
    res = summarise(preds, labels, groups)
    res['kappa'] = float(cohen_kappa_score(labels, preds))
    res['itr_bpm'] = calculate_itr(NUM_CLASSES, res['accuracy'], TRIAL_SAMPLES / SAMPLING_RATE) if res['p_value'] < 0.05 else 0.0
    res['confusion_matrix'] = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist()
    return res


def run_loso_cross_validation(k_shots=(1, 3, 5, 10), epochs_per_fold=NUM_EPOCHS, model_type="spatial_temporal",
                              alignment="euclidean", adabn=True, use_angular=False, control=True,
                              save_checkpoints=True, tag=None):
    files = find_subject_files()
    if len(files) < 2:
        raise SystemExit("[Error] Need at least 2 subject_*.npz files for LOSO-CV. Run convert_raw_data first.")
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = tag or f"{model_type}_{alignment}{'_adabn' if adabn else ''}{'_angular' if use_angular else ''}"
    print("=" * 75, flush=True)
    print(f"LOSO-CV on {len(files)} subjects | model {model_type} | alignment {alignment} | AdaBN {adabn} | "
          f"angular {use_angular} | K {list(k_shots)}", flush=True)
    print("=" * 75, flush=True)

    results = {'config': {'k_shots': list(k_shots), 'epochs': epochs_per_fold, 'model_type': model_type,
                          'alignment': alignment, 'adabn': adabn, 'use_angular': use_angular, 'seed': SEED},
               'folds': {}}
    rng = np.random.default_rng(SEED)
    for t_idx, target_file in enumerate(files):
        train_files = [f for i, f in enumerate(files) if i != t_idx]
        train_ds = EEGSubjectDataset(train_files, alignment=alignment)
        target_ds = EEGSubjectDataset([target_file], alignment=alignment)
        name = target_ds.subjects[0]
        print(f"\n[Fold {t_idx + 1}/{len(files)}] target subject {name} | training subjects {train_ds.subjects}", flush=True)

        runs = {'real': train_ds}
        if control:
            shuffled = train_ds.subset(np.arange(len(train_ds)))
            shuffled.y = shuffled_block_labels(shuffled, rng)
            runs['shuffled_control'] = shuffled
        fold = {}
        for run_name, ds in runs.items():
            ckpt = os.path.join(CHECKPOINTS_DIR, f"loso_{tag}_target-{name}.pth") if (save_checkpoints and run_name == 'real') else None
            model, history = train_meta_model(ds, model_type=model_type, epochs=epochs_per_fold, use_angular=use_angular,
                                              device=device, save_path=ckpt, verbose=(run_name == 'real'))
            if adabn:
                adapt_batchnorm(model, target_ds.X, device=device)
            fold[run_name] = {'best_epoch': int(np.nanargmin([h['val_loss'] for h in history]) + 1), 'k': {}}
            for k in k_shots:
                res = evaluate_fewshot_on_target(model, target_ds, k, rng, device)
                if res is None:
                    continue
                fold[run_name]['k'][k] = res
                print(f"    {run_name:16s} K={k:2d}: {fmt_result(res)} | kappa {res['kappa']:.3f}", flush=True)
        results['folds'][name] = fold

    out_json = os.path.join(RESULTS_DIR, f"loso_{tag}.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    write_markdown(results, os.path.join(RESULTS_DIR, f"loso_{tag}.md"))
    plot_fewshot_curve(results, os.path.join(RESULTS_DIR, f"fewshot_scaling_curve_{tag}.png"))
    plot_aggregate_confusion_matrix(results, 5, os.path.join(RESULTS_DIR, f"confusion_matrix_k5_{tag}.png"))
    print(f"\nLOSO-CV complete. Results: {out_json}", flush=True)
    return results


def write_markdown(results, path):
    cfg = results['config']
    lines = [f"# LOSO few-shot results ({cfg['model_type']}, alignment={cfg['alignment']}, AdaBN={cfg['adabn']}, "
             f"angular={cfg['use_angular']})", "",
             "Calibrate on one recording of the target subject, test on all its other recordings. "
             "Accuracy [95% block-bootstrap CI], block-permutation p. Chance = 25%.", "",
             "| Target | Run | " + " | ".join(f"K={k}" for k in cfg['k_shots']) + " |",
             "|---|---|" + "---|" * len(cfg['k_shots'])]
    for subj, fold in results['folds'].items():
        for run_name, run in fold.items():
            cells = [fmt_result(run['k'][k]) if k in run['k'] else 'n/a' for k in cfg['k_shots']]
            lines.append(f"| {subj} | {run_name} | " + " | ".join(cells) + " |")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def plot_fewshot_curve(results, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ks = results['config']['k_shots']
    fig, ax = plt.subplots(figsize=(8, 5))
    for subj, fold in results['folds'].items():
        for run_name, style in [('real', '-'), ('shuffled_control', ':')]:
            if run_name not in fold:
                continue
            run = fold[run_name]['k']
            kk = [k for k in ks if k in run]
            acc = np.array([run[k]['accuracy'] for k in kk]) * 100
            lo = np.array([run[k]['ci_low'] for k in kk]) * 100
            hi = np.array([run[k]['ci_high'] for k in kk]) * 100
            line, = ax.plot(kk, acc, style, marker='o', label=f"{subj} ({'shuffled control' if run_name != 'real' else 'real labels'})")
            if run_name == 'real':
                ax.fill_between(kk, lo, hi, alpha=0.15, color=line.get_color())
    ax.axhline(25.0, color='k', linestyle='--', label='Chance (25%)')
    ax.set_title('Few-shot accuracy on unseen subjects (95% CI)')
    ax.set_xlabel('Calibration windows per direction (K)')
    ax.set_ylabel('4-way accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def plot_aggregate_confusion_matrix(results, k_target, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    total = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for fold in results['folds'].values():
        if k_target in fold['real']['k']:
            total += np.array(fold['real']['k'][k_target]['confusion_matrix'])
    if total.sum() == 0:
        return
    cm = total / total.sum(axis=1, keepdims=True) * 100
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', vmin=0, vmax=100,
                xticklabels=[d.capitalize() for d in DIRECTIONS], yticklabels=[d.capitalize() for d in DIRECTIONS])
    plt.title(f'Aggregate confusion matrix, K={k_target} (row %)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LOSO few-shot evaluation")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--model", default="spatial_temporal", choices=["spatial_temporal", "eegnet"])
    ap.add_argument("--alignment", default="euclidean", choices=["none", "euclidean"])
    ap.add_argument("--no-adabn", action="store_true")
    ap.add_argument("--angular", action="store_true")
    ap.add_argument("--no-control", action="store_true")
    a = ap.parse_args()
    run_loso_cross_validation(a.shots, a.epochs, a.model, a.alignment, not a.no_adabn, a.angular, not a.no_control)
