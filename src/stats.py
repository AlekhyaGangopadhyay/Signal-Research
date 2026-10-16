"""
Statistics for decoding results.

Windows from the same stimulus block are not independent (they overlap in time and share a label),
so both the confidence interval and the permutation test work at the level of blocks.
"""

import numpy as np

from src.config import N_PERMUTATIONS, SEED


def cluster_bootstrap_ci(correct, groups, n_boot=2000, alpha=0.05, rng=None):
    """95% CI of accuracy, resampling whole groups (blocks) with replacement."""
    rng = rng or np.random.default_rng(SEED)
    correct = np.asarray(correct, float)
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=correct)
    counts = np.bincount(inv)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        boots[i] = sums[pick].sum() / counts[pick].sum()
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def block_permutation_test(preds, labels, groups, n_perm=N_PERMUTATIONS, rng=None):
    """
    p-value for accuracy above chance. Each group (block) has a single true label; the null shuffles
    which label belongs to which group, keeping the predictions and the block structure fixed.
    Returns (observed_accuracy, p_value, null_mean).
    """
    rng = rng or np.random.default_rng(SEED)
    preds, labels, groups = map(np.asarray, (preds, labels, groups))
    ug, inv = np.unique(groups, return_inverse=True)
    g_label = np.zeros(len(ug), dtype=labels.dtype)
    g_label[inv] = labels
    obs = float((preds == labels).mean())
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = (preds == rng.permutation(g_label)[inv]).mean()
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return obs, float(p), float(null.mean())


def summarise(preds, labels, groups, n_perm=N_PERMUTATIONS):
    """Accuracy, block-bootstrap 95% CI, block-permutation p-value, and the permutation null mean."""
    preds, labels = np.asarray(preds), np.asarray(labels)
    acc, p, null_mean = block_permutation_test(preds, labels, groups, n_perm)
    lo, hi = cluster_bootstrap_ci(preds == labels, groups)
    return {'accuracy': acc, 'ci_low': lo, 'ci_high': hi, 'p_value': p, 'null_mean': null_mean, 'n': int(len(labels))}


def fmt_result(r):
    return (f"{r['accuracy'] * 100:5.1f}% [{r['ci_low'] * 100:4.1f}, {r['ci_high'] * 100:4.1f}] "
            f"p={r['p_value']:.3f} (n={r['n']})")
