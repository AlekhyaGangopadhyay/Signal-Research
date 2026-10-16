"""
Classical baselines that every deep-model result must be compared with.

Pipelines (all on stimulus-locked windows, labels from the stimulus timeline):
  bandpower_lr : log band power (6 bands x 18 ch) + logistic regression
  riemann_ts   : spatial covariance -> Riemannian tangent space + logistic regression
  fbcsp_lr     : filter bank (4 bands) x multiclass CSP (log-variance) + logistic regression
  riemann_mdm  : minimum distance to Riemannian class means

Schemes:
  within-subject, leave-one-file-out      (direction, generalising across recordings)
  within-subject, leave-one-format-out    (train on two of arrow/letter/word, test on the third)
  within-subject, shuffled-label control  (block labels shuffled within each recording)
  cross-subject, leave-one-subject-out    (Euclidean-aligned per recording)
  few-shot K-shot nearest class mean      (same calibration splits as the deep model)

Usage: python -m src.baselines [--quick]
"""

import argparse
import json
import os
import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS_DIR, SAMPLING_RATE, SEED, NUM_CLASSES, ensure_dirs
from src.dataset import EEGSubjectDataset, calibration_splits, find_subject_files, shuffled_block_labels
from src.stats import summarise, fmt_result

warnings.filterwarnings('ignore', category=UserWarning)
BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
FB_BANDS = [(4, 8), (8, 13), (13, 20), (20, 30)]


class BandPower(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fr, p = welch(X, fs=SAMPLING_RATE, nperseg=int(SAMPLING_RATE), axis=-1)
        f = np.stack([p[..., (fr >= a) & (fr < b)].mean(-1) for a, b in BANDS], axis=-1)
        return np.log(f + 1e-12).reshape(len(X), -1)


class FilterBankCSP(BaseEstimator, TransformerMixin):
    def __init__(self, nfilter=4):
        self.nfilter = nfilter

    def _bands(self, X):
        for a, b in FB_BANDS:
            sos = butter(4, [a, b], btype='band', fs=SAMPLING_RATE, output='sos')
            yield sosfiltfilt(sos, X, axis=-1)

    def fit(self, X, y):
        from pyriemann.estimation import Covariances
        from pyriemann.spatialfilters import CSP
        self.csps_ = []
        for Xb in self._bands(X):
            C = Covariances('oas').fit_transform(Xb)
            self.csps_.append(CSP(nfilter=self.nfilter, log=True).fit(C, y))
        return self

    def transform(self, X):
        from pyriemann.estimation import Covariances
        return np.hstack([csp.transform(Covariances('oas').fit_transform(Xb))
                          for csp, Xb in zip(self.csps_, self._bands(X))])


def make_pipelines():
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    lr = lambda: LogisticRegression(C=0.1, max_iter=3000)
    return {
        'bandpower_lr': lambda: make_pipeline(BandPower(), StandardScaler(), lr()),
        'riemann_ts': lambda: make_pipeline(Covariances('oas'), TangentSpace(), StandardScaler(), lr()),
        'fbcsp_lr': lambda: make_pipeline(FilterBankCSP(), StandardScaler(), lr()),
        'riemann_mdm': lambda: make_pipeline(Covariances('oas'), MDM()),
    }


def cv_predict(make, X, y, groups):
    return cross_val_predict(make(), X, y, groups=groups, cv=LeaveOneGroupOut())


def fewshot_ncm(ds, k, rng):
    """K-shot nearest class mean on standardised band-power features; same splits as the deep model."""
    F = BandPower().transform(ds.X)
    Z = (F - F.mean(0)) / (F.std(0) + 1e-6)   # unsupervised scaling on the target's own data (no labels)
    preds, labels, groups = [], [], []
    for s, q, _ in calibration_splits(ds, k, rng):
        protos = np.stack([Z[s][ds.y[s] == c].mean(0) for c in range(NUM_CLASSES)])
        d = ((Z[q][:, None, :] - protos[None]) ** 2).sum(-1)
        preds.append(d.argmin(1)); labels.append(ds.y[q]); groups.append(ds.group[q] + 10_000 * len(preds))
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help="Only band-power and tangent-space pipelines")
    args = ap.parse_args()
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    files = find_subject_files()
    pipes = make_pipelines()
    if args.quick:
        pipes = {k: pipes[k] for k in ['bandpower_lr', 'riemann_ts']}
    results = {}
    lines = ["# Baseline results", "",
             "Accuracy [95% block-bootstrap CI], block-permutation p-value. Chance = 25%.", ""]

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log("## Within-subject")
    for f in files:
        ds = EEGSubjectDataset([f])
        subj = ds.subjects[0]
        y_shuf = shuffled_block_labels(ds, rng)
        log(f"\n### Subject {subj} ({len(ds)} windows, {len(ds.recordings)} recordings)")
        log("| Pipeline | Leave-one-file-out | Leave-one-format-out | Shuffled-label control (LOFO) |")
        log("|---|---|---|---|")
        for name, make in pipes.items():
            r_file = summarise(cv_predict(make, ds.X, ds.y, ds.rec), ds.y, ds.group)
            r_fmt = summarise(cv_predict(make, ds.X, ds.y, ds.fmt), ds.y, ds.group)
            r_shuf = summarise(cv_predict(make, ds.X, y_shuf, ds.rec), y_shuf, ds.group)
            results[f"{subj}/{name}"] = {'lofo': r_file, 'format_out': r_fmt, 'shuffled': r_shuf}
            log(f"| {name} | {fmt_result(r_file)} | {fmt_result(r_fmt)} | {fmt_result(r_shuf)} |")

    log("\n## Cross-subject (leave-one-subject-out, Euclidean alignment per recording)")
    all_ds = EEGSubjectDataset(files, alignment='euclidean')
    log("| Pipeline | " + " | ".join(f"Target {s}" for s in all_ds.subjects) + " |")
    log("|---|" + "---|" * len(all_ds.subjects))
    for name, make in pipes.items():
        cells = []
        for s_id, s in enumerate(all_ds.subjects):
            tr, te = all_ds.subject != s_id, all_ds.subject == s_id
            model = make().fit(all_ds.X[tr], all_ds.y[tr])
            r = summarise(model.predict(all_ds.X[te]), all_ds.y[te], all_ds.group[te])
            results[f"loso/{s}/{name}"] = r
            cells.append(fmt_result(r))
        log(f"| {name} | " + " | ".join(cells) + " |")

    log("\n## Few-shot nearest class mean on band power (calibrate on one recording, test on the others)")
    log("| Subject | K=1 | K=3 | K=5 | K=10 |")
    log("|---|---|---|---|---|")
    for f in files:
        ds = EEGSubjectDataset([f])
        cells = []
        for k in (1, 3, 5, 10):
            r = summarise(*fewshot_ncm(ds, k, rng))
            results[f"fewshot_ncm/{ds.subjects[0]}/K{k}"] = r
            cells.append(fmt_result(r))
        log(f"| {ds.subjects[0]} | " + " | ".join(cells) + " |")

    with open(os.path.join(RESULTS_DIR, 'baselines.md'), 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(RESULTS_DIR, 'baselines.json'), 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {os.path.join(RESULTS_DIR, 'baselines.md')}")


if __name__ == '__main__':
    main()
