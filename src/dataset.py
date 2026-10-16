"""
Dataset and Episodic Few-Shot Sampler for PyTorch Training.

Rules that keep evaluation honest:
- Never sample with replacement.
- Support and query windows of an episode always come from different stimulus blocks
  (windows of the same block overlap in time and share a recording fingerprint).
- For a target subject, calibration (support) windows come from one recording and the
  test (query) windows from that subject's other recordings.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import TENSORS_DIR, NUM_CLASSES
from src.preprocessing import euclidean_alignment


class EEGSubjectDataset(Dataset):
    """
    Loads per-subject NPZ files written by convert_raw_data.py.
    Keeps stimulus windows (not blank) that are not flagged as artifacts.

    alignment: 'none' or 'euclidean' (per recording, unsupervised; see preprocessing.euclidean_alignment).
    """

    def __init__(self, npz_paths, alignment='none', include_bad=False):
        X, y, block, fmt, rec, subj = [], [], [], [], [], []
        self.recordings = []        # list of (subject, file)
        self.subjects = []
        for path in npz_paths:
            d = np.load(path, allow_pickle=False)
            keep = d['is_stim'] & (include_bad | ~d['bad'])
            s_name = str(d['subject'])
            s_id = len(self.subjects)
            self.subjects.append(s_name)
            files = [str(f) for f in d['files']]
            Xs = d['X'][keep]
            fidx = d['file_idx'][keep]
            if alignment == 'euclidean':
                Xs = Xs.copy()
                for fi in np.unique(fidx):
                    Xs[fidx == fi] = euclidean_alignment(Xs[fidx == fi])
            elif alignment != 'none':
                raise ValueError(f"unknown alignment {alignment!r}")
            rec_ids = np.empty(len(fidx), dtype=np.int64)
            for fi in np.unique(fidx):
                rec_ids[fidx == fi] = len(self.recordings)
                self.recordings.append((s_name, files[fi]))
            X.append(Xs); y.append(d['y'][keep]); block.append(d['block'][keep])
            fmt.append(d['fmt'][keep]); rec.append(rec_ids); subj.append(np.full(keep.sum(), s_id))

        if X:
            self.X = np.concatenate(X).astype(np.float32)
            self.y = np.concatenate(y).astype(np.int64)
            self.block = np.concatenate(block).astype(np.int64)
            self.fmt = np.concatenate(fmt).astype(np.int64)
            self.rec = np.concatenate(rec)
            self.subject = np.concatenate(subj)
        else:
            self.X = np.empty((0, 18, 512), np.float32)
            self.y = self.block = self.fmt = self.rec = self.subject = np.empty(0, np.int64)
        # A stimulus block within one recording is the unit of dependence
        self.group = self.rec * 100 + self.block

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

    def subset(self, mask):
        """Returns a shallow copy restricted to `mask` (boolean or index array)."""
        new = object.__new__(EEGSubjectDataset)
        new.recordings, new.subjects = self.recordings, self.subjects
        for name in ['X', 'y', 'block', 'fmt', 'rec', 'subject', 'group']:
            setattr(new, name, getattr(self, name)[mask])
        return new

    # ------------------------------------------------------------------ training episodes
    def sample_fewshot_episode(self, n_way=NUM_CLASSES, k_shot=5, q_query=5, rng=None):
        """
        Samples an N-way episode without replacement. Support and query come from disjoint
        stimulus blocks. Raises ValueError if a class lacks enough data.
        Returns (support_x, support_y, query_x, query_y) with support ordered class by class.
        """
        rng = rng or np.random.default_rng()
        sx, sy, qx, qy = [], [], [], []
        for c in range(n_way):
            groups = np.unique(self.group[self.y == c])
            if len(groups) < 2:
                raise ValueError(f"class {c} needs windows from at least 2 blocks")
            rng.shuffle(groups)
            s_idx, q_idx = [], []
            for g in groups:
                idx = np.flatnonzero((self.group == g) & (self.y == c))
                rng.shuffle(idx)
                if len(s_idx) < k_shot:
                    s_idx.extend(idx[:k_shot - len(s_idx)])
                elif len(q_idx) < q_query:
                    q_idx.extend(idx[:q_query - len(q_idx)])
                else:
                    break
            if len(s_idx) < k_shot or len(q_idx) < q_query:
                raise ValueError(f"class {c}: not enough blocks for k={k_shot}, q={q_query}")
            sx.append(self.X[s_idx]); sy.append(np.full(k_shot, c))
            qx.append(self.X[q_idx]); qy.append(np.full(q_query, c))
        to_t = lambda a, dt: torch.tensor(np.concatenate(a), dtype=dt)
        return to_t(sx, torch.float32), to_t(sy, torch.long), to_t(qx, torch.float32), to_t(qy, torch.long)


def calibration_splits(target, k_shot, rng=None, n_way=NUM_CLASSES):
    """
    Few-shot evaluation splits for one target subject. Each recording of the target takes a turn as
    the calibration session: K windows per class are drawn from it (without replacement, spread over
    its blocks), and every stimulus window of the target's *other* recordings is the test set.
    Yields (support_idx, query_idx, calibration_recording_id).
    """
    rng = rng or np.random.default_rng()
    for r in np.unique(target.rec):
        in_rec = target.rec == r
        support = []
        ok = True
        for c in range(n_way):
            idx = np.flatnonzero(in_rec & (target.y == c))
            if len(idx) < k_shot:
                ok = False
                break
            # round-robin over blocks so K windows cover as many blocks as possible
            by_block = [list(rng.permutation(idx[target.block[idx] == b]))
                        for b in rng.permutation(np.unique(target.block[idx]))]
            picked = []
            while len(picked) < k_shot:
                for windows in by_block:
                    if windows and len(picked) < k_shot:
                        picked.append(windows.pop())
            support.append(np.array(picked))
        if not ok:
            continue
        query = np.flatnonzero(~in_rec)
        yield np.concatenate(support), query, int(r)


def shuffled_block_labels(ds, rng):
    """Shuffle the label of each block among the blocks of the same recording."""
    y = ds.y.copy()
    for r in np.unique(ds.rec):
        in_rec = np.flatnonzero(ds.rec == r)
        blocks = np.unique(ds.block[in_rec])
        labels = np.array([ds.y[in_rec][ds.block[in_rec] == b][0] for b in blocks])
        for b, lab in zip(blocks, rng.permutation(labels)):
            y[in_rec[ds.block[in_rec] == b]] = lab
    return y


def find_subject_files(tensors_dir=TENSORS_DIR):
    return sorted(glob.glob(os.path.join(tensors_dir, "subject_*.npz")))
