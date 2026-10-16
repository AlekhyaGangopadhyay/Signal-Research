import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CHANNELS, SAMPLING_RATE, TRIAL_SAMPLES  # noqa: E402
from src.stimulus import FORMATS, stimulus_windows  # noqa: E402

RAW_LABELS = ['FP2- F8', 'F8 - T4', 'T4 - T6', 'T6 - O2', 'FP2- F4', 'F4 - C4', 'C4 - P4', 'P4 - O2',
              'FP1- F7', 'F7 - T3', 'T3 - T5', 'T5 - O1', 'FP1- F3', 'F3 - C3', 'C3 - P3', 'P3 - O1',
              'FZ - CZ', 'CZ - PZ', 'BP1- REF', 'BP2- REF']


def write_raw_file(path, n_pages=3, page_len=2560, seed=0, labels=RAW_LABELS, corrupt=None):
    """Writes a file in the raw export layout. Returns the [20, n_pages*page_len] int data written."""
    rng = np.random.default_rng(seed)
    data = rng.integers(-30, 30, size=(len(labels), n_pages * page_len))
    tokens = []
    for p in range(n_pages):
        tokens += [f"Page-{p + 1}", "", "", ""]
        for c, lab in enumerate(labels):
            vals = ",".join(str(v) for v in data[c, p * page_len:(p + 1) * page_len]) + ","
            if corrupt == 'bad_number' and p == 1 and c == 2:
                vals = vals.replace(",", ",x", 1)
            tokens += [lab, vals]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\t".join(tokens))
    return data


@pytest.fixture
def raw_file(tmp_path):
    path = str(tmp_path / "Raw data(Subject T)" / "AllPages_1.txt")
    data = write_raw_file(path)
    return path, data


def make_subject_npz(path, subject='T', n_files=3, duration=180.0, seed=0, signal=0.0):
    """Synthetic NPZ in the converter's format. `signal` adds a class-dependent offset to channel 0."""
    rng = np.random.default_rng(seed)
    X, y, is_stim, block, fmt, file_idx, start = [], [], [], [], [], [], []
    for f in range(n_files):
        for w in stimulus_windows(duration):
            x = rng.standard_normal((len(CHANNELS), TRIAL_SAMPLES)).astype(np.float32)
            if w.kind == 'stim':
                x[0] += signal * w.label
            X.append(x); y.append(w.label); is_stim.append(w.kind == 'stim'); block.append(w.block)
            fmt.append(FORMATS.index(w.format)); file_idx.append(f); start.append(w.start_s)
    np.savez_compressed(
        path, X=np.stack(X), y=np.array(y, np.int64), is_stim=np.array(is_stim), block=np.array(block, np.int16),
        fmt=np.array(fmt, np.int8), file_idx=np.array(file_idx, np.int16), start_s=np.array(start, np.float32),
        bad=np.zeros(len(y), bool), files=np.array([f"AllPages_{i}.txt" for i in range(n_files)]),
        file_offsets_s=np.zeros(n_files, np.float32), file_durations_s=np.full(n_files, duration, np.float32),
        subject=np.array(subject), fs=np.array(SAMPLING_RATE), channels=np.array(CHANNELS))
    return path


@pytest.fixture
def subject_npz(tmp_path):
    return make_subject_npz(str(tmp_path / "subject_t.npz"))
