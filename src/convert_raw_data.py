"""
ETL Pipeline: parses the raw multi-page EEG exports, filters each whole recording, cuts windows
locked to the stimulus blocks of the video, and saves one NPZ per subject with full metadata.

Labels come from the stimulus timeline (src/stimulus.py) and each file's offset
(stimulus_offsets.json), never from file order.
"""

import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    RAW_DATA_DIR, CSV_TRIALS_DIR, TENSORS_DIR, METADATA_PATH, OFFSETS_PATH,
    CHANNELS, NUM_CHANNELS, SAMPLING_RATE, TRIAL_SAMPLES, DIRECTIONS, BLANK_LABEL, ensure_dirs
)
from src.preprocessing import preprocess_continuous_eeg, robust_scale_recording, flag_artifacts
from src.raw_io import parse_raw_txt_file, subject_from_folder
from src.stimulus import FORMATS, load_offsets, offset_for, stimulus_windows


def subject_npz_name(subject):
    clean = ''.join(ch if ch.isalnum() else '_' for ch in subject).lower()
    return f"subject_{clean}.npz"


def epochs_from_recording(recording, offset_s, fs=SAMPLING_RATE):
    """Filters and scales one RawRecording and returns its windows plus per-window metadata."""
    filtered = preprocess_continuous_eeg(recording.data, fs=fs)
    scaled = robust_scale_recording(filtered)
    windows = stimulus_windows(recording.duration_s, offset_s=offset_s)
    X, meta = [], []
    for w in windows:
        a = int(round(w.start_s * fs))
        b = a + TRIAL_SAMPLES
        if b > scaled.shape[1]:
            continue
        X.append(scaled[:, a:b])
        meta.append(w)
    if not X:
        return np.empty((0, NUM_CHANNELS, TRIAL_SAMPLES), np.float32), []
    return np.stack(X).astype(np.float32), meta


def convert_subject(sub_folder, offsets, default_offset, write_csv=False):
    subject = subject_from_folder(os.path.basename(sub_folder))
    txt_files = sorted(glob.glob(os.path.join(sub_folder, "*.txt")))
    X_all, rows = [], []
    files, file_offsets, durations = [], [], []

    for f_idx, path in enumerate(tqdm(txt_files, desc=f"  {subject}", leave=False)):
        rec = parse_raw_txt_file(path)
        off = offset_for(subject, rec.file, offsets, default_offset)
        X, meta = epochs_from_recording(rec, off)
        bad, reasons = flag_artifacts(X) if len(X) else (np.zeros(0, bool), [])
        files.append(rec.file)
        file_offsets.append(off)
        durations.append(rec.duration_s)
        X_all.append(X)
        for i, w in enumerate(meta):
            rows.append({
                'subject': subject,
                'file': rec.file,
                'file_idx': f_idx,
                'offset_s': off,
                'start_s': w.start_s,
                'kind': w.kind,
                'block': w.block,
                'format': w.format,
                'label_id': w.label,
                'direction': DIRECTIONS[w.label] if w.label != BLANK_LABEL else 'blank',
                'bad': bool(bad[i]),
                'bad_reason': reasons[i],
            })

    X = np.concatenate(X_all, axis=0) if X_all else np.empty((0, NUM_CHANNELS, TRIAL_SAMPLES), np.float32)
    df = pd.DataFrame(rows)
    npz_path = os.path.join(TENSORS_DIR, subject_npz_name(subject))
    np.savez_compressed(
        npz_path,
        X=X,
        y=df['label_id'].to_numpy(np.int64),
        is_stim=(df['kind'] == 'stim').to_numpy(),
        block=df['block'].to_numpy(np.int16),
        fmt=df['format'].map({f: i for i, f in enumerate(FORMATS)}).to_numpy(np.int8),
        file_idx=df['file_idx'].to_numpy(np.int16),
        start_s=df['start_s'].to_numpy(np.float32),
        bad=df['bad'].to_numpy(),
        files=np.array(files),
        file_offsets_s=np.array(file_offsets, np.float32),
        file_durations_s=np.array(durations, np.float32),
        subject=np.array(subject),
        fs=np.array(SAMPLING_RATE),
        channels=np.array(CHANNELS),
    )

    if write_csv:
        for i, r in df.iterrows():
            name = f"{subject}_{os.path.splitext(r['file'])[0]}_b{r['block']:02d}_{r['start_s']:07.2f}s_{r['direction']}.csv"
            out = pd.DataFrame(X[i].T, columns=CHANNELS)
            out.insert(0, 'time_ms', np.arange(TRIAL_SAMPLES) * (1000.0 / SAMPLING_RATE))
            out.to_csv(os.path.join(CSV_TRIALS_DIR, name), index=False)
    return df, npz_path


def run_conversion(write_csv=False, offsets_path=OFFSETS_PATH):
    print("=" * 70, flush=True)
    print("Raw EEG conversion: all pages, 256 Hz, stimulus-locked windows", flush=True)
    print("=" * 70, flush=True)
    ensure_dirs()
    # Remove stale outputs so old labels can never mix with new ones
    for stale in [TENSORS_DIR, CSV_TRIALS_DIR]:
        if os.path.isdir(stale):
            shutil.rmtree(stale)
    os.makedirs(TENSORS_DIR, exist_ok=True)
    if write_csv:
        os.makedirs(CSV_TRIALS_DIR, exist_ok=True)

    offsets, default_offset, provenance = load_offsets(offsets_path)
    print(f"Stimulus offsets: {provenance}", flush=True)

    sub_folders = sorted(d for d in glob.glob(os.path.join(RAW_DATA_DIR, '*')) if os.path.isdir(d))
    if not sub_folders:
        raise FileNotFoundError(f"No subject directories found in {RAW_DATA_DIR}")

    frames = []
    for sub_folder in sub_folders:
        df, npz_path = convert_subject(sub_folder, offsets, default_offset, write_csv)
        stim = df[df['kind'] == 'stim']
        counts = stim['direction'].value_counts().reindex(DIRECTIONS, fill_value=0).tolist()
        print(f"  {df['subject'].iloc[0]:6s}: {df['file'].nunique():2d} files | {len(stim):4d} stim + "
              f"{(df['kind'] == 'blank').sum():3d} blank windows | per class L/R/F/B {counts} | "
              f"flagged bad {df['bad'].mean() * 100:4.1f}% -> {os.path.basename(npz_path)}", flush=True)
        frames.append(df)

    meta = pd.concat(frames, ignore_index=True)
    meta.insert(0, 'epoch_id', np.arange(len(meta)))
    meta.to_csv(METADATA_PATH, index=False)
    print(f"Saved {len(meta)} windows; manifest: {METADATA_PATH}", flush=True)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw EEG exports to stimulus-locked epochs")
    parser.add_argument("--csv", action="store_true", help="Also write one CSV per window (slow, large)")
    parser.add_argument("--offsets", default=OFFSETS_PATH, help="Path to stimulus_offsets.json")
    args = parser.parse_args()
    run_conversion(write_csv=args.csv, offsets_path=args.offsets)
