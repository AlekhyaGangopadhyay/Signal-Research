"""
ETL Pipeline: Parses Raw Bipolar EEG text dumps, applies signal conditioning,
segments into 2.0s directional epochs, and exports both structured CSVs and PyTorch NPZ archives.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    RAW_DATA_DIR, CSV_TRIALS_DIR, TENSORS_DIR, METADATA_PATH,
    CHANNELS, NUM_CHANNELS, SAMPLING_RATE, TRIAL_SAMPLES,
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ, DIRECTIONS
)
from src.preprocessing import preprocess_continuous_eeg, robust_standardize

def parse_raw_txt_file(file_path):
    """
    Parses bipolar channel arrays from raw multi-page text dump.
    Returns: numpy array of shape (18, Total_Samples) or None if invalid.
    """
    channels_data = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split lines and extract continuous comma-separated numbers
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        for part in parts:
            part = part.strip()
            if ',' in part and not part.startswith('Page'):
                # Extract comma-separated numeric sequences
                try:
                    num_strings = [s.strip() for s in part.split(',') if s.strip()]
                    if len(num_strings) >= 100:
                        # Valid channel time series
                        numeric_vals = [float(v) for v in num_strings if re.match(r'^-?\d+(\.\d+)?$', v)]
                        if len(numeric_vals) >= 100:
                            channels_data.append(numeric_vals)
                except Exception:
                    continue

    if len(channels_data) < NUM_CHANNELS:
        return None

    # Align to common sample length
    selected_channels = channels_data[:NUM_CHANNELS]
    min_length = min(len(c) for c in selected_channels)
    
    matrix = np.array([c[:min_length] for c in selected_channels], dtype=np.float32)
    return matrix

def run_conversion():
    """
    Scans all subjects in 'Raw data/', cleans signals, extracts epochs,
    and saves individual CSVs + consolidated NPZ archives.
    """
    print("=" * 70)
    print("Starting Raw EEG Data Conversion & Preprocessing Pipeline")
    print("=" * 70)

    metadata_records = []
    subject_folders = sorted([
        d for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ])

    if not subject_folders:
        print(f"[Error] No subject directories found in {RAW_DATA_DIR}")
        return

    print(f"Found {len(subject_folders)} subjects: {subject_folders}\n")

    for sub_idx, sub_name in enumerate(subject_folders):
        sub_path = os.path.join(RAW_DATA_DIR, sub_name)
        txt_files = sorted(glob.glob(os.path.join(sub_path, "*.txt")))
        
        print(f"[{sub_idx + 1}/{len(subject_folders)}] Processing {sub_name} ({len(txt_files)} files)...")

        subject_epochs = []
        subject_labels = []

        for f_idx, file_path in enumerate(tqdm(txt_files, desc=f"  Files for {sub_name}")):
            raw_matrix = parse_raw_txt_file(file_path)
            if raw_matrix is None:
                continue

            # 1. Filter continuous recording (Bandpass + Notch)
            filtered_matrix = preprocess_continuous_eeg(
                raw_matrix,
                fs=SAMPLING_RATE,
                lowcut=BANDPASS_LOW,
                highcut=BANDPASS_HIGH,
                notch_f=NOTCH_FREQ
            )

            # 2. Determine class label (cyclical allocation 0:left, 1:right, 2:forward, 3:backward)
            dir_id = f_idx % len(DIRECTIONS)
            dir_label = DIRECTIONS[dir_id]

            # 3. Epoch into fixed trial windows (e.g., 2.0s = 500 samples)
            total_samples = filtered_matrix.shape[1]
            num_epochs = total_samples // TRIAL_SAMPLES

            for ep_i in range(num_epochs):
                start_idx = ep_i * TRIAL_SAMPLES
                end_idx = start_idx + TRIAL_SAMPLES
                epoch_raw = filtered_matrix[:, start_idx:end_idx]

                # Robust channel-wise standardization
                epoch_norm = robust_standardize(epoch_raw)

                # 4. Save individual human-readable CSV trial
                trial_filename = f"{sub_name.replace(' ', '_')}_f{f_idx+1:02d}_ep{ep_i+1:03d}_{dir_label}.csv"
                csv_path = os.path.join(CSV_TRIALS_DIR, trial_filename)

                df = pd.DataFrame(epoch_norm.T, columns=CHANNELS)
                df.insert(0, 'time_ms', np.arange(TRIAL_SAMPLES) * (1000.0 / SAMPLING_RATE))
                df.to_csv(csv_path, index=False)

                subject_epochs.append(epoch_norm)
                subject_labels.append(dir_id)

                metadata_records.append({
                    'trial_id': len(metadata_records) + 1,
                    'trial_filename': trial_filename,
                    'subject': sub_name,
                    'source_file': os.path.basename(file_path),
                    'epoch_index': ep_i,
                    'direction': dir_label,
                    'label_id': dir_id,
                    'channels': NUM_CHANNELS,
                    'samples': TRIAL_SAMPLES,
                    'duration_sec': TRIAL_SAMPLES / SAMPLING_RATE
                })

        # Save Subject Consolidated Tensor Archive (.npz)
        if subject_epochs:
            clean_sub_name = re.sub(r'[^a-zA-Z0-9_]', '_', sub_name).lower()
            npz_path = os.path.join(TENSORS_DIR, f"{clean_sub_name}.npz")
            X_tensor = np.stack(subject_epochs).astype(np.float32) # [N, 18, 500]
            y_tensor = np.array(subject_labels, dtype=np.int64)    # [N]
            
            np.savez_compressed(npz_path, X=X_tensor, y=y_tensor)
            print(f"  --> Saved {len(subject_epochs)} epoched trials to {npz_path}\n")

    # Save Master Metadata Manifest
    if metadata_records:
        meta_df = pd.DataFrame(metadata_records)
        meta_df.to_csv(METADATA_PATH, index=False)
        print("=" * 70)
        print(f"SUCCESS: Generated {len(metadata_records)} total trials across all subjects.")
        print(f"Master metadata saved to: {METADATA_PATH}")
        print("=" * 70)

if __name__ == "__main__":
    run_conversion()
