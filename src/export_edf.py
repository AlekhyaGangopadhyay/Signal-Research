"""
Converts the raw NeuroMax text exports (AllPages_*.txt) to EDF so they can be opened in standard EEG
viewers (EDFbrowser, MNE, EEGLAB).

- All pages are joined in time; 18 bipolar channels at 256 Hz (BP1/BP2 auxiliary inputs dropped,
  or kept with --keep-aux).
- The stimulus blocks are added as EDF annotations ("left / arrow", ...), placed with the per-file
  offset from stimulus_offsets.json. Those offsets are unverified, so annotations say "(offset assumed)"
  unless a measured offset is supplied.
- Units: the text values are written as microvolts. Their real unit and gain are NOT confirmed
  (see WORKLOG.md); converting does not change or repair the data.

Usage: python -m src.export_edf [--out processed_data/edf] [--keep-aux] [--no-annotations]
"""

import argparse
import glob
import os

import mne
import numpy as np

from src.config import AUX_CHANNELS, CHANNELS, OFFSETS_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR, SAMPLING_RATE
from src.raw_io import parse_raw_txt_file
from src.stimulus import BLOCK_DURATION_S, load_offsets, offset_for, video_blocks

UV = 1e-6   # MNE stores EEG in volts; EDF stores the physical unit as uV


def edf_channel_name(name):
    """'FP2_F8' -> 'FP2-F8' (bipolar derivation label as shown in EEG viewers)."""
    return name.replace('_', '-')


def recording_to_raw(rec, offset_s=0.0, offset_verified=False, annotate=True):
    names = rec.channel_names
    ch_types = ['misc' if n in AUX_CHANNELS else 'eeg' for n in names]
    info = mne.create_info([edf_channel_name(n) for n in names], sfreq=rec.fs, ch_types=ch_types)
    raw = mne.io.RawArray(rec.data.astype(np.float64) * UV, info, verbose='error')
    if annotate:
        tag = '' if offset_verified else ' (offset assumed)'
        onsets, durations, descs = [], [], []
        for b in video_blocks():
            onset = b.onset_s + offset_s
            if 0 <= onset < rec.duration_s:
                onsets.append(onset)
                durations.append(min(BLOCK_DURATION_S, rec.duration_s - onset))
                descs.append(f"{b.direction} / {b.format}{tag}")
        if 0 <= offset_s < rec.duration_s:
            onsets.append(offset_s)
            durations.append(0.0)
            descs.append(f"video start{tag}")
        raw.set_annotations(mne.Annotations(onsets, durations, descs))
    return raw


def convert_all(out_dir, keep_aux=False, annotate=True, offsets_path=OFFSETS_PATH):
    offsets, default, provenance = load_offsets(offsets_path)
    verified = 'NOT VERIFIED' not in provenance.upper()
    files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, '*', '*.txt')))
    if not files:
        raise SystemExit(f"No .txt files under {RAW_DATA_DIR}")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for path in files:
        rec = parse_raw_txt_file(path, keep_aux=keep_aux)
        raw = recording_to_raw(rec, offset_for(rec.subject, rec.file, offsets, default), verified, annotate)
        name = f"subject_{rec.subject}_{os.path.splitext(rec.file)[0]}.edf".replace(' ', '_')
        out = os.path.join(out_dir, name)
        raw.export(out, fmt='edf', overwrite=True, verbose='error')
        written.append(out)
        print(f"{rec.subject:6s} {rec.file:18s} {rec.duration_s:6.1f} s, {len(raw.ch_names)} ch -> {name}", flush=True)
    print(f"\nWrote {len(written)} EDF files to {out_dir}")
    return written


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Convert raw NeuroMax text exports to EDF")
    ap.add_argument('--out', default=os.path.join(PROCESSED_DATA_DIR, 'edf'))
    ap.add_argument('--keep-aux', action='store_true', help="Also write BP1/BP2 auxiliary channels (as misc)")
    ap.add_argument('--no-annotations', action='store_true', help="Do not add stimulus-block annotations")
    a = ap.parse_args()
    convert_all(a.out, a.keep_aux, not a.no_annotations)
