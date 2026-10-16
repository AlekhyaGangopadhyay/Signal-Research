"""
Stimulus timeline of the video in Stimuli/ and mapping from video time to recording time.

Verified from the video frames (analysis/stimulus_timeline.py, WORKLOG step 4):
    title card 0.5-1.5 s, then 12 blocks of 9.2 s with onsets at 4.2 + 15*i s (i = 0..11),
    separated by 5.8 s blank (white) screens. Block order:
    arrows  <- -> ^ v   |  letters L R F B  |  words LEFT RIGHT FORWARD BACKWARD

Each recording is one viewing of the video. Where the video starts inside the recording is
given per file by `stimulus_offsets.json`: recording_time = video_time + offset_s.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

from src.config import (
    DIRECTIONS, OFFSETS_PATH, EPOCH_DURATION, EPOCH_START, EPOCH_STRIDE, BLANK_LABEL
)

VIDEO_DURATION_S = 180.2
FIRST_ONSET_S = 4.2
BLOCK_PERIOD_S = 15.0
BLOCK_DURATION_S = 9.2
N_BLOCKS = 12
FORMATS = ['arrow', 'letter', 'word']


@dataclass(frozen=True)
class Block:
    index: int          # 0..11, position in the video
    onset_s: float      # video time
    direction: str
    label: int          # index into DIRECTIONS
    format: str         # arrow / letter / word
    repetition: int     # 0..2, same as the format index


def video_blocks():
    """The 12 stimulus blocks in video time."""
    blocks = []
    for i in range(N_BLOCKS):
        d = i % len(DIRECTIONS)
        blocks.append(Block(
            index=i,
            onset_s=FIRST_ONSET_S + BLOCK_PERIOD_S * i,
            direction=DIRECTIONS[d],
            label=d,
            format=FORMATS[i // len(DIRECTIONS)],
            repetition=i // len(DIRECTIONS),
        ))
    return blocks


def recording_key(subject, file):
    return f"{subject}/{file}"


def load_offsets(path=OFFSETS_PATH):
    """
    Returns (offsets_dict, default_offset, provenance). Missing file -> every offset is the default 0.0.
    Format:
        {"default_offset_s": 0.0, "provenance": "...", "offsets_s": {"Gope/AllPages_7.txt": 3.25, ...}}
    """
    if not os.path.exists(path):
        return {}, 0.0, "assumed: no stimulus_offsets.json"
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg.get('offsets_s', {}), float(cfg.get('default_offset_s', 0.0)), cfg.get('provenance', '')


def offset_for(subject, file, offsets=None, default=0.0):
    offsets = offsets or {}
    return float(offsets.get(recording_key(subject, file), default))


@dataclass(frozen=True)
class Window:
    start_s: float      # recording time
    label: int          # direction id, or BLANK_LABEL
    block: int          # block index 0..11 (blank windows carry the block they follow)
    format: str
    kind: str           # 'stim' or 'blank'


def stimulus_windows(duration_s, offset_s=0.0, win_s=EPOCH_DURATION,
                     start_s=EPOCH_START, stride_s=EPOCH_STRIDE, include_blank=True):
    """
    Windows locked to block onsets, in recording time. Windows that would fall outside
    [0, duration_s] are dropped.

    stim windows:  onset + start_s + k*stride_s, fully inside the 9.2 s block (7 per block by default)
    blank windows: non-overlapping windows starting 0.5 s after block offset, fully inside the 5.8 s gap (2 per block)
    """
    out = []
    eps = 1e-9
    for b in video_blocks():
        onset = b.onset_s + offset_s
        t = onset + start_s
        while t + win_s <= onset + BLOCK_DURATION_S + eps:
            if t >= 0 and t + win_s <= duration_s + eps:
                out.append(Window(round(t, 6), b.label, b.index, b.format, 'stim'))
            t += stride_s
        if include_blank:
            gap_end = onset + BLOCK_PERIOD_S
            t = onset + BLOCK_DURATION_S + 0.5
            while t + win_s <= gap_end + eps:
                if t >= 0 and t + win_s <= duration_s + eps:
                    out.append(Window(round(t, 6), BLANK_LABEL, b.index, b.format, 'blank'))
                t += win_s
    return out


def stimulus_on_mask(n_samples, fs, offset_s=0.0):
    """Boolean [n_samples] mask, True while a stimulus block is on screen."""
    t = np.arange(n_samples) / fs - offset_s
    mask = np.zeros(n_samples, dtype=bool)
    for b in video_blocks():
        mask |= (t >= b.onset_s) & (t < b.onset_s + BLOCK_DURATION_S)
    return mask
