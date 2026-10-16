import numpy as np
import pytest

from src.config import BLANK_LABEL, DIRECTIONS
from src.stimulus import (BLOCK_DURATION_S, stimulus_on_mask, stimulus_windows, video_blocks, offset_for,
                          recording_key)


def test_video_blocks_order():
    blocks = video_blocks()
    assert len(blocks) == 12
    assert [b.direction for b in blocks[:4]] == DIRECTIONS
    assert [b.format for b in blocks] == ['arrow'] * 4 + ['letter'] * 4 + ['word'] * 4
    assert blocks[0].onset_s == pytest.approx(4.2)
    assert blocks[11].onset_s == pytest.approx(169.2)


def test_windows_inside_blocks_and_gaps():
    blocks = {b.index: b for b in video_blocks()}
    for w in stimulus_windows(200.0, offset_s=2.0):
        onset = blocks[w.block].onset_s + 2.0
        if w.kind == 'stim':
            assert onset <= w.start_s and w.start_s + 2.0 <= onset + BLOCK_DURATION_S + 1e-9
            assert w.label == blocks[w.block].label
        else:
            assert w.label == BLANK_LABEL
            assert w.start_s >= onset + BLOCK_DURATION_S


def test_window_counts_and_truncation():
    full = [w for w in stimulus_windows(180.0) if w.kind == 'stim']
    short = [w for w in stimulus_windows(170.0) if w.kind == 'stim']
    assert len(full) == 12 * 7
    assert len(short) == 11 * 7          # block 12 does not fit in 170 s


def test_offset_shifts_windows():
    a = stimulus_windows(200.0, offset_s=0.0)
    b = stimulus_windows(200.0, offset_s=3.0)
    assert b[0].start_s == pytest.approx(a[0].start_s + 3.0)


def test_negative_offset_drops_windows_before_zero():
    assert all(w.start_s >= 0 for w in stimulus_windows(180.0, offset_s=-10.0))


def test_on_mask():
    m = stimulus_on_mask(int(20 * 256), 256.0)
    assert not m[int(4.0 * 256)] and m[int(5.0 * 256)] and not m[int(14.0 * 256)]


def test_offset_lookup():
    offs = {recording_key('Gope', 'a.txt'): 2.5}
    assert offset_for('Gope', 'a.txt', offs, 0.0) == 2.5
    assert offset_for('Gope', 'b.txt', offs, 1.0) == 1.0
