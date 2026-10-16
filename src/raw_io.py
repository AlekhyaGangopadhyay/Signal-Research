"""
Reader for the raw multi-page EEG text exports.

File layout (one line, tab-separated tokens):
    Page-1 <tab> FP2- F8 <tab> v,v,v,... <tab> F8 - T4 <tab> v,v,v,... ... BP2- REF <tab> v,v,... <tab> Page-2 ...
Each page holds 20 named series (18 bipolar EEG channels + BP1/BP2 auxiliary inputs) of 2560 samples
(10.0 s at 256 Hz). Pages are consecutive in time, so they are concatenated.
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np

from src.config import CHANNELS, AUX_CHANNELS, SAMPLING_RATE

EXPECTED_SERIES = CHANNELS + AUX_CHANNELS
_PAGE_RE = re.compile(r'^Page-(\d+)$')


class RawFormatError(ValueError):
    """Raised when a raw export does not match the expected layout."""


@dataclass
class RawRecording:
    path: str
    subject: str
    file: str
    data: np.ndarray            # [18, T] float32, EEG channels only, in CHANNELS order
    fs: float = SAMPLING_RATE
    n_pages: int = 0
    page_samples: int = 0
    channel_names: list = field(default_factory=lambda: list(CHANNELS))

    @property
    def duration_s(self):
        return self.data.shape[1] / self.fs


def normalise_channel_name(name):
    """'FP2- F8' -> 'FP2_F8', 'F8 - T4' -> 'F8_T4', 'BP1- REF' -> 'BP1_REF'."""
    parts = [p.strip().upper() for p in name.split('-')]
    if len(parts) != 2 or not all(parts):
        raise RawFormatError(f"Unrecognised channel label {name!r}")
    return f"{parts[0]}_{parts[1]}"


def subject_from_folder(folder_name):
    """'Raw data(Subject Gope)' -> 'Gope', 'Raw data(Subject 3)' -> '3'."""
    m = re.search(r'\(Subject\s*([^)]*)\)', folder_name)
    return m.group(1).strip() if m else folder_name


def _parse_series(token, where):
    values = token.split(',')
    if values and values[-1].strip() == '':
        values = values[:-1]  # exports end each series with a trailing comma
    try:
        return np.array([float(v) for v in values], dtype=np.float32)
    except ValueError as e:
        raise RawFormatError(f"{where}: non-numeric sample ({e})") from None


def parse_raw_txt_file(file_path, keep_aux=False):
    """
    Parses every page of a raw export and concatenates them in time.

    Raises RawFormatError on any structural problem (missing or reordered channels,
    unequal series lengths, malformed numbers) instead of silently dropping data.
    Returns a RawRecording with data [18, n_pages * page_samples].
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = [t.strip() for t in f.read().split('\t')]
    tokens = [t for t in tokens if t]

    pages = []          # list of dict name -> series
    current = None
    pending_name = None
    for tok in tokens:
        m = _PAGE_RE.match(tok)
        if m:
            if pending_name is not None:
                raise RawFormatError(f"{file_path}: channel {pending_name} has no data before {tok}")
            current = {}
            pages.append(current)
            continue
        if current is None:
            raise RawFormatError(f"{file_path}: data before the first 'Page-N' marker")
        if pending_name is None:
            pending_name = normalise_channel_name(tok)
        else:
            where = f"{os.path.basename(file_path)} page {len(pages)} {pending_name}"
            if pending_name in current:
                raise RawFormatError(f"{where}: duplicated channel")
            current[pending_name] = _parse_series(tok, where)
            pending_name = None
    if pending_name is not None:
        raise RawFormatError(f"{file_path}: channel {pending_name} has no data at end of file")
    if not pages:
        raise RawFormatError(f"{file_path}: no pages found")

    page_len = None
    blocks = []
    for p_idx, page in enumerate(pages, start=1):
        if list(page.keys()) != EXPECTED_SERIES:
            raise RawFormatError(
                f"{file_path} page {p_idx}: channels {list(page.keys())} != expected {EXPECTED_SERIES}")
        lengths = {len(v) for v in page.values()}
        if len(lengths) != 1:
            raise RawFormatError(f"{file_path} page {p_idx}: unequal series lengths {sorted(lengths)}")
        n = lengths.pop()
        if page_len is None:
            page_len = n
        elif n != page_len:
            raise RawFormatError(f"{file_path} page {p_idx}: {n} samples, earlier pages have {page_len}")
        names = EXPECTED_SERIES if keep_aux else CHANNELS
        blocks.append(np.stack([page[c] for c in names]))

    folder = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
    return RawRecording(
        path=file_path,
        subject=subject_from_folder(folder),
        file=os.path.basename(file_path),
        data=np.concatenate(blocks, axis=1).astype(np.float32),
        n_pages=len(pages),
        page_samples=page_len,
        channel_names=list(EXPECTED_SERIES if keep_aux else CHANNELS),
    )
