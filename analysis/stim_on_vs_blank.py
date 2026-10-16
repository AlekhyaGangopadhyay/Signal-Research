"""
Stimulus-on vs blank decoding as a function of the assumed video offset.

The first version scanned lags -5, 0, 5, 10, 20, 40 s. Blocks repeat every 15 s, so those are only
3 distinct offsets (0, 5, 10 s mod 15). This version scans 0-15 s in 0.25 s steps using src.sanity.

Usage: python -I analysis/stim_on_vs_blank.py <repo root>
"""
import os
import sys

sys.path.insert(0, os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.'))

from src.preprocessing import preprocess_continuous_eeg  # noqa: E402
from src.sanity import lag_scan, load_all  # noqa: E402

recs = load_all()
for subj in sorted({r.subject for r in recs}):
    xs = [preprocess_continuous_eeg(r.data) for r in recs if r.subject == subj]
    lags, acc = lag_scan(xs)
    print(f"Sub {subj}: stim-ON vs blank, balanced accuracy over {len(lags)} offsets (chance 50%): "
          f"best {acc.max() * 100:.1f}% at {lags[acc.argmax()]:.2f} s, mean {acc.mean() * 100:.1f}%")
