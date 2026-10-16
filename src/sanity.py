"""
Data-validity checks to run before any decoding work.

1. signal_quality:   does each recording look like EEG at all? (1/f spectrum, heavy tails,
                     smooth sample-to-sample changes, blinks in frontal-pole channels)
2. folding_test:     alignment-free test for any response time-locked to the 15 s block cycle.
                     The recording is cut into 15 s cycles; if a stimulus response exists, the cycles
                     share a common shape regardless of where the video starts. Null distribution:
                     shift every cycle by an independent random circular lag.
3. lag_scan:         stimulus-on vs blank decoding for every offset 0..15 s in 0.25 s steps
                     (the block period is 15 s, so larger offsets repeat these).
4. positive control: inject a small synthetic 15 s periodic response into the real noise and
                     check the folding test finds it, to show the test has power.

Usage:  python -m src.sanity [--estimate-offsets] [--quick]
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    RAW_DATA_DIR, RESULTS_DIR, SAMPLING_RATE, CHANNELS, OCCIPITAL_CHANNELS, FRONTAL_POLE_CHANNELS,
    BASE_DIR, SEED, ensure_dirs
)
from src.preprocessing import preprocess_continuous_eeg
from src.raw_io import parse_raw_txt_file
from src.stimulus import BLOCK_PERIOD_S, recording_key, stimulus_on_mask

FS = SAMPLING_RATE
OCC = [CHANNELS.index(c) for c in OCCIPITAL_CHANNELS]
FP = [CHANNELS.index(c) for c in FRONTAL_POLE_CHANNELS]


# ----------------------------------------------------------------------------- 1. signal quality
def signal_quality(x, fs=FS):
    """x: raw [C, T]. Returns a dict of EEG-likeness statistics and a verdict."""
    x = x.astype(np.float64) - x.mean(axis=1, keepdims=True)
    fr, p = welch(x, fs=fs, nperseg=int(4 * fs))
    pm = p.mean(axis=0)
    m = (fr >= 2) & (fr <= 40)
    slope = np.polyfit(np.log10(fr[m]), np.log10(pm[m]), 1)[0]
    band = lambda a, b: pm[(fr >= a) & (fr < b)].mean()
    line = band(49, 51) / (0.5 * (band(45, 48) + band(52, 55)))
    lag1 = float(np.median([np.corrcoef(ch[:-1], ch[1:])[0, 1] for ch in x]))
    kurt = float(np.median(kurtosis(x, axis=1)))
    # blinks: large slow deflections in frontal-pole channels relative to their own noise level
    sos = butter(4, [0.5, 5], btype='band', fs=fs, output='sos')
    fp = sosfiltfilt(sos, x[FP], axis=-1)
    mad = np.median(np.abs(fp), axis=1, keepdims=True) / 0.6745
    blink_rate = float(((np.abs(fp) > 6 * mad).any(axis=0)).mean())
    stats = {
        'spectral_slope_2_40Hz': float(slope),       # EEG: about -1 to -2.5
        'delta_over_beta': float(band(1, 4) / band(13, 30)),  # EEG: >> 1
        'lag1_autocorr': lag1,                        # EEG at 256 Hz: > 0.9
        'excess_kurtosis': kurt,                      # EEG: > 0 (blinks, artifacts)
        'line_noise_ratio_50Hz': float(line),
        'fp_blink_sample_fraction': blink_rate,       # Gaussian noise: ~0
        'unique_values_ch0': int(len(np.unique(x[0]))),
    }
    criteria = [slope < -0.5, stats['delta_over_beta'] > 2, lag1 > 0.8, kurt > 0.5]
    eeg_like = sum(bool(c) for c in criteria)       # count; adding numpy bools would be a logical OR
    stats['eeg_like_score'] = eeg_like
    stats['verdict'] = 'EEG-like' if eeg_like >= 3 else ('unclear' if eeg_like == 2 else 'noise-like')
    return stats


# ----------------------------------------------------------------------------- 2. folding test
def fold_signals(x, fs=FS):
    """Derived 1-D signals [name -> array] in which a block-locked response would show up."""
    out = {}
    occ = x[OCC]
    sos = butter(4, [0.5, 12], btype='band', fs=fs, output='sos')
    out['occipital_evoked'] = sosfiltfilt(sos, occ, axis=-1).mean(axis=0)
    sos = butter(4, [8, 13], btype='band', fs=fs, output='sos')
    env = np.abs(hilbert(sosfiltfilt(sos, occ, axis=-1), axis=-1)).mean(axis=0)
    out['occipital_alpha_power'] = sosfiltfilt(butter(2, 1.0, fs=fs, output='sos'), env)
    sos = butter(4, [0.3, 5], btype='band', fs=fs, output='sos')
    out['frontal_eye_movement'] = np.abs(sosfiltfilt(sos, x[FP], axis=-1)).mean(axis=0)
    return out


def folding_test(sig, fs=FS, period_s=BLOCK_PERIOD_S, n_perm=500, rng=None):
    """
    Cuts `sig` into whole cycles of `period_s` and measures how much of the variance is explained by
    the across-cycle mean (a time-locked response). Null: each cycle circularly shifted by an
    independent random lag. Returns (statistic, p_value, mean_cycle, peak_phase_s).
    """
    rng = rng or np.random.default_rng(SEED)
    step = max(1, int(fs // 32))                     # decimate to ~32 Hz for speed
    s = sig[::step]
    fsd = fs / step
    L = int(round(period_s * fsd))
    n = len(s) // L
    if n < 3:
        raise ValueError("need at least 3 cycles")
    cyc = s[:n * L].reshape(n, L)
    cyc = (cyc - cyc.mean(axis=1, keepdims=True)) / (cyc.std(axis=1, keepdims=True) + 1e-12)

    def stat(c):
        return float(np.var(c.mean(axis=0)) / np.mean(np.var(c, axis=1)))

    obs = stat(cyc)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shifts = rng.integers(0, L, size=n)
        null[i] = stat(np.stack([np.roll(c, k) for c, k in zip(cyc, shifts)]))
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    mean_cycle = cyc.mean(axis=0)
    return obs, p, mean_cycle, float(np.argmax(np.abs(mean_cycle)) / fsd)


# ----------------------------------------------------------------------------- 3. lag scan
BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]


def _bandpower_grid(x, fs=FS, win_s=2.0, hop_s=0.25):
    """Log band-power features for 2 s windows every `hop_s` seconds. Returns (starts_s, feats)."""
    w = int(win_s * fs)
    hop = int(hop_s * fs)
    starts = np.arange(0, x.shape[1] - w + 1, hop)
    segs = np.stack([x[:, a:a + w] for a in starts])
    fr, p = welch(segs, fs=fs, nperseg=int(fs), axis=-1)
    f = np.stack([p[..., (fr >= a) & (fr < b)].mean(-1) for a, b in BANDS], axis=-1)
    return starts / fs, np.log(f + 1e-12).reshape(len(starts), -1)


def lag_scan(recordings, lags=np.arange(0, 15, 0.25)):
    """
    recordings: list of filtered [C, T] arrays from one subject.
    For each lag, label each 2 s window 'stimulus' if it lies fully inside a block and 'blank'
    if fully inside a gap, then decode with leave-one-file-out CV. Returns accuracies per lag.
    """
    grids = [_bandpower_grid(x) for x in recordings]
    accs = []
    for lag in lags:
        X, y, g = [], [], []
        for gi, ((starts, feats), x) in enumerate(zip(grids, recordings)):
            on = stimulus_on_mask(x.shape[1], FS, offset_s=lag)
            a = (starts * FS).astype(int)
            b = a + int(2 * FS)
            frac_on = np.array([on[s:e].mean() for s, e in zip(a, b)])
            keep = (frac_on == 1) | (frac_on == 0)
            # require windows to be inside the video span (some blank before the first block)
            keep &= (starts >= lag) & (starts + 2 <= lag + 180)
            X.append(feats[keep]); y.append(frac_on[keep] == 1); g.append(np.full(keep.sum(), gi))
        X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(g)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced'))
        accs.append(cross_val_score(clf, X, y, groups=g, cv=LeaveOneGroupOut(), scoring='balanced_accuracy').mean())
    return np.asarray(lags), np.asarray(accs)


# ----------------------------------------------------------------------------- 4. positive control
def positive_control(x, amplitude_sd=0.3, fs=FS, offset_s=7.0):
    """
    Adds a synthetic block-locked occipital response (a 1 s, 5 Hz burst at each block onset,
    `amplitude_sd` x the channel SD) to a real recording and reruns the folding test on it.
    """
    y = x.copy()
    t = np.arange(x.shape[1]) / fs
    resp = np.zeros_like(t)
    for k in range(int(t[-1] // BLOCK_PERIOD_S) + 1):
        t0 = offset_s + 4.2 + BLOCK_PERIOD_S * k
        m = (t >= t0) & (t < t0 + 1.0)
        resp[m] = np.sin(2 * np.pi * 5 * (t[m] - t0)) * np.hanning(m.sum())
    for c in OCC:
        y[c] += amplitude_sd * x[c].std() * resp
    return folding_test(fold_signals(y)['occipital_evoked'], n_perm=200)[1]


# ----------------------------------------------------------------------------- driver
def load_all(raw_dir=RAW_DATA_DIR):
    recs = [parse_raw_txt_file(f) for f in sorted(glob.glob(os.path.join(raw_dir, '*', '*.txt')))]
    return recs


def main():
    ap = argparse.ArgumentParser(description="Data-validity checks")
    ap.add_argument('--estimate-offsets', action='store_true',
                    help="Write stimulus_offsets.estimated.json for files with a significant folding response")
    ap.add_argument('--quick', action='store_true', help="Fewer permutations and a coarser lag scan")
    args = ap.parse_args()
    ensure_dirs()
    n_perm = 100 if args.quick else 500
    rng = np.random.default_rng(SEED)
    lines = []

    def log(s=''):
        print(s, flush=True)
        lines.append(s)

    recs = load_all()
    log("## 1. Signal quality (is it EEG?)")
    log("| Subject | File | slope 2-40 Hz | delta/beta | lag-1 r | kurtosis | 50 Hz ratio | FP blink frac | verdict |")
    log("|---|---|---|---|---|---|---|---|---|")
    quality = {}
    for r in recs:
        q = signal_quality(r.data)
        quality[recording_key(r.subject, r.file)] = q
        log(f"| {r.subject} | {r.file} | {q['spectral_slope_2_40Hz']:.2f} | {q['delta_over_beta']:.2f} | "
            f"{q['lag1_autocorr']:.2f} | {q['excess_kurtosis']:.2f} | {q['line_noise_ratio_50Hz']:.1f} | "
            f"{q['fp_blink_sample_fraction']:.4f} | {q['verdict']} |")
    verdicts = [q['verdict'] for q in quality.values()]
    log(f"\nEEG-like: {verdicts.count('EEG-like')}, unclear: {verdicts.count('unclear')}, "
        f"noise-like: {verdicts.count('noise-like')} of {len(verdicts)} files")

    log("\n## 2. Alignment-free folding test (15 s block cycle), p-values")
    log("| Subject | File | occipital evoked | occipital alpha | frontal eye movement |")
    log("|---|---|---|---|---|")
    filtered = {}
    fold_p = {}
    phases = {}
    for r in recs:
        x = preprocess_continuous_eeg(r.data)
        filtered[recording_key(r.subject, r.file)] = x
        res = {name: folding_test(sig, n_perm=n_perm, rng=rng) for name, sig in fold_signals(x).items()}
        key = recording_key(r.subject, r.file)
        fold_p[key] = {k: v[1] for k, v in res.items()}
        phases[key] = res['occipital_evoked'][3]
        log(f"| {r.subject} | {r.file} | " + " | ".join(f"{res[k][1]:.3f}" for k in res) + " |")
    allp = np.array([list(v.values()) for v in fold_p.values()])
    n_tests = allp.size
    log(f"\n{(allp < 0.05).sum()} of {n_tests} tests have p < 0.05 (expected by chance: {0.05 * n_tests:.1f}); "
        f"{(allp < 0.05 / n_tests).sum()} survive Bonferroni (p < {0.05 / n_tests:.4f}).")

    log("\n## 3. Positive control: synthetic block-locked response injected into real recordings")
    log("| Amplitude (x channel SD) | files detected at p < 0.05 |")
    log("|---|---|")
    sample = recs[::4]
    for amp in (0.1, 0.2, 0.3, 0.5):
        det = sum(positive_control(preprocess_continuous_eeg(r.data), amplitude_sd=amp) < 0.05 for r in sample)
        log(f"| {amp} | {det}/{len(sample)} |")

    log("\n## 4. Stimulus-on vs blank decoding, lag scan 0-15 s (balanced accuracy, chance 50%)")
    lags = np.arange(0, 15, 1.0 if args.quick else 0.25)
    scan = {}
    for subj in sorted({r.subject for r in recs}):
        xs = [filtered[recording_key(r.subject, r.file)] for r in recs if r.subject == subj]
        lg, acc = lag_scan(xs, lags)
        scan[subj] = acc
        log(f"- Subject {subj}: best {acc.max() * 100:.1f}% at lag {lg[acc.argmax()]:.2f} s; "
            f"mean {acc.mean() * 100:.1f}%, range {acc.min() * 100:.1f}-{acc.max() * 100:.1f}% over {len(lg)} lags")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        for subj, acc in scan.items():
            ax.plot(lags, acc * 100, marker='.', label=f"Subject {subj}")
        ax.axhline(50, color='k', ls=':', label='chance')
        ax.set_xlabel('Assumed offset of video start in recording (s, modulo 15)')
        ax.set_ylabel('Stim vs blank balanced accuracy (%)')
        ax.set_title('Lag scan: is any offset decodable?')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'sanity_lag_scan.png'), dpi=150)
        plt.close(fig)
    except Exception as e:  # plotting is optional
        log(f"(plot skipped: {e})")

    if args.estimate_offsets:
        est = {}
        for key, p in fold_p.items():
            if p['occipital_evoked'] < 0.05 / len(fold_p):
                # mean cycle peak marks the response; assume ~0.15 s latency after onset (4.2 s into the cycle)
                est[key] = round(float((phases[key] - 0.15 - 4.2) % BLOCK_PERIOD_S), 2)
        out = os.path.join(BASE_DIR, 'stimulus_offsets.estimated.json')
        if not est:
            log("\nNo file has a significant block-locked response; no offsets estimated.")
        else:
            with open(out, 'w', encoding='utf-8') as f:
                json.dump({'provenance': 'Estimated from folding-test phase (mod 15 s); only files with a '
                                         'Bonferroni-significant occipital response. Review before use.',
                           'offsets_s': est}, f, indent=2)
            log(f"\nEstimated offsets for {len(est)} files written to {out}")

    report = os.path.join(RESULTS_DIR, 'sanity_report.md')
    with open(report, 'w', encoding='utf-8') as f:
        f.write("# Data-validity report\n\nGenerated by `python -m src.sanity`.\n\n" + "\n".join(lines) + "\n")
    print(f"\nReport: {report}")


if __name__ == '__main__':
    main()
