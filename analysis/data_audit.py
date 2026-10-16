"""
Data audit: compares every raw recording with a real EEG reference recording and draws the figures used
in DATA_AUDIT.md.

Reference: PhysioNet EEG Motor Movement/Imagery dataset, subject 1, run 1 (eyes open, 1 min) and run 2
(eyes closed, 1 min), converted to the same 18-channel bipolar montage and resampled to 256 Hz.

Usage: python -I analysis/data_audit.py <repo root> <folder with MNE-eegbci-data, or where to download it>
"""
import glob
import json
import os
import sys

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')
REF_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'processed_data', 'reference')
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, sosfiltfilt, welch  # noqa: E402
from scipy.stats import kurtosis  # noqa: E402

from src.config import CHANNELS, SAMPLING_RATE  # noqa: E402
from src.raw_io import parse_raw_txt_file  # noqa: E402

FS = SAMPLING_RATE
OUT = os.path.join(ROOT, 'docs', 'figures', 'data_audit')
os.makedirs(OUT, exist_ok=True)

# Colours: validated categorical slots 1-3 (dataviz reference palette, light mode)
YOURS, REF_EO, REF_EC = '#eb6834', '#2a78d6', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
                     'axes.titleweight': 'bold', 'axes.titlesize': 11, 'lines.linewidth': 2,
                     'grid.color': GRID, 'grid.linewidth': 0.8})

# 10-20 names used by the montage -> 10-10 names in the reference recording
REF_NAME = {'FP1': 'Fp1', 'FP2': 'Fp2', 'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
            'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz'}


def bipolar_from_reference(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose='error')
    mne.datasets.eegbci.standardize(raw)
    raw.resample(FS, verbose='error')
    x = raw.get_data() * 1e6          # volts -> microvolts
    idx = {n: i for i, n in enumerate(raw.ch_names)}
    rows = []
    for ch in CHANNELS:
        a, b = ch.split('_')
        rows.append(x[idx[REF_NAME.get(a, a.capitalize() if len(a) > 1 else a)]] - x[idx[REF_NAME.get(b, b.capitalize())]])
    return np.array(rows)


def bandpass(x, lo=0.5, hi=45.0):
    return sosfiltfilt(butter(4, [lo, hi], btype='band', fs=FS, output='sos'), x - x.mean(1, keepdims=True), axis=-1)


def stats(x_raw):
    """x_raw: [18, T] in the file's own units."""
    x = x_raw.astype(float) - x_raw.mean(1, keepdims=True)
    fr, p = welch(x, fs=FS, nperseg=int(4 * FS))
    pm = p.mean(0)
    band = lambda a, b: pm[(fr >= a) & (fr < b)].mean()
    m = (fr >= 2) & (fr <= 40)
    slope = np.polyfit(np.log10(fr[m]), np.log10(pm[m]), 1)[0]
    xf = bandpass(x)
    lag1 = np.median([np.corrcoef(c[:-1], c[1:])[0, 1] for c in xf])
    win = xf[:, :xf.shape[1] // (2 * int(FS)) * 2 * int(FS)].reshape(18, -1, 2 * int(FS)).std(-1)
    C = {n: i for i, n in enumerate(CHANNELS)}
    adj = [('FP2_F8', 'F8_T4'), ('F8_T4', 'T4_T6'), ('T4_T6', 'T6_O2'), ('FP2_F4', 'F4_C4'), ('F4_C4', 'C4_P4'),
           ('C4_P4', 'P4_O2'), ('FP1_F7', 'F7_T3'), ('F7_T3', 'T3_T5'), ('T3_T5', 'T5_O1'), ('FP1_F3', 'F3_C3'),
           ('F3_C3', 'C3_P3'), ('C3_P3', 'P3_O1'), ('FZ_CZ', 'CZ_PZ')]
    nb = np.mean([np.corrcoef(xf[C[a]], xf[C[b]])[0, 1] for a, b in adj])
    fp = [C['FP2_F4'], C['FP1_F3'], C['FP2_F8'], C['FP1_F7']]
    occ = [C['P4_O2'], C['P3_O1'], C['T6_O2'], C['T5_O1']]
    fo, po = welch(x[occ], fs=FS, nperseg=int(4 * FS))
    po = po.mean(0)
    alpha_peak = po[(fo >= 8) & (fo <= 13)].max() / np.median(po[((fo >= 4) & (fo < 7)) | ((fo > 15) & (fo <= 20))])
    return {
        'Spectral slope 2-40 Hz': slope,
        'Slow/fast power (1-4 Hz vs 13-30 Hz)': band(1, 4) / band(13, 30),
        'Occipital alpha peak (x background)': alpha_peak,
        'Sample-to-sample smoothness (lag-1 r)': lag1,
        'Spikiness (excess kurtosis, frontal)': float(np.median(kurtosis(xf[fp], axis=1))),
        'Amplitude change over time (CV of 2 s SD)': float(np.median(win.std(1) / win.mean(1))),
        'Neighbouring-channel correlation': nb,
        'Distinct values per channel': float(np.median([len(np.unique(c)) for c in x_raw])),
    }


# ------------------------------------------------------------------ load
recs = [parse_raw_txt_file(f) for f in sorted(glob.glob(os.path.join(ROOT, 'Raw data', '*', '*.txt')))]
ref_paths = mne.datasets.eegbci.load_data(subjects=1, runs=[1, 2], path=REF_DIR, update_path=False, verbose='error')
ref_eo, ref_ec = bipolar_from_reference(ref_paths[0]), bipolar_from_reference(ref_paths[1])

ours = [stats(r.data) for r in recs]
table = {'yours_median': {k: float(np.median([s[k] for s in ours])) for k in ours[0]},
         'yours_min': {k: float(np.min([s[k] for s in ours])) for k in ours[0]},
         'yours_max': {k: float(np.max([s[k] for s in ours])) for k in ours[0]},
         'ref_eyes_open': {k: float(v) for k, v in stats(ref_eo).items()},
         'ref_eyes_closed': {k: float(v) for k, v in stats(ref_ec).items()}}
offsets = np.array([r.data.mean(1) for r in recs])
table['dc_offset_spread_across_files'] = {c: float(s) for c, s in zip(CHANNELS, offsets.std(0))}
table['dc_offset_mean'] = {c: float(m) for c, m in zip(CHANNELS, offsets.mean(0))}
with open(os.path.join(OUT, 'audit_stats.json'), 'w', encoding='utf-8') as f:
    json.dump(table, f, indent=2)
for k in ours[0]:
    print(f"{k:45s} yours {table['yours_median'][k]:8.3f} [{table['yours_min'][k]:.3f}, {table['yours_max'][k]:.3f}]"
          f" | real EO {table['ref_eyes_open'][k]:8.3f} | real EC {table['ref_eyes_closed'][k]:8.3f}")

example = next(r for r in recs if r.subject == 'Gope' and r.file == 'AllPages_7.txt')

# ------------------------------------------------------------------ fig 1: traces
show = ['FP2_F4', 'F4_C4', 'C4_P4', 'P4_O2', 'FP1_F3', 'F3_C3', 'C3_P3', 'P3_O1']
C = {n: i for i, n in enumerate(CHANNELS)}


def trace_panel(ax, x, color, title, t0, unit, scale):
    xf = bandpass(x)
    seg = xf[:, int(t0 * FS):int((t0 + 10) * FS)]
    t = np.arange(seg.shape[1]) / FS
    for k, ch in enumerate(show):
        ax.plot(t, seg[C[ch]] / scale - k, color=color, lw=0.8)
    ax.set_yticks(-np.arange(len(show)))
    ax.set_yticklabels([c.replace('_', '-') for c in show])
    ax.set_xlabel('Time (s)')
    ax.set_title(title, loc='left')
    ax.set_xlim(0, 10)
    ax.set_ylim(-len(show) + 0.2, 1.2)
    ax.plot([9.6, 9.6], [0.9, 0.9 - 1], color=INK, lw=2)
    ax.text(9.5, 0.4, f"{scale:g} {unit}", ha='right', va='center', color=INK2, fontsize=9)


s_ours = 4 * np.median(bandpass(example.data).std(1))
s_ref = 4 * np.median(bandpass(ref_eo).std(1))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
trace_panel(axes[0], example.data, YOURS, 'Your recording (Gope, AllPages_7)', 30, 'units', round(s_ours, 1))
trace_panel(axes[1], ref_eo, REF_EO, 'Real EEG (PhysioNet, eyes open)', 0, 'µV', round(s_ref, -1))
fig.suptitle('10 seconds of the same 8 channels: yours is uniform fuzz, real EEG has rolling waves and blinks',
             x=0.01, ha='left', fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT, '1_traces.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ fig 2: spectra
def norm_psd(x, chans=None):
    xx = x if chans is None else x[chans]
    fr, p = welch(xx - xx.mean(1, keepdims=True), fs=FS, nperseg=int(4 * FS))
    pm = p.mean(0)
    m = (fr >= 1) & (fr <= 45)
    return fr[m], pm[m] / np.trapezoid(pm[m], fr[m])


occ = [C[c] for c in ['P4_O2', 'P3_O1', 'T6_O2', 'T5_O1']]
fig, ax = plt.subplots(figsize=(9, 5.5))
ours_curves = np.array([norm_psd(r.data, occ)[1] for r in recs])
f, _ = norm_psd(recs[0].data, occ)
ax.fill_between(f, ours_curves.min(0), ours_curves.max(0), color=YOURS, alpha=0.2, lw=0)
ax.plot(f, np.median(ours_curves, 0), color=YOURS)
for x, col in [(ref_eo, REF_EO), (ref_ec, REF_EC)]:
    ax.plot(*norm_psd(x, occ), color=col)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 45)
ax.set_xticks([1, 2, 4, 8, 10, 13, 20, 30, 45])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('Frequency (Hz, log scale)')
ax.set_ylabel('Relative power (log scale)')
ax.grid(True, which='major')
ax.text(1.1, np.median(ours_curves, 0)[0] * 0.55, 'Your 23 recordings (flat = noise)', color=INK, fontsize=10)
fe, pe = norm_psd(ref_eo, occ)
fc, pc = norm_psd(ref_ec, occ)
ax.text(1.1, pe[0] * 1.15, 'Real EEG, eyes open', color=INK, fontsize=10)
k = np.argmax(pc * ((fc > 8) & (fc < 13)))
ax.annotate('Real EEG, eyes closed:\nalpha peak at ~10 Hz', (fc[k], pc[k]), (16, pc[k] * 0.45), color=INK, fontsize=10,
            arrowprops={'arrowstyle': '-', 'color': INK2, 'lw': 1})
ax.set_title('Occipital power spectrum: real EEG slopes down and has an alpha peak; yours is flat', loc='left')
handles = [plt.Line2D([], [], color=c, lw=2) for c in (YOURS, REF_EO, REF_EC)]
ax.legend(handles, ['Your recordings (median, band = all 23)', 'Real EEG, eyes open', 'Real EEG, eyes closed'],
          frameon=False, loc='lower left')
fig.tight_layout()
fig.savefig(os.path.join(OUT, '2_spectrum.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ fig 3: amplitude distribution (frontal)
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(-12, 12, 97)
for x, col, lab in [(example.data, YOURS, 'Your recording (Gope, AllPages_7)'), (ref_eo, REF_EO, 'Real EEG, eyes open')]:
    xf = bandpass(x, 0.5, 10)[C['FP1_F3']]
    z = (xf - np.median(xf)) / (np.subtract(*np.percentile(xf, [75, 25])) / 1.349)
    h, e = np.histogram(z, bins=bins, density=True)
    ax.step(e[:-1], h, where='post', color=col, label=lab)
g = np.exp(-0.5 * bins ** 2) / np.sqrt(2 * np.pi)
ax.plot(bins, g, color=INK2, lw=1, ls='--', label='Perfect random noise (Gaussian)')
ax.set_yscale('log')
ax.set_ylim(1e-5, 1)
ax.set_xlabel('Frontal channel FP1-F3 amplitude (in units of its own typical size)')
ax.set_ylabel('How often (log scale)')
ax.set_title('Frontal channel: real EEG has big rare swings (blinks); yours never leaves the noise curve', loc='left')
ax.legend(frameon=False)
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, '3_amplitude_distribution.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ fig 4: DC offsets
fig, ax = plt.subplots(figsize=(12, 5))
markers = {'3': 'o', 'Gope': 's', 'Supra': '^'}
for subj, mk in markers.items():
    rows = [i for i, r in enumerate(recs) if r.subject == subj]
    for j, i in enumerate(rows):
        ax.scatter(np.arange(18) + {'3': -0.2, 'Gope': 0, 'Supra': 0.2}[subj], offsets[i], s=22, marker=mk,
                   color=YOURS, edgecolor=SURF, linewidth=0.6, label=f"Subject {subj}" if j == 0 else None, zorder=3)
ax.set_xticks(np.arange(18))
ax.set_xticklabels([c.replace('_', '-') for c in CHANNELS], rotation=45, ha='right')
ax.set_ylabel('Average level of the channel (file units)')
ax.axhline(0, color=INK2, lw=0.8)
ax.grid(True, axis='y')
ax.set_title('Each channel sits at the same level for all 3 people in all 23 recordings: '
             'that is the amplifier, not a head', loc='left')
ax.legend(frameon=False, ncol=3, loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUT, '4_dc_offsets.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ fig 5: amplitude over time
def sd_over_time(x, ch):
    xf = bandpass(x)[C[ch]]
    n = len(xf) // int(2 * FS)
    s = xf[:n * int(2 * FS)].reshape(n, -1).std(1)
    return np.arange(n) * 2, s / np.median(s)


fig, ax = plt.subplots(figsize=(10, 4.5))
for x, col, lab in [(example.data, YOURS, 'Your recording (Gope, AllPages_7)'), (ref_eo, REF_EO, 'Real EEG, eyes open')]:
    t, s = sd_over_time(x, 'FP2_F4')
    ax.plot(t, s, color=col, marker='o', ms=4, label=lab)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Size of signal in each 2 s\n(1 = its usual size)')
ax.set_title('Real EEG gets bigger and smaller over time (blinks, movement, attention); yours stays constant', loc='left')
ax.legend(frameon=False)
ax.grid(True, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(OUT, '5_amplitude_over_time.png'), dpi=150)
plt.close(fig)
print('figures written to', OUT)
