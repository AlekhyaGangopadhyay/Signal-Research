"""
Signal Preprocessing and Filtering Utilities for Bipolar EEG Signals.

Offline path (training/evaluation): zero-phase second-order-section filters over a whole recording,
then per-recording robust scaling.
Online path (real-time inference): `CausalPreprocessor`, same filters applied causally with state.
"""

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos

from src.config import (
    SAMPLING_RATE, BANDPASS_LOW, BANDPASS_HIGH, BUTTER_ORDER, NOTCH_FREQ, NOTCH_Q
)


def bandpass_sos(fs=SAMPLING_RATE, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH, order=BUTTER_ORDER):
    return butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')


def notch_sos(fs=SAMPLING_RATE, freq=NOTCH_FREQ, q=NOTCH_Q):
    b, a = iirnotch(freq, q, fs)
    return tf2sos(b, a)


def butter_bandpass_filter(data, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH, fs=SAMPLING_RATE, order=BUTTER_ORDER):
    """Zero-phase Butterworth band-pass in second-order sections. data: (..., time)."""
    return sosfiltfilt(bandpass_sos(fs, lowcut, highcut, order), data, axis=-1)


def notch_filter(data, freq=NOTCH_FREQ, q=NOTCH_Q, fs=SAMPLING_RATE):
    """Zero-phase IIR notch for power-line interference. data: (..., time)."""
    return sosfiltfilt(notch_sos(fs, freq, q), data, axis=-1)


def preprocess_continuous_eeg(raw_matrix, fs=SAMPLING_RATE, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH,
                              notch_f=NOTCH_FREQ):
    """
    Offline pipeline for a whole recording [C, T]:
    1. Remove per-channel DC offset
    2. Band-pass (0.5 - 45 Hz)
    3. Notch (50 Hz)
    Filter the full recording before epoching so filter edge transients only touch its ends.
    """
    x = np.asarray(raw_matrix, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    x = butter_bandpass_filter(x, lowcut=lowcut, highcut=highcut, fs=fs)
    x = notch_filter(x, freq=notch_f, fs=fs)
    return x.astype(np.float32)


def robust_scale_stats(x, eps=1e-6):
    """Per-channel median and IQR of a [C, T] recording."""
    median = np.median(x, axis=-1, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25], axis=-1, keepdims=True)
    iqr = q75 - q25
    iqr[iqr < eps] = 1.0
    return median, iqr


def robust_scale_recording(x):
    """
    Scales a whole recording [C, T] per channel with its median and IQR.
    Unlike per-epoch scaling, this keeps power differences between epochs of the same recording.
    """
    median, iqr = robust_scale_stats(x)
    return ((x - median) / iqr).astype(np.float32)


def robust_standardize(epoch_data, eps=1e-6):
    """
    Per-epoch, per-channel median/IQR scaling of a [C, T] epoch.
    Kept for compatibility; removes absolute power differences between epochs, so prefer
    `robust_scale_recording` or `euclidean_alignment`.
    """
    median, iqr = robust_scale_stats(epoch_data, eps)
    return (epoch_data - median) / (iqr + eps)


def flag_artifacts(epochs, max_abs=12.0, flat_std=1e-3):
    """
    Flags epochs [N, C, T] that are scaled per recording (`robust_scale_recording`).
    An epoch is bad if any channel exceeds `max_abs` robust units (blinks, movement, saturation)
    or is flat. Returns (bad_mask [N], reasons list[str]).
    """
    peak = np.abs(epochs).max(axis=-1)              # [N, C]
    std = epochs.std(axis=-1)                       # [N, C]
    too_big = (peak > max_abs).any(axis=1)
    flat = (std < flat_std).any(axis=1)
    reasons = []
    for big, fl in zip(too_big, flat):
        r = []
        if big:
            r.append('amplitude')
        if fl:
            r.append('flat')
        reasons.append('+'.join(r))
    return too_big | flat, reasons


def _inv_sqrtm(mat):
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 1e-10, None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def euclidean_alignment(epochs):
    """
    Euclidean alignment (He & Wu, 2019): whitens epochs [N, C, T] by the inverse square root of
    their mean spatial covariance so every recording/subject has an identity mean covariance.
    Uses no labels.
    """
    covs = np.einsum('nct,ndt->ncd', epochs, epochs) / epochs.shape[-1]
    r_inv = _inv_sqrtm(covs.mean(axis=0))
    return np.einsum('cd,ndt->nct', r_inv, epochs).astype(np.float32)


class CausalPreprocessor:
    """
    Streaming version of the offline filters for real-time use. Feed consecutive chunks [C, n];
    filter state is carried across calls, so output only depends on past samples.
    Scaling uses median/IQR measured on a calibration recording (`fit`).
    """

    def __init__(self, n_channels, fs=SAMPLING_RATE):
        self.sos = np.vstack([bandpass_sos(fs), notch_sos(fs)])
        self.n_channels = n_channels
        self.zi = None
        self.median = np.zeros((n_channels, 1), dtype=np.float32)
        self.iqr = np.ones((n_channels, 1), dtype=np.float32)

    def reset(self):
        self.zi = None

    def _filter(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float64)
        if self.zi is None:
            zi = sosfilt_zi(self.sos)                                   # [sections, 2]
            self.zi = zi[:, None, :] * chunk[:, :1][None, :, :]        # [sections, C, 2]
        out, self.zi = sosfilt(self.sos, chunk, axis=-1, zi=self.zi)
        return out

    def fit(self, calibration_raw):
        """Measure scaling on a raw calibration recording [C, T] (filter state is reset afterwards)."""
        self.reset()
        filtered = self._filter(calibration_raw)
        self.median, self.iqr = robust_scale_stats(filtered[:, filtered.shape[1] // 10:])  # skip warm-up
        self.reset()
        return self

    def process(self, chunk):
        return ((self._filter(chunk) - self.median) / self.iqr).astype(np.float32)
