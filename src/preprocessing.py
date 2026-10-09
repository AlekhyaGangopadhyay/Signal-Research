"""
Signal Preprocessing and Filtering Utilities for Bipolar EEG Signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def butter_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=250.0, order=4):
    """
    Applies zero-phase 4th order Butterworth bandpass filter.
    data: shape (..., time_samples)
    """
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-4)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def notch_filter(data, freq=50.0, q=30.0, fs=250.0):
    """
    Applies zero-phase IIR notch filter to remove powerline interference.
    data: shape (..., time_samples)
    """
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, data, axis=-1)

def robust_standardize(epoch_data, eps=1e-6):
    """
    Standardizes data channel-wise using median and Interquartile Range (IQR).
    Robust against high-amplitude transient voltage spikes.
    epoch_data: shape (Channels, Time)
    """
    median = np.median(epoch_data, axis=-1, keepdims=True)
    q75 = np.percentile(epoch_data, 75, axis=-1, keepdims=True)
    q25 = np.percentile(epoch_data, 25, axis=-1, keepdims=True)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return (epoch_data - median) / (iqr + eps)

def preprocess_continuous_eeg(raw_matrix, fs=250.0, lowcut=0.5, highcut=45.0, notch_f=50.0):
    """
    Complete continuous signal preprocessing pipeline:
    1. Bandpass filter (0.5 - 45 Hz)
    2. Notch filter (50 Hz)
    """
    filtered = butter_bandpass_filter(raw_matrix, lowcut=lowcut, highcut=highcut, fs=fs)
    filtered = notch_filter(filtered, freq=notch_f, fs=fs)
    return filtered
