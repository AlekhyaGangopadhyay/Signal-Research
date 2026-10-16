"""
Configuration Module for Few-Shot EEG Direction Detection Pipeline.

Single source of truth for paths, signal parameters and training defaults.
Importing this module has no side effects; call `ensure_dirs()` before writing outputs.
"""

import os
import random

import numpy as np

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "Raw data")
STIMULI_DIR = os.path.join(BASE_DIR, "Stimuli")
OFFSETS_PATH = os.path.join(BASE_DIR, "stimulus_offsets.json")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
CSV_TRIALS_DIR = os.path.join(PROCESSED_DATA_DIR, "csv_trials")
TENSORS_DIR = os.path.join(PROCESSED_DATA_DIR, "tensors")
METADATA_PATH = os.path.join(PROCESSED_DATA_DIR, "metadata.csv")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def ensure_dirs():
    """Create output directories. Call this from entry points, not at import time."""
    for folder in [PROCESSED_DATA_DIR, TENSORS_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
        os.makedirs(folder, exist_ok=True)


# 10-20 Longitudinal Bipolar Montage Channels (18 channels), in file order.
# The raw files label them e.g. "FP2- F8"; `raw_io.normalise_channel_name` maps that to "FP2_F8".
CHANNELS = [
    'FP2_F8', 'F8_T4', 'T4_T6', 'T6_O2',
    'FP2_F4', 'F4_C4', 'C4_P4', 'P4_O2',
    'FP1_F7', 'F7_T3', 'T3_T5', 'T5_O1',
    'FP1_F3', 'F3_C3', 'C3_P3', 'P3_O1',
    'FZ_CZ', 'CZ_PZ'
]
NUM_CHANNELS = len(CHANNELS)
# Auxiliary series present in every page; constant offsets, not EEG. Parsed and discarded.
AUX_CHANNELS = ['BP1_REF', 'BP2_REF']
OCCIPITAL_CHANNELS = ['T6_O2', 'P4_O2', 'T5_O1', 'P3_O1']
FRONTAL_POLE_CHANNELS = ['FP2_F8', 'FP2_F4', 'FP1_F7', 'FP1_F3']

# Signal Processing Parameters
# 256 Hz: one page = 2560 samples = 10.0 s, and line noise lands exactly on 50.0 Hz (see CRITICAL_REVIEW.md §3.2).
SAMPLING_RATE = 256.0       # Hz
EPOCH_DURATION = 2.0        # Seconds per trial
TRIAL_SAMPLES = int(SAMPLING_RATE * EPOCH_DURATION)  # 512 samples
EPOCH_START = 0.5           # Seconds after block onset where the first window starts
EPOCH_STRIDE = 1.0          # Seconds between consecutive windows inside a block

BANDPASS_LOW = 0.5          # Hz
BANDPASS_HIGH = 45.0        # Hz
BUTTER_ORDER = 4
NOTCH_FREQ = 50.0           # Hz (Powerline frequency)
NOTCH_Q = 30.0

# Class Definitions
DIRECTIONS = ['left', 'right', 'forward', 'backward']
CLASS_TO_ID = {d: i for i, d in enumerate(DIRECTIONS)}
ID_TO_CLASS = {i: d for i, d in enumerate(DIRECTIONS)}
NUM_CLASSES = len(DIRECTIONS)  # 4
BLANK_LABEL = -1            # Label stored for blank-screen windows (used by sanity checks only)

# Few-Shot Learning Hyperparameters (the only defaults; entry points must not redefine them)
N_WAY = 4                   # 4 directions
K_SHOT = 5                  # Default support shots per class
TRAIN_K_RANGE = (1, 5)      # K is drawn uniformly from this range for each training episode
Q_QUERY = 5                 # Query samples per class in each training episode
LATENT_DIM = 64             # Latent embedding dimension
TEMPERATURE = 1.0           # Metric temperature (squared distances on the unit sphere are <= 4)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_EPISODES = 50         # Number of episodes per epoch
NUM_EPOCHS = 60
PATIENCE = 10               # Early-stopping patience in epochs
DROPOUT_RATE = 0.5
USE_ANGULAR_LOSS = False    # Directional angular loss; off by default until an ablation supports it
ANGULAR_LAMBDA = 0.1

# Evaluation
N_PERMUTATIONS = 1000
SEED = 42


def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
