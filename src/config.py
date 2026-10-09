"""
Configuration Module for Few-Shot EEG Direction Detection Pipeline.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "Raw data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
CSV_TRIALS_DIR = os.path.join(PROCESSED_DATA_DIR, "csv_trials")
TENSORS_DIR = os.path.join(PROCESSED_DATA_DIR, "tensors")
METADATA_PATH = os.path.join(PROCESSED_DATA_DIR, "metadata.csv")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure required output directories exist
for folder in [PROCESSED_DATA_DIR, CSV_TRIALS_DIR, TENSORS_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# 10-20 Longitudinal Bipolar Montage Channels (18 channels)
CHANNELS = [
    'FP2_F8', 'F8_T4', 'T4_T6', 'T6_O2',
    'FP2_F4', 'F4_C4', 'C4_P4', 'P4_O2',
    'FP1_F7', 'F7_T3', 'T3_T5', 'T5_O1',
    'FP1_F3', 'F3_C3', 'C3_P3', 'P3_O1',
    'FZ_CZ', 'CZ_PZ'
]
NUM_CHANNELS = len(CHANNELS)

# Signal Processing Parameters
SAMPLING_RATE = 250.0       # Hz
EPOCH_DURATION = 2.0        # Seconds per trial
TRIAL_SAMPLES = int(SAMPLING_RATE * EPOCH_DURATION) # 500 samples

BANDPASS_LOW = 0.5          # Hz
BANDPASS_HIGH = 45.0        # Hz
BUTTER_ORDER = 4
NOTCH_FREQ = 50.0           # Hz (Powerline frequency)
NOTCH_Q = 30.0

# Class Definitions
DIRECTIONS = ['left', 'right', 'forward', 'backward']
CLASS_TO_ID = {d: i for i, d in enumerate(DIRECTIONS)}
ID_TO_CLASS = {i: d for i, d in enumerate(DIRECTIONS)}
NUM_CLASSES = len(DIRECTIONS) # 4

# Few-Shot Learning Hyperparameters
N_WAY = 4                   # 4 directions
K_SHOT = 5                  # Default support shots per class
Q_QUERY = 5                 # Query samples per class in each episode
LATENT_DIM = 128            # Latent embedding dimension
TEMPERATURE = 0.1           # Metric temperature scale
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_EPISODES = 100        # Number of episodes per epoch
NUM_EPOCHS = 80
DROPOUT_RATE = 0.25
