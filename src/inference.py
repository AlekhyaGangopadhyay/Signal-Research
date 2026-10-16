"""
Test-Time and Real-Time Inference Module:
Predicts direction (Left, Right, Forward, Backward) for a single trial given K-shot support calibration trials.

Two input paths:
- Preprocessed windows [18, 512] (as stored in processed_data/tensors): `predict_trial`.
- A live raw stream: `StreamingPredictor` filters causally (no look-ahead) and classifies the latest 2 s.
"""

import os

import numpy as np
import pandas as pd
import torch

from src.config import DIRECTIONS, NUM_CLASSES, TRIAL_SAMPLES, LATENT_DIM, DROPOUT_RATE, TEMPERATURE, NUM_CHANNELS
from src.losses import compute_prototypes
from src.models import adapt_batchnorm, build_model
from src.preprocessing import CausalPreprocessor, euclidean_alignment


class DirectionPredictor:
    def __init__(self, checkpoint_path=None, device="cpu", model_type="spatial_temporal"):
        self.device = device
        cfg = {'model_type': model_type, 'latent_dim': LATENT_DIM, 'dropout_rate': DROPOUT_RATE,
               'num_channels': NUM_CHANNELS, 'temperature': TEMPERATURE}
        state_dict = None
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            state_dict = ckpt.get('model_state_dict', ckpt)
            cfg.update(ckpt.get('config', {}))
        self.config = cfg
        self.temperature = float(cfg['temperature'])
        self.model = build_model(cfg['model_type'], cfg['num_channels'], cfg['latent_dim'], cfg['dropout_rate'])
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.to(device).eval()
        self.prototypes = None

    def calibrate(self, support_trials, support_labels, adapt_bn_on=None):
        """
        support_trials: [N, 18, T] preprocessed windows from the new subject, any order.
        support_labels: [N] direction ids (0..3) or names ('left', ...). Every direction needs >= 1 trial.
        adapt_bn_on:    optional unlabelled windows [M, 18, T] from the same subject for AdaBN.
        """
        labels = np.array([DIRECTIONS.index(l) if isinstance(l, str) else int(l) for l in support_labels])
        if len(labels) != len(support_trials):
            raise ValueError("support_trials and support_labels have different lengths")
        missing = [DIRECTIONS[c] for c in range(NUM_CLASSES) if not np.any(labels == c)]
        if missing:
            raise ValueError(f"no calibration trials for: {missing}")
        if adapt_bn_on is not None:
            adapt_batchnorm(self.model, adapt_bn_on, device=self.device)
        x = torch.as_tensor(np.asarray(support_trials), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z = self.model(x)
            self.prototypes = compute_prototypes(z, torch.as_tensor(labels, device=self.device), NUM_CLASSES)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        print(f"Calibrated with {dict(zip(DIRECTIONS, counts.tolist()))} trials per direction.")

    def predict_trial(self, trial_data):
        """
        Predicts direction for a single preprocessed trial [18, T].
        Returns: predicted direction string, confidence scores dictionary.
        """
        if self.prototypes is None:
            raise RuntimeError("Model must be calibrated with support trials before prediction!")
        x = torch.as_tensor(np.asarray(trial_data), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            z_q = self.model(x)
            dists = torch.cdist(z_q, self.prototypes, p=2) ** 2
            probs = torch.softmax(-dists / self.temperature, dim=1).cpu().numpy().flatten()
        pred_id = int(np.argmax(probs))
        return DIRECTIONS[pred_id], {DIRECTIONS[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    def predict_from_csv(self, csv_file_path):
        """
        Predicts from a CSV window written by `convert_raw_data --csv` (already filtered and scaled).
        """
        df = pd.read_csv(csv_file_path)
        channel_cols = [c for c in df.columns if c != 'time_ms']
        if len(channel_cols) != self.config['num_channels']:
            raise ValueError(f"expected {self.config['num_channels']} channels, got {len(channel_cols)}")
        return self.predict_trial(df[channel_cols].values.T.astype(np.float32))


class StreamingPredictor:
    """
    Real-time wrapper: push raw chunks [18, n] as they arrive; `predict` classifies the latest 2 s.
    Filtering is causal (state carried across chunks), so no future samples are used.
    """

    def __init__(self, predictor, calibration_raw, n_channels=NUM_CHANNELS):
        self.predictor = predictor
        self.pre = CausalPreprocessor(n_channels).fit(calibration_raw)
        self.buffer = np.zeros((n_channels, 0), dtype=np.float32)

    def push(self, raw_chunk):
        self.buffer = np.concatenate([self.buffer, self.pre.process(raw_chunk)], axis=1)[:, -4 * TRIAL_SAMPLES:]

    def ready(self):
        return self.buffer.shape[1] >= TRIAL_SAMPLES

    def predict(self):
        if not self.ready():
            raise RuntimeError("need at least one full window of data")
        return self.predictor.predict_trial(self.buffer[:, -TRIAL_SAMPLES:])


def align_windows(windows):
    """Euclidean alignment for a set of windows from one recording, as used in training (alignment='euclidean')."""
    return euclidean_alignment(np.asarray(windows, dtype=np.float32))
