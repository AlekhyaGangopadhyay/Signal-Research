"""
Test-Time and Real-Time Inference Module:
Predicts direction (Left, Right, Forward, Backward) for a single trial given K-shot support calibration trials.
"""

import os
import torch
import numpy as np
import pandas as pd

from src.config import (
    CHECKPOINTS_DIR, DIRECTIONS, SAMPLING_RATE,
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ
)
from src.preprocessing import preprocess_continuous_eeg, robust_standardize
from src.models import SpatialTemporalProtoNet

class DirectionPredictor:
    def __init__(self, checkpoint_path=None, device="cpu"):
        self.device = device
        self.model = SpatialTemporalProtoNet(num_channels=18, latent_dim=128)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(state_dict)
            print(f"Loaded trained checkpoint: {checkpoint_path}")
        
        self.model.to(device)
        self.model.eval()
        self.prototypes = None

    def calibrate(self, support_trials, support_labels, k_shot=5):
        """
        Calibrates with K support trials per direction from target subject.
        support_trials: [4 * k_shot, 18, 500]
        support_labels: [4 * k_shot]
        """
        x_tensor = torch.tensor(support_trials, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            z = self.model(x_tensor) # [4*K, D]
            z_reshaped = z.view(4, k_shot, -1)
            self.prototypes = z_reshaped.mean(dim=1) # [4, D]
        print(f"Calibrated with {k_shot}-shot support trials.")

    def predict_trial(self, trial_data):
        """
        Predicts direction for a single trial [18, 500].
        Returns: predicted direction string, confidence scores dictionary.
        """
        if self.prototypes is None:
            raise RuntimeError("Model must be calibrated with support trials before prediction!")

        x_tensor = torch.tensor(trial_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z_q = self.model(x_tensor) # [1, D]
            dists = torch.cdist(z_q, self.prototypes, p=2) ** 2
            probs = torch.softmax(-dists / 0.1, dim=1).cpu().numpy().flatten()

        pred_id = int(np.argmax(probs))
        pred_label = DIRECTIONS[pred_id]
        
        confidence_dict = {DIRECTIONS[i]: float(probs[i]) for i in range(len(DIRECTIONS))}
        return pred_label, confidence_dict

    def predict_from_csv(self, csv_file_path):
        """
        Loads and predicts directly from a CSV trial file.
        """
        df = pd.read_csv(csv_file_path)
        channel_cols = [c for c in df.columns if c != 'time_ms']
        data = df[channel_cols].values.T.astype(np.float32) # [18, T]
        return self.predict_trial(data)
