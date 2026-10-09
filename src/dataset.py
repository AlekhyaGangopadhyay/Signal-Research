"""
Dataset and Episodic Few-Shot Sampler for PyTorch Training.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import TENSORS_DIR, NUM_CLASSES

class EEGSubjectDataset(Dataset):
    """
    Loads preprocessed NPZ tensor files for individual or multiple subjects.
    """
    def __init__(self, npz_paths):
        self.X_list = []
        self.y_list = []
        
        for path in npz_paths:
            if os.path.exists(path):
                data = np.load(path)
                self.X_list.append(data['X']) # [N_trials, 18, 500]
                self.y_list.append(data['y']) # [N_trials]

        if self.X_list:
            self.X = np.concatenate(self.X_list, axis=0).astype(np.float32)
            self.y = np.concatenate(self.y_list, axis=0).astype(np.int64)
        else:
            self.X = np.empty((0, 18, 500), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.int64)

        # Index trials per class for fast episodic sampling
        self.class_indices = {c: [] for c in range(NUM_CLASSES)}
        for idx, lbl in enumerate(self.y):
            if lbl in self.class_indices:
                self.class_indices[lbl].append(idx)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

    def sample_fewshot_episode(self, n_way=4, k_shot=5, q_query=5):
        """
        Samples an episodic task:
        - Support set: n_way * k_shot samples
        - Query set: n_way * q_query samples
        """
        support_x, support_y = [], []
        query_x, query_y = [], []

        for c in range(n_way):
            available = self.class_indices[c]
            if len(available) == 0:
                continue

            # If not enough samples in class, sample with replacement
            replace = len(available) < (k_shot + q_query)
            selected = np.random.choice(available, size=k_shot + q_query, replace=replace)

            s_indices = selected[:k_shot]
            q_indices = selected[k_shot:k_shot + q_query]

            support_x.append(self.X[s_indices])
            support_y.append(np.full(k_shot, c))

            query_x.append(self.X[q_indices])
            query_y.append(np.full(q_query, c))

        support_x = torch.tensor(np.concatenate(support_x), dtype=torch.float32)
        support_y = torch.tensor(np.concatenate(support_y), dtype=torch.long)
        query_x = torch.tensor(np.concatenate(query_x), dtype=torch.float32)
        query_y = torch.tensor(np.concatenate(query_y), dtype=torch.long)

        return support_x, support_y, query_x, query_y
