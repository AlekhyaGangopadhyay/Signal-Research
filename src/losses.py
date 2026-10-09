"""
Loss Functions for Metric Meta-Learning and Directional Topology Enforcement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalLoss(nn.Module):
    """
    Computes Prototypical Loss based on Euclidean distance to class prototypes.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_query, prototypes, query_y):
        """
        z_query: [N_query, Latent_Dim]
        prototypes: [N_way, Latent_Dim]
        query_y: [N_query] (integer class indices from 0 to N_way-1)
        """
        # Compute squared Euclidean distance between every query and prototype
        # dists[i, k] = || z_query[i] - prototypes[k] ||^2
        dists = torch.cdist(z_query, prototypes, p=2) ** 2
        logits = -dists / self.temperature
        loss = F.cross_entropy(logits, query_y)
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == query_y).float().mean()
        return loss, acc, logits

class DirectionalAngularMarginLoss(nn.Module):
    """
    Novel Directional Angular Loss:
    Enforces geometric orthogonality among orthogonal directions in latent space:
    - Left (0) and Forward (2) are orthogonal -> cosine sim = 0
    - Right (1) and Forward (2) are orthogonal -> cosine sim = 0
    - Left (0) and Right (1) are opposite -> cosine sim = -1 (or near -0.8)
    """
    def __init__(self, lambda_angular=0.1):
        super().__init__()
        self.lambda_angular = lambda_angular

    def forward(self, prototypes):
        """
        prototypes: shape [4, D] (Classes: 0: Left, 1: Right, 2: Forward, 3: Backward)
        """
        if prototypes.shape[0] < 4:
            return torch.tensor(0.0, device=prototypes.device)

        # Normalized prototypes
        p = F.normalize(prototypes, p=2, dim=-1)
        # Cosine similarity matrix [4, 4]
        sim_mat = torch.mm(p, p.t())

        # Target similarity matrix based on true 2D direction angles:
        # Angles: Right: 0°, Forward: 90°, Left: 180°, Backward: 270°
        # Target Cosine: cos(θ_i - θ_j)
        # Class 0: Left (180°), Class 1: Right (0°), Class 2: Forward (90°), Class 3: Backward (270°)
        target_sim = torch.tensor([
            [ 1.0, -1.0,  0.0,  0.0], # Left vs [Left, Right, Fwd, Bwd]
            [-1.0,  1.0,  0.0,  0.0], # Right
            [ 0.0,  0.0,  1.0, -1.0], # Forward
            [ 0.0,  0.0, -1.0,  1.0]  # Backward
        ], device=prototypes.device, dtype=prototypes.dtype)

        angular_penalty = F.mse_loss(sim_mat, target_sim)
        return self.lambda_angular * angular_penalty
