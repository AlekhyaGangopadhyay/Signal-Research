"""
Loss Functions for Metric Meta-Learning and Directional Topology Enforcement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_prototypes(z_support, support_y, n_way):
    """Class means of support embeddings, using the labels (no assumption about ordering)."""
    return torch.stack([z_support[support_y == c].mean(dim=0) for c in range(n_way)])


class PrototypicalLoss(nn.Module):
    """
    Computes Prototypical Loss based on squared Euclidean distance to class prototypes.
    With L2-normalised embeddings squared distances lie in [0, 4], so temperature ~1 keeps the
    softmax out of saturation (0.1 gave logits up to 40).
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_query, prototypes, query_y):
        """
        z_query: [N_query, Latent_Dim]
        prototypes: [N_way, Latent_Dim]
        query_y: [N_query] (integer class indices from 0 to N_way-1)
        """
        dists = torch.cdist(z_query, prototypes, p=2) ** 2
        logits = -dists / self.temperature
        loss = F.cross_entropy(logits, query_y)

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == query_y).float().mean()
        return loss, acc, logits


class DirectionalAngularMarginLoss(nn.Module):
    """
    Directional angular regulariser on the class prototypes. Target cosine similarities follow the
    2-D compass angles (Right 0°, Forward 90°, Left 180°, Backward 270°): cos(θ_i - θ_j), so
      Left/Right and Forward/Backward are opposite (-1), every other pair is orthogonal (0).

    This is an untested inductive bias (nothing shows the brain codes arrows/letters/words this way).
    It is off by default (config.USE_ANGULAR_LOSS) and must be justified by an ablation.
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

        p = F.normalize(prototypes, p=2, dim=-1)
        sim_mat = torch.mm(p, p.t())

        target_sim = torch.tensor([
            [ 1.0, -1.0,  0.0,  0.0],  # Left vs [Left, Right, Fwd, Bwd]
            [-1.0,  1.0,  0.0,  0.0],  # Right
            [ 0.0,  0.0,  1.0, -1.0],  # Forward
            [ 0.0,  0.0, -1.0,  1.0]   # Backward
        ], device=prototypes.device, dtype=prototypes.dtype)

        return self.lambda_angular * F.mse_loss(sim_mat, target_sim)
