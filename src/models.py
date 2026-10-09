"""
Model Architectures for Few-Shot EEG Spatial-Temporal Direction Classification.
Includes:
1. SpatialTemporalProtoNet (Multi-scale temporal convolutions + Spatial depthwise conv + Metric head)
2. EEGNetBackbone (Compact baseline)
3. RiemannianMDMClassifier (Riemannian geometry baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialTemporalProtoNet(nn.Module):
    """
    Proposed Spatial-Temporal Prototypical Backbone:
    - Branch 1: Temporal Conv capturing multi-frequency oscillations (Theta/Alpha/Beta/Gamma).
    - Branch 2: Spatial Depthwise Conv modeling bipolar differential electrode pairs.
    - Branch 3: Temporal Separable Attention & Pooling.
    - Branch 4: L2-Normalized Latent Embedding Projection.
    """
    def __init__(self, num_channels=18, latent_dim=128, dropout_rate=0.25):
        super().__init__()
        
        # 1. Multi-scale temporal filter bank
        self.conv_temporal_slow = nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.conv_temporal_fast = nn.Conv2d(1, 16, kernel_size=(1, 32), padding=(0, 16), bias=False)
        self.bn_temporal = nn.BatchNorm2d(32)

        # 2. Spatial depthwise convolution over 18 bipolar channels
        self.conv_spatial = nn.Conv2d(32, 64, kernel_size=(num_channels, 1), bias=False)
        self.bn_spatial = nn.BatchNorm2d(64)

        # 3. Separable Temporal & Subsampling
        self.conv_separable = nn.Conv2d(64, 64, kernel_size=(1, 16), padding=(0, 8), bias=False)
        self.bn_separable = nn.BatchNorm2d(64)

        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout = nn.Dropout(dropout_rate)

        # 4. Global Average Pooling & Metric Embedding
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x):
        """
        Input x: [Batch, 1, Channels=18, Time=500] or [Batch, Channels=18, Time=500]
        Output z: [Batch, Latent_Dim] (L2-normalized)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1) # [B, 1, C, T]

        # Multi-scale temporal feature maps
        t_slow = self.conv_temporal_slow(x)
        t_fast = self.conv_temporal_fast(x)
        
        # Match temporal dimensions if slight 1-sample difference due to padding
        min_w = min(t_slow.shape[-1], t_fast.shape[-1])
        t_cat = torch.cat([t_slow[..., :min_w], t_fast[..., :min_w]], dim=1)
        x = F.gelu(self.bn_temporal(t_cat))

        # Spatial depthwise filtering over bipolar montage
        x = F.gelu(self.bn_spatial(self.conv_spatial(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        # Separable temporal filtering
        x = F.gelu(self.bn_separable(self.conv_separable(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        # Latent projection
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        z = self.fc(x)

        # L2-sphere normalization for stable metric distance
        z = F.normalize(z, p=2, dim=-1)
        return z

class EEGNetBackbone(nn.Module):
    """
    Standard Compact EEGNet Architecture (Lawhern et al., 2018).
    """
    def __init__(self, num_channels=18, latent_dim=128, F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(F2, latent_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.separable(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        z = self.fc(x)
        return F.normalize(z, p=2, dim=-1)
