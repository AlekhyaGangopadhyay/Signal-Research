"""
Model Architectures for Few-Shot EEG Spatial-Temporal Direction Classification.
Includes:
1. SpatialTemporalProtoNet (multi-scale temporal convolutions + depthwise spatial conv + metric head)
2. EEGNetBackbone (compact baseline, Lawhern et al., 2018)
3. adapt_batchnorm: re-estimates BatchNorm statistics on unlabelled target-subject data (AdaBN)

Riemannian baselines live in src/baselines.py (pyriemann).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTemporalProtoNet(nn.Module):
    """
    Spatial-Temporal Prototypical Backbone (EEGNet-style, multi-scale):
    - Temporal filter bank with three kernel lengths (0.5 s, 0.25 s, 0.125 s at 256 Hz).
    - Depthwise spatial convolution (groups = temporal filters): each temporal filter gets
      `depth` spatial patterns over the 18 bipolar channels.
    - Separable temporal convolution + pooling.
    - Linear projection to an L2-normalised embedding.
    """

    def __init__(self, num_channels=18, latent_dim=64, dropout_rate=0.5, f1_per_scale=8, depth=2, f2=32,
                 kernels=(128, 64, 32)):
        super().__init__()
        self.temporal = nn.ModuleList([
            nn.Conv2d(1, f1_per_scale, kernel_size=(1, k), padding=(0, k // 2), bias=False) for k in kernels
        ])
        f1 = f1_per_scale * len(kernels)
        self.bn_temporal = nn.BatchNorm2d(f1)

        self.conv_spatial = nn.Conv2d(f1, f1 * depth, kernel_size=(num_channels, 1), groups=f1, bias=False)
        self.bn_spatial = nn.BatchNorm2d(f1 * depth)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv_depthwise_t = nn.Conv2d(f1 * depth, f1 * depth, kernel_size=(1, 16), padding=(0, 8),
                                          groups=f1 * depth, bias=False)
        self.conv_pointwise = nn.Conv2d(f1 * depth, f2, kernel_size=1, bias=False)
        self.bn_separable = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout = nn.Dropout(dropout_rate)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(f2, latent_dim)

    def forward(self, x):
        """
        Input x: [Batch, Channels=18, Time] or [Batch, 1, Channels, Time]
        Output z: [Batch, Latent_Dim] (L2-normalised)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        T = x.shape[-1]
        x = torch.cat([conv(x)[..., :T] for conv in self.temporal], dim=1)
        x = self.bn_temporal(x)

        x = F.elu(self.bn_spatial(self.conv_spatial(x)))
        x = self.dropout(self.pool1(x))

        x = F.elu(self.bn_separable(self.conv_pointwise(self.conv_depthwise_t(x))))
        x = self.dropout(self.pool2(x))

        x = torch.flatten(self.adaptive_pool(x), 1)
        return F.normalize(self.fc(x), p=2, dim=-1)


class EEGNetBackbone(nn.Module):
    """
    Standard Compact EEGNet Architecture (Lawhern et al., 2018).
    """
    def __init__(self, num_channels=18, latent_dim=64, F1=8, D=2, F2=16, kernel_length=128, dropout_rate=0.5):
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


def build_model(model_type, num_channels=18, latent_dim=64, dropout_rate=0.5):
    if model_type == "spatial_temporal":
        return SpatialTemporalProtoNet(num_channels=num_channels, latent_dim=latent_dim, dropout_rate=dropout_rate)
    if model_type == "eegnet":
        return EEGNetBackbone(num_channels=num_channels, latent_dim=latent_dim, dropout_rate=dropout_rate)
    raise ValueError(f"unknown model_type {model_type!r}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def adapt_batchnorm(model, X, batch_size=128, device="cpu"):
    """
    AdaBN: re-estimate BatchNorm running statistics on unlabelled data from the target subject, so the
    network is not normalised with statistics from the training subjects. Leaves the model in eval mode.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return model
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None          # cumulative average over all batches
    model.train()
    for m in model.modules():       # keep dropout off while collecting statistics
        if isinstance(m, nn.Dropout):
            m.eval()
    for i in range(0, len(X), batch_size):
        model(torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device))
    model.eval()
    return model
