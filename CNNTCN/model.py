accuracy matter"""
CNN-TCN hybrid model for ENSO forecasting + South Asian climate impact assessment.

Implements the architecture described in Section 4.7 of the report:
  - Stage 1: 2D CNN spatial encoder, applied identically to each of the 12
    monthly snapshots (shared weights across time -> Eq. 4-15).
  - Stage 2: Temporal Convolutional Network (TCN) with dilated causal
    convolutions over the 12-step sequence of spatial feature vectors
    (Bai et al. 2018, ref [8] in your bibliography) -> Eq. 4-16.
  - Stage 3: two task-specific heads sharing the encoded representation z:
      ENSO head    -> scalar Nino 3.4 index at 6-month lead (Eq. 4-17)
      Impact head  -> (2, G) South Asia precip + temp anomaly map (Eq. 4-18)

Input tensor shape per sample: (T=12, H=30, W=100, C=4)
  T = months in sliding window
  H = 30 latitude cells, W = 100 longitude cells (tropical Pacific, 2x2 deg)
  C = 4 channels: sst_anom, msl_anom, avg_iews_anom, sohtc300_anom
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stage 1: Spatial Encoder (CNN)
# ---------------------------------------------------------------------------
class SpatialEncoderCNN(nn.Module):
    """
    2D CNN applied independently (shared weights) to each monthly snapshot.
    Input per snapshot:  (B, C=4, H=30, W=100)
    Output per snapshot: (B, d)  -- a compact spatial feature vector f(t)

    Kept intentionally shallow: with only ~450 training samples (Section
    5.7.2: 542 total windows), a deep CNN will overfit badly. Three conv
    blocks + global average pooling keeps the parameter count sane.
    """

    def __init__(self, in_channels: int = 4, d: int = 64, dropout: float = 0.2):
        super().__init__()
        self.d = d

        self.block1 = self._conv_block(in_channels, 16, dropout)
        self.block2 = self._conv_block(16, 32, dropout)
        self.block3 = self._conv_block(32, 64, dropout)

        # Global average pool -> fixed-size vector regardless of exact
        # spatial dims after strided convs (robust if you crop the Pacific
        # box slightly differently later).
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, d)

    @staticmethod
    def _conv_block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # halves H, W each block
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.block1(x)   # -> (B, 16, 15, 50)
        x = self.block2(x)   # -> (B, 32, 7, 25)
        x = self.block3(x)   # -> (B, 64, 3, 12)
        x = self.global_pool(x).flatten(1)   # -> (B, 64)
        return self.proj(x)                   # -> (B, d)


# ---------------------------------------------------------------------------
# Stage 2: Temporal Encoder (TCN)
# ---------------------------------------------------------------------------
class Chomp1d(nn.Module):
    """Removes the extra right-padding added for causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    One dilated causal residual block (Bai et al. 2018 style):
    Conv1d -> Chomp -> ReLU -> Dropout, x2, with a residual connection.
    Dilation follows Eq. 4-16: d_l = 2^(l-1).
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal: pad left only (via chomp)

        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size,
                                padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size,
                                padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        # 1x1 conv to match channel dims for the residual add, if needed
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu_out = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TemporalEncoderTCN(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilation.
    Input:  (B, T=12, d)  -- sequence of per-month spatial features
    Output: (B, h)        -- single encoded representation z (Eq. output)

    With T=12 and kernel_size=3, 4 layers gives a receptive field of
    1 + 2*(3-1)*(1+2+4+8) = 61 >> 12, i.e. the last time step already
    "sees" the entire 12-month window -- exactly what we want for z.
    """

    def __init__(self, d_in: int, num_channels=(64, 64, 64, 64),
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        c_prev = d_in
        for l, c_out in enumerate(num_channels):
            dilation = 2 ** l   # Eq. 4-16: d_l = 2^(l-1), l starting at 1
            layers.append(
                TemporalBlock(c_prev, c_out, kernel_size,
                               dilation=dilation, dropout=dropout)
            )
            c_prev = c_out
        self.network = nn.Sequential(*layers)
        self.h = c_prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d) -> conv1d wants (B, d, T)
        x = x.transpose(1, 2)
        out = self.network(x)          # (B, h, T)
        z = out[:, :, -1]               # last (causal) time step = full-window summary
        return z                        # (B, h)


# ---------------------------------------------------------------------------
# Stage 3: Multi-task CNN-TCN model
# ---------------------------------------------------------------------------
class CNNTCN(nn.Module):
    """
    Full pipeline (Eq. 4-20):
      (B, T=12, H=30, W=100, C=4)
        --CNN(shared across T)--> (B, T, d)
        --TCN-->                  z in (B, h)
        --heads-->                 y_enso in (B,), y_impact in (B, 2, G)
    """

    def __init__(self,
                 in_channels: int = 4,
                 spatial_dim: int = 64,        # d
                 tcn_channels=(64, 64, 64, 64),
                 tcn_kernel_size: int = 3,
                 num_impact_cells: int = 340,  # G, South Asia grid cells
                 dropout: float = 0.2,
                 head_hidden: int = 64):
        super().__init__()
        self.spatial_encoder = SpatialEncoderCNN(in_channels, spatial_dim, dropout)
        self.temporal_encoder = TemporalEncoderTCN(
            spatial_dim, tcn_channels, tcn_kernel_size, dropout
        )
        h = self.temporal_encoder.h
        self.G = num_impact_cells

        # ENSO head (Eq. 4-17): scalar Nino 3.4 index
        self.enso_head = nn.Sequential(
            nn.Linear(h, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        # Impact head (Eq. 4-18): (2, G) precip + temp anomaly map
        self.impact_head = nn.Sequential(
            nn.Linear(h, head_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden * 2, 2 * self.G),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, H, W, C)  raw input tensor as constructed in Section 5.7.2
        returns:
            y_enso:   (B,)        predicted Nino 3.4 index, 6-month lead
            y_impact: (B, 2, G)   predicted [precip_anom, temp_anom] per cell
        """
        B, T, H, W, C = x.shape

        # Apply CNN with SHARED weights across all T time steps.
        # Reshape to (B*T, C, H, W) so a single conv pass handles everything.
        x = x.permute(0, 1, 4, 2, 3).contiguous()      # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)                     # (B*T, C, H, W)
        f = self.spatial_encoder(x)                     # (B*T, d)
        f = f.view(B, T, -1)                             # (B, T, d)

        z = self.temporal_encoder(f)                     # (B, h)

        y_enso = self.enso_head(z).squeeze(-1)            # (B,)
        y_impact = self.impact_head(z).view(B, 2, self.G)  # (B, 2, G)

        return y_enso, y_impact


# ---------------------------------------------------------------------------
# Multi-task loss (Eq. 4-19)
# ---------------------------------------------------------------------------
def multitask_loss(y_enso_pred, y_enso_true, y_impact_pred, y_impact_true,
                    alpha: float = 0.5):
    """
    L_total = alpha * L_ENSO + (1 - alpha) * L_impact
    alpha is tuned on the validation set (Section 4.7).
    """
    l_enso = F.mse_loss(y_enso_pred, y_enso_true)
    l_impact = F.mse_loss(y_impact_pred, y_impact_true)
    total = alpha * l_enso + (1 - alpha) * l_impact
    return total, l_enso.detach(), l_impact.detach()


if __name__ == "__main__":
    # Quick shape sanity check matching your spec: (12, 30, 100, 4) -> G=340
    model = CNNTCN(num_impact_cells=340)
    dummy_x = torch.randn(8, 12, 30, 100, 4)   # batch of 8
    y_enso, y_impact = model(dummy_x)
    print("y_enso:", y_enso.shape)      # torch.Size([8])
    print("y_impact:", y_impact.shape)  # torch.Size([8, 2, 340])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")
