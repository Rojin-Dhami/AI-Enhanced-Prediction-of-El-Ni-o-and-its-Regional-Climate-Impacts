"""
Dual-Encoder CNN-TCN hybrid model for ENSO forecasting + South Asian climate impact.

UPDATED ARCHITECTURE with separate encoders for Pacific and South Asia:
  
  Stage 1a: Pacific Spatial Encoder (CNN) - 2D conv on tropical Pacific data
    Input per month: (B, 6, 31, 81) -> SST, MSL, wind E-W, wind N-S, OHC, thermocline
    Output per month: (B, d_pacific)
    
  Stage 1b: South Asia Spatial Encoder (CNN) - 2D conv on South Asia data  
    Input per month: (B, 5, 18, 21) -> t2m, tp (precip), MSL, wind E-W, wind N-S
    Output per month: (B, d_sa)
    
  Stage 2: Temporal Encoder (TCN) - processes merged spatial features
    Input: (B, 12, d_pacific + d_sa) -> concatenated features across time
    Output: (B, h) -> encoded representation z
    
  Stage 3: Two task-specific heads
    ENSO head: z -> scalar Niño 3.4 index at 6-month lead
    Impact head: z -> (2, G) South Asia precip + temp anomaly map

This dual-encoder approach captures both:
  - Large-scale ENSO teleconnection patterns (Pacific encoder)
  - Regional South Asian atmospheric state (SA encoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stage 1a: Pacific Spatial Encoder (CNN)
# ---------------------------------------------------------------------------
class PacificSpatialEncoder(nn.Module):
    """
    2D CNN for tropical Pacific (31×81 grid, 6 channels).
    Applied independently to each monthly snapshot with shared weights.
    
    Input per snapshot:  (B, C=6, H=31, W=81)
      - SST anomaly
      - MSL anomaly  
      - Wind stress E-W
      - Wind stress N-S
      - Ocean heat content (0-300m)
      - Thermocline depth (20°C isotherm)
    Output per snapshot: (B, d)
    """

    def __init__(self, in_channels: int = 6, d: int = 64, dropout: float = 0.2):
        super().__init__()
        self.d = d

        self.block1 = self._conv_block(in_channels, 16, dropout)
        self.block2 = self._conv_block(16, 32, dropout)
        self.block3 = self._conv_block(32, 64, dropout)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, d)

    @staticmethod
    def _conv_block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=6, H=31, W=81)
        x = self.block1(x)   # -> (B, 16, 15, 50)
        x = self.block2(x)   # -> (B, 32, 7, 25)
        x = self.block3(x)   # -> (B, 64, 3, 12)
        x = self.global_pool(x).flatten(1)   # -> (B, 64)
        return self.proj(x)                   # -> (B, d)


# ---------------------------------------------------------------------------
# Stage 1b: South Asia Spatial Encoder (CNN)
# ---------------------------------------------------------------------------
class SouthAsiaSpatialEncoder(nn.Module):
    """
    2D CNN for South Asia region (18×21 grid, 5 channels).
    Applied independently to each monthly snapshot with shared weights.
    
    Input per snapshot:  (B, C=5, H=18, W=21)
      - t2m (temperature) anomaly
      - tp (precipitation) anomaly
      - MSL anomaly
      - Wind stress E-W
      - Wind stress N-S
    Output per snapshot: (B, d)
    """

    def __init__(self, in_channels: int = 5, d: int = 48, dropout: float = 0.2):
        super().__init__()
        self.d = d

        # Smaller spatial domain -> shallower network
        self.block1 = self._conv_block(in_channels, 16, dropout)
        self.block2 = self._conv_block(16, 32, dropout)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(32, d)

    @staticmethod
    def _conv_block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=5, H=18, W=21)
        x = self.block1(x)   # -> (B, 16, 9, 10)
        x = self.block2(x)   # -> (B, 32, 4, 5)
        x = self.global_pool(x).flatten(1)   # -> (B, 32)
        return self.proj(x)                   # -> (B, d)


# ---------------------------------------------------------------------------
# Stage 2: Temporal Encoder (TCN) - unchanged from original
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
    One dilated causal residual block (Bai et al. 2018).
    Dilation: d_l = 2^(l-1)
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

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
    Output: (B, h)        -- single encoded representation z
    """

    def __init__(self, d_in: int, num_channels=(64, 64, 64, 64),
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        c_prev = d_in
        for l, c_out in enumerate(num_channels):
            dilation = 2 ** l
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
        z = out[:, :, -1]               # last time step = full-window summary
        return z                        # (B, h)


# ---------------------------------------------------------------------------
# Stage 3: Dual-Encoder CNN-TCN Model
# ---------------------------------------------------------------------------
class DualEncoderCNNTCN(nn.Module):
    """
    Full dual-encoder pipeline:
      Pacific input:  (B, T=12, H_pac=31, W_pac=81, C_pac=6)
      SA input:       (B, T=12, H_sa=18, W_sa=21, C_sa=5)
        --Pacific CNN--> (B, T, d_pac)
        --SA CNN-->      (B, T, d_sa)
        --concat-->      (B, T, d_pac + d_sa)
        --TCN-->         z in (B, h)
        --heads-->       y_enso in (B,), y_impact in (B, 2, G)
    """

    def __init__(self,
                 # Pacific encoder
                 pacific_channels: int = 6,
                 pacific_dim: int = 64,
                 # South Asia encoder
                 sa_channels: int = 5,
                 sa_dim: int = 48,
                 # TCN
                 tcn_channels=(64, 64, 64, 64),
                 tcn_kernel_size: int = 3,
                 # Output heads
                 num_impact_cells: int = 378,  # 18×21 SA grid
                 dropout: float = 0.2,
                 head_hidden: int = 64):
        super().__init__()
        
        self.pacific_encoder = PacificSpatialEncoder(
            pacific_channels, pacific_dim, dropout
        )
        self.sa_encoder = SouthAsiaSpatialEncoder(
            sa_channels, sa_dim, dropout
        )
        
        # TCN input = concatenated features
        merged_dim = pacific_dim + sa_dim
        self.temporal_encoder = TemporalEncoderTCN(
            merged_dim, tcn_channels, tcn_kernel_size, dropout
        )
        
        h = self.temporal_encoder.h
        self.G = num_impact_cells

        # ENSO head (scalar Niño 3.4 index)
        self.enso_head = nn.Sequential(
            nn.Linear(h, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        # Impact head (2, G) precip + temp anomalies
        self.impact_head = nn.Sequential(
            nn.Linear(h, head_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden * 2, 2 * self.G),
        )

    def forward(self, x_pacific: torch.Tensor, x_sa: torch.Tensor):
        """
        x_pacific: (B, T, H_pac, W_pac, C_pac)
        x_sa:      (B, T, H_sa, W_sa, C_sa)
        
        returns:
            y_enso:   (B,)        predicted Niño 3.4 index, 6-month lead
            y_impact: (B, 2, G)   predicted [precip_anom, temp_anom] per cell
        """
        B, T = x_pacific.shape[:2]

        # Process Pacific with shared CNN across time
        x_pac = x_pacific.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, C, H, W)
        x_pac = x_pac.view(B * T, *x_pac.shape[2:])            # (B*T, C, H, W)
        f_pac = self.pacific_encoder(x_pac)                     # (B*T, d_pac)
        f_pac = f_pac.view(B, T, -1)                            # (B, T, d_pac)

        # Process South Asia with shared CNN across time
        x_sa = x_sa.permute(0, 1, 4, 2, 3).contiguous()        # (B, T, C, H, W)
        x_sa = x_sa.view(B * T, *x_sa.shape[2:])               # (B*T, C, H, W)
        f_sa = self.sa_encoder(x_sa)                            # (B*T, d_sa)
        f_sa = f_sa.view(B, T, -1)                              # (B, T, d_sa)

        # Concatenate features
        f_merged = torch.cat([f_pac, f_sa], dim=-1)             # (B, T, d_pac+d_sa)

        # Temporal encoding
        z = self.temporal_encoder(f_merged)                     # (B, h)

        # Task-specific heads
        y_enso = self.enso_head(z).squeeze(-1)                  # (B,)
        y_impact = self.impact_head(z).view(B, 2, self.G)       # (B, 2, G)

        return y_enso, y_impact


# ---------------------------------------------------------------------------
# Multi-task loss (unchanged)
# ---------------------------------------------------------------------------
def multitask_loss(y_enso_pred, y_enso_true, y_impact_pred, y_impact_true,
                    alpha: float = 0.5):
    """
    L_total = alpha * L_ENSO + (1 - alpha) * L_impact
    """
    l_enso = F.mse_loss(y_enso_pred, y_enso_true)
    l_impact = F.mse_loss(y_impact_pred, y_impact_true)
    total = alpha * l_enso + (1 - alpha) * l_impact
    return total, l_enso.detach(), l_impact.detach()


if __name__ == "__main__":
    # Shape sanity check
    model = DualEncoderCNNTCN(
        pacific_channels=6,
        pacific_dim=64,
        sa_channels=5,
        sa_dim=48,
        num_impact_cells=378  # 18×21 SA grid
    )
    
    # Dummy inputs
    x_pacific = torch.randn(8, 12, 31, 81, 6)  # batch of 8, Pacific
    x_sa = torch.randn(8, 12, 18, 21, 5)        # batch of 8, South Asia
    
    y_enso, y_impact = model(x_pacific, x_sa)
    
    print("Dual-Encoder CNN-TCN Model")
    print("="*60)
    print(f"Pacific input:  {x_pacific.shape}")
    print(f"SA input:       {x_sa.shape}")
    print(f"ENSO output:    {y_enso.shape}")
    print(f"Impact output:  {y_impact.shape}")
    print("="*60)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")
