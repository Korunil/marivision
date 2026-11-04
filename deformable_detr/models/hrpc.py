"""
models/hrpc.py
H0-1: Horizon-Relative Positional Conditioning (HRPC) & Parameter-Matched Abs-Y Control
- HRPC-v1: Modulates multi-scale 2D positional encodings with signed distance to horizon
- AbsYControl: Parameter-matched control using centered normalized vertical coordinate
- Symmetrical availability gating: both models scale modulation by g in {0, 1}
- Zero-initialized final conv guarantees exact R0 forward equivalence at step 0
  while ensuring nonzero gradient flow into W_final on the first backward pass.
"""

import torch
import torch.nn as nn


class HRPC(nn.Module):
    """
    Horizon-Relative Positional Conditioning (HRPC-v1)
    """
    def __init__(self, num_feature_levels=4, d_model=256, hidden_dim=64):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.d_model = d_model

        # Projection network: R^1 -> R^d_model (16,896 parameters)
        self.proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, d_model, kernel_size=1)
        )

        # Staged optimization: Zero-initialize final conv
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

        # Per-level scale gains (4 parameters)
        self.alpha = nn.Parameter(torch.ones(num_feature_levels))

    def build_distance_grid(self, eta_L, eta_R, H_l, W_l, device, dtype):
        B = eta_L.shape[0]
        y_coords = (torch.arange(H_l, device=device, dtype=dtype) + 0.5) / float(H_l)
        x_coords = (torch.arange(W_l, device=device, dtype=dtype) + 0.5) / float(W_l)

        h_x = (1.0 - x_coords.view(1, 1, W_l)) * eta_L.view(B, 1, 1) + x_coords.view(1, 1, W_l) * eta_R.view(B, 1, 1)
        y_grid = y_coords.view(1, H_l, 1)

        d_l = (y_grid - h_x).unsqueeze(1)  # [B, 1, H_l, W_l]
        return d_l

    def forward(self, pos_embeds, horizons):
        """
        pos_embeds: list of 4 tensors, each [B, C, H_l, W_l]
        horizons: [B, 3] tensor containing (eta_L, eta_R, g)
        """
        eta_L = horizons[:, 0]
        eta_R = horizons[:, 1]
        g = horizons[:, 2].view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        device = pos_embeds[0].device
        dtype = pos_embeds[0].dtype

        modulated_pos = []
        for l, p in enumerate(pos_embeds):
            B, C, H_l, W_l = p.shape
            d_l = self.build_distance_grid(eta_L, eta_R, H_l, W_l, device, dtype)
            phi_d = self.proj(d_l)
            p_mod = p + g * (self.alpha[l] * phi_d)
            modulated_pos.append(p_mod)

        return modulated_pos


class AbsYControl(nn.Module):
    """
    Parameter-Matched Absolute-Y Control (Coord-Control)
    Identical parameter count and availability gating, conditioning on normalized y.
    """
    def __init__(self, num_feature_levels=4, d_model=256, hidden_dim=64):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.d_model = d_model

        self.proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, d_model, kernel_size=1)
        )

        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

        self.alpha = nn.Parameter(torch.ones(num_feature_levels))

    def forward(self, pos_embeds, availability=None):
        """
        pos_embeds: list of 4 tensors, each [B, C, H_l, W_l]
        availability: Optional [B] or [B, 1] tensor containing g in {0, 1}
        """
        if availability is None:
            g = 1.0
        else:
            g = availability.view(-1, 1, 1, 1)

        device = pos_embeds[0].device
        dtype = pos_embeds[0].dtype

        modulated_pos = []
        for l, p in enumerate(pos_embeds):
            B, C, H_l, W_l = p.shape
            y_coords = (torch.arange(H_l, device=device, dtype=dtype) + 0.5) / float(H_l) - 0.5
            y_grid = y_coords.view(1, 1, H_l, 1).expand(B, 1, H_l, W_l)

            phi_y = self.proj(y_grid)
            p_mod = p + g * (self.alpha[l] * phi_y)
            modulated_pos.append(p_mod)

        return modulated_pos