"""Factorized Fourier neural operator backbone.

The backbone follows FourierFlow's ``FNOFactorized2DBlock``: independent
one-dimensional Fourier operators along x/y, a backcast feed-forward block,
and residual stacking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def _linear(in_features: int, out_features: int, use_weight_norm: bool) -> nn.Module:
    layer = nn.Linear(in_features, out_features)
    return weight_norm(layer) if use_weight_norm else layer


class FeedForward(nn.Module):
    """FourierFlow-style pointwise feed-forward block."""

    def __init__(
        self,
        width: int,
        factor: int = 4,
        n_layers: int = 2,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        layers = []
        for index in range(n_layers):
            in_dim = width if index == 0 else width * factor
            out_dim = width if index == n_layers - 1 else width * factor
            layers.append(_linear(in_dim, out_dim, use_weight_norm))
            if index < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FactorizedSpectralConv2d(nn.Module):
    """One F-FNO residual branch."""

    def __init__(
        self,
        width: int,
        modes: int,
        factor: int = 4,
        n_ff_layers: int = 2,
        use_weight_norm: bool = True,
        fourier_weight: nn.ParameterList | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        if fourier_weight is None:
            fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = nn.Parameter(torch.empty(width, width, modes, 2))
                nn.init.xavier_normal_(weight)
                fourier_weight.append(weight)
        self.fourier_weight = fourier_weight
        self.backcast_ff = FeedForward(
            width,
            factor=factor,
            n_layers=n_ff_layers,
            use_weight_norm=use_weight_norm,
        )

    def forward_fourier(self, x: torch.Tensor) -> torch.Tensor:
        # Channels-last [B,H,W,C] matches the official FourierFlow block.
        x_cf = x.permute(0, 3, 1, 2)
        batch, _, height, width = x_cf.shape

        y_ft = torch.fft.rfft(x_cf, dim=-1, norm="ortho")
        y_modes = min(self.modes, y_ft.shape[-1])
        y_raw = y_ft[:, :, :, :y_modes]
        y_out_ft = y_ft.new_zeros(batch, self.width, height, y_ft.shape[-1])
        y_out_ft[:, :, :, :y_modes] = torch.einsum(
            "bixy,ioy->boxy",
            y_raw,
            torch.view_as_complex(self.fourier_weight[0][:, :, :y_modes].contiguous()),
        )
        y_out = torch.fft.irfft(y_out_ft, n=width, dim=-1, norm="ortho")

        x_ft = torch.fft.rfft(x_cf, dim=-2, norm="ortho")
        x_modes = min(self.modes, x_ft.shape[-2])
        x_raw = x_ft[:, :, :x_modes, :]
        x_out_ft = x_ft.new_zeros(batch, self.width, x_ft.shape[-2], width)
        x_out_ft[:, :, :x_modes, :] = torch.einsum(
            "bixy,iox->boxy",
            x_raw,
            torch.view_as_complex(self.fourier_weight[1][:, :, :x_modes].contiguous()),
        )
        x_out = torch.fft.irfft(x_out_ft, n=height, dim=-2, norm="ortho")
        return (x_out + y_out).permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backcast_ff(self.forward_fourier(x))


class FFNO2d(nn.Module):
    """Plain F-FNO one-step operator.

    Input is ``[B,H,W,3]`` (normalized state plus coordinates), output is
    ``[B,H,W]``.  Fourier weights are shared across depth by default, matching
    the published 24-layer FourierFlow Navier--Stokes configuration.
    """

    def __init__(
        self,
        in_dim: int = 3,
        width: int = 64,
        modes: int = 16,
        depth: int = 24,
        factor: int = 4,
        n_ff_layers: int = 2,
        share_weight: bool = True,
        use_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_proj = _linear(in_dim, width, use_weight_norm)

        shared_weight = None
        if share_weight:
            shared_weight = nn.ParameterList([])
            for _ in range(2):
                weight = nn.Parameter(torch.empty(width, width, modes, 2))
                nn.init.xavier_normal_(weight, gain=0.1)
                shared_weight.append(weight)
        self.spectral_layers = nn.ModuleList(
            FactorizedSpectralConv2d(
                width=width,
                modes=modes,
                factor=factor,
                n_ff_layers=n_ff_layers,
                use_weight_norm=use_weight_norm,
                fourier_weight=shared_weight,
            )
            for _ in range(depth)
        )
        self.out = nn.Sequential(
            _linear(width, 128, use_weight_norm),
            _linear(128, 1, use_weight_norm),
        )

    def parameter_breakdown(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"backbone": total, "gate": 0, "adapter": 0, "total": total}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The common fno_dual data path supplies coordinates in [0, 1]; the
        # official FourierFlow Markov routine uses [-1, 1].
        if x.shape[-1] >= 3:
            x = torch.cat((x[..., :-2], 2.0 * x[..., -2:] - 1.0), dim=-1)
        x = self.in_proj(x)
        backcast = x
        for layer in self.spectral_layers:
            backcast = layer(x)
            x = x + backcast
        return self.out(backcast).squeeze(-1)
