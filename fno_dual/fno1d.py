"""Clean one-dimensional Fourier neural operator backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = width
        self.modes = modes
        scale = 1.0 / (width * width)
        self.weight = nn.Parameter(
            scale * torch.randn(width, width, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        spectrum = torch.fft.rfft(x, dim=-1, norm="ortho")
        modes = min(self.modes, spectrum.shape[-1])
        output = spectrum.new_zeros(x.shape[0], self.width, spectrum.shape[-1])
        output[:, :, :modes] = torch.einsum(
            "bim,iom->bom", spectrum[:, :, :modes], self.weight[:, :, :modes]
        )
        return torch.fft.irfft(output, n=length, dim=-1, norm="ortho")


class FNO1d(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        width: int = 32,
        modes: int = 16,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.spectral_layers = nn.ModuleList(
            SpectralConv1d(width, modes) for _ in range(depth)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv1d(width, width, 1) for _ in range(depth)
        )
        self.out = nn.Sequential(
            nn.Linear(width, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(inputs).permute(0, 2, 1)
        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            x = F.gelu(spectral(x) + pointwise(x))
        return self.out(x.permute(0, 2, 1)).squeeze(-1)

