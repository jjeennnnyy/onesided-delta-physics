"""Known-physics branches for vorticity time stepping."""

from __future__ import annotations

import torch


def spectral_laplacian(field: torch.Tensor) -> torch.Tensor:
    """Apply the Laplacian on a unit periodic square in Fourier space."""
    height, width = field.shape[-2:]
    ky = 2.0 * torch.pi * torch.fft.fftfreq(
        height, d=1.0 / height, device=field.device, dtype=field.dtype
    )
    kx = 2.0 * torch.pi * torch.fft.fftfreq(
        width, d=1.0 / width, device=field.device, dtype=field.dtype
    )
    multiplier = -(ky[:, None].square() + kx[None, :].square())
    spectrum = torch.fft.fft2(field, dim=(-2, -1), norm="ortho")
    return torch.fft.ifft2(
        spectrum * multiplier, dim=(-2, -1), norm="ortho"
    ).real


def diffusion_semigroup(
    vorticity: torch.Tensor,
    viscosity: float,
    delta_t: float,
) -> torch.Tensor:
    """Apply exp(nu * Laplacian * dt) on a unit periodic square."""
    if viscosity < 0 or delta_t < 0:
        raise ValueError("viscosity and delta_t must be non-negative")
    height, width = vorticity.shape[-2:]
    ky = 2.0 * torch.pi * torch.fft.fftfreq(
        height, d=1.0 / height, device=vorticity.device, dtype=vorticity.dtype
    )
    kx = 2.0 * torch.pi * torch.fft.fftfreq(
        width, d=1.0 / width, device=vorticity.device, dtype=vorticity.dtype
    )
    decay = torch.exp(
        -viscosity * delta_t * (ky[:, None].square() + kx[None, :].square())
    )
    spectrum = torch.fft.fft2(vorticity, dim=(-2, -1), norm="ortho")
    return torch.fft.ifft2(spectrum * decay, dim=(-2, -1), norm="ortho").real


def relative_diffusion_correction(
    state: torch.Tensor,
    viscosity: float,
    reference_viscosity: float,
    delta_t: float,
) -> torch.Tensor:
    """Additive change in diffusion propagation relative to training viscosity."""
    return diffusion_semigroup(state, viscosity, delta_t) - diffusion_semigroup(
        state, reference_viscosity, delta_t
    )


def diffusion_semigroup_1d(
    state: torch.Tensor,
    viscosity: float,
    delta_t: float,
) -> torch.Tensor:
    """Apply exp(nu * d_xx * dt) on a unit periodic interval."""
    if viscosity < 0 or delta_t < 0:
        raise ValueError("viscosity and delta_t must be non-negative")
    length = state.shape[-1]
    wave_number = 2.0 * torch.pi * torch.fft.rfftfreq(
        length, d=1.0 / length, device=state.device, dtype=state.dtype
    )
    decay = torch.exp(-viscosity * delta_t * wave_number.square())
    spectrum = torch.fft.rfft(state, dim=-1, norm="ortho")
    return torch.fft.irfft(spectrum * decay, n=length, dim=-1, norm="ortho")


def relative_diffusion_correction_1d(
    state: torch.Tensor,
    viscosity: float,
    reference_viscosity: float,
    delta_t: float,
) -> torch.Tensor:
    return diffusion_semigroup_1d(state, viscosity, delta_t) - diffusion_semigroup_1d(
        state, reference_viscosity, delta_t
    )
