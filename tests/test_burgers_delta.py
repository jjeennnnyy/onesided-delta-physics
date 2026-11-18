import torch

from fno_dual.data import generate_burgers_trajectories
from fno_dual.fno1d import FNO1d
from fno_dual.physics import diffusion_semigroup_1d, relative_diffusion_correction_1d


def test_burgers_trajectory_shape_and_finite_values():
    trajectories = generate_burgers_trajectories(
        n_trajectories=2, n_grid=24, n_steps=3, seed=4
    )
    assert trajectories.shape == (2, 4, 24)
    assert torch.isfinite(trajectories).all()


def test_1d_diffusion_single_mode_and_zero_relative_correction():
    n, mode = 32, 3
    x = torch.arange(n, dtype=torch.float32) / n
    state = torch.sin(2.0 * torch.pi * mode * x)[None]
    viscosity, delta_t = 0.01, 0.02
    expected = state * torch.exp(
        torch.tensor(-viscosity * delta_t * (2.0 * torch.pi * mode) ** 2)
    )
    actual = diffusion_semigroup_1d(state, viscosity, delta_t)
    assert torch.allclose(actual, expected, atol=2e-6)
    correction = relative_diffusion_correction_1d(
        state, viscosity, viscosity, delta_t
    )
    assert torch.count_nonzero(correction) == 0


def test_clean_fno1d_forward_backward():
    model = FNO1d(width=8, modes=6, depth=2)
    output = model(torch.randn(3, 32, 2))
    assert output.shape == (3, 32)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
