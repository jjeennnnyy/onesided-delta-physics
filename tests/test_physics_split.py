import torch

from fno_dual.physics import diffusion_semigroup, relative_diffusion_correction


def test_diffusion_semigroup_preserves_constant_mode():
    state = torch.full((2, 16, 16), 3.25)
    propagated = diffusion_semigroup(state, viscosity=1e-3, delta_t=0.05)
    assert torch.allclose(propagated, state, atol=1e-6)


def test_diffusion_semigroup_has_expected_single_mode_attenuation():
    n = 32
    x = torch.arange(n, dtype=torch.float32) / n
    state = torch.sin(2.0 * torch.pi * 3.0 * x)[None, :, None].expand(1, n, n)
    viscosity, delta_t = 2e-3, 0.05
    expected = state * torch.exp(torch.tensor(-viscosity * delta_t * (2.0 * torch.pi * 3.0) ** 2))
    actual = diffusion_semigroup(state, viscosity, delta_t)
    assert torch.allclose(actual, expected, atol=2e-6)


def test_relative_diffusion_is_exactly_zero_at_reference_viscosity():
    state = torch.randn(2, 12, 12)
    correction = relative_diffusion_correction(state, 1e-3, 1e-3, 0.05)
    assert torch.count_nonzero(correction) == 0


def test_relative_diffusion_matches_difference_of_propagators():
    state = torch.randn(2, 12, 12)
    correction = relative_diffusion_correction(state, 2e-3, 1e-3, 0.05)
    expected = diffusion_semigroup(state, 2e-3, 0.05) - diffusion_semigroup(state, 1e-3, 0.05)
    assert torch.equal(correction, expected)
