import torch
import torch.nn as nn

from fno_dual.normalizer import UnitGaussianNormalizer
from fno_dual.physics import spectral_laplacian
from run_ffno_equation_recast_3seed import recast_predict_next


class ZeroIncrement(nn.Module):
    def forward(self, inputs):
        return inputs[..., 0] * 0.0


def _normalizer():
    return UnitGaussianNormalizer(torch.tensor([-1.0, 1.0]))


def test_spectral_laplacian_on_fourier_mode():
    n = 32
    x = torch.arange(n, dtype=torch.float64) / n
    field = torch.sin(2.0 * torch.pi * 3.0 * x)[None, :, None].expand(1, n, n)
    expected = -(2.0 * torch.pi * 3.0) ** 2 * field
    assert torch.allclose(spectral_laplacian(field), expected, atol=1e-9, rtol=1e-9)


def test_recast_is_exact_increment_at_reference_viscosity():
    state = torch.randn(2, 8, 8)
    normalizer = _normalizer()
    prediction, diagnostics = recast_predict_next(
        ZeroIncrement(), state, 1e-3, 1e-3, 0.05,
        normalizer, normalizer, tolerance=1e-5, max_iterations=30,
    )
    assert torch.equal(prediction, state)
    assert torch.equal(diagnostics["iterations"], torch.zeros(2, dtype=torch.int64))
    assert torch.equal(diagnostics["nn_evaluations"], torch.ones(2, dtype=torch.int64))
    assert diagnostics["converged"].all()


def test_recast_records_iteration_diagnostics():
    state = torch.randn(2, 8, 8)
    normalizer = _normalizer()
    prediction, diagnostics = recast_predict_next(
        ZeroIncrement(), state, 2e-3, 1e-3, 0.05,
        normalizer, normalizer, tolerance=1e-5, max_iterations=3,
    )
    assert prediction.shape == state.shape
    assert (diagnostics["nn_evaluations"] >= 2).all()
    assert (diagnostics["iterations"] >= 1).all()
