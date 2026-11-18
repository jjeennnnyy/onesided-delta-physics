import torch

from fno_dual.ffno import FFNO2d


def test_ffno_forward_backward_and_parameter_count():
    model = FFNO2d(width=16, modes=8, depth=8, factor=4)
    output = model(torch.randn(2, 24, 24, 3))
    assert output.shape == (2, 24, 24)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) == 28370
