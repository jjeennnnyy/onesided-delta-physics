import torch

from fno_dual.models import FNO2d


def _count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def test_shared_adapter_forward_backward_and_shapes():
    model = FNO2d(
        width=8,
        modes1=4,
        modes2=4,
        depth=2,
        gate_variant="data_model",
        gate_mode="dual",
        use_adapter=True,
        n_bands=4,
        adapter_rank=2,
    )
    output = model(torch.randn(3, 12, 12, 3))
    assert output.shape == (3, 12, 12)
    output.square().mean().backward()
    assert model.spec_layers[0].adapter1_b_real.grad is not None
    assert model.spec_layers[0].adapter1_a_real.shape == (4, 2, 8)
    assert model.spec_layers[0].adapter1_b_real.shape == (4, 8, 2)


def test_default_shared_adapter_parameter_overhead_is_below_five_percent():
    fno = FNO2d(width=32, modes1=12, modes2=12, depth=4)
    shared = FNO2d(
        width=32,
        modes1=12,
        modes2=12,
        depth=4,
        gate_variant="data_model",
        gate_mode="dual",
        use_adapter=True,
        n_bands=4,
        adapter_rank=4,
    )
    assert _count(shared) / _count(fno) <= 1.05
    assert shared.parameter_breakdown()["adapter"] == 16384


def test_dual_gate_has_only_scalar_fusion_parameters():
    model = FNO2d(
        width=8,
        modes1=4,
        modes2=4,
        depth=2,
        gate_variant="data_model",
        gate_mode="dual",
        use_adapter=True,
    )
    for gate in model.gates:
        assert gate.raw_w_data.shape == torch.Size([])
        assert gate.raw_w_model.shape == torch.Size([])
        assert gate.bias.shape == torch.Size([])
        assert not hasattr(gate, "adapter_route")
        assert not hasattr(gate, "route_scale")
