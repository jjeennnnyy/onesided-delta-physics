import torch
import torch.nn as nn
import torch.nn.functional as F

from .gates import DualDrivenSpectralGate1d, DualDrivenSpectralGate2d


class SpectralConv1d(nn.Module):
    def __init__(self, channels, modes, gate=None, use_adapter=False, adapter_rank=4):
        super().__init__()
        self.channels, self.modes, self.gate, self.use_adapter = channels, modes, gate, use_adapter
        scale = 1.0 / (channels * channels)
        self.weight_real = nn.Parameter(scale * torch.randn(channels, channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(channels, channels, modes))
        if use_adapter:
            adapter_scale = 1.0 / max(1, channels * adapter_rank)
            self.adapter_a_real = nn.Parameter(adapter_scale * torch.randn(adapter_rank, channels, modes))
            self.adapter_a_imag = nn.Parameter(adapter_scale * torch.randn(adapter_rank, channels, modes))
            self.adapter_b_real = nn.Parameter(torch.zeros(channels, adapter_rank, modes))
            self.adapter_b_imag = nn.Parameter(torch.zeros(channels, adapter_rank, modes))

    @staticmethod
    def compl_mul1d(x_hat, w_real, w_imag):
        xr, xi = x_hat.real, x_hat.imag
        out_real = torch.einsum("bcm,ocm->bom", xr, w_real) - torch.einsum("bcm,ocm->bom", xi, w_imag)
        out_imag = torch.einsum("bcm,ocm->bom", xr, w_imag) + torch.einsum("bcm,ocm->bom", xi, w_real)
        return torch.complex(out_real, out_imag)

    def forward(self, x):
        batch, channels, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(batch, channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        raw = x_ft[:, :, :m]
        modes = raw if self.gate is None else self.gate(raw.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.compl_mul1d(modes, self.weight_real[:, :, :m], self.weight_imag[:, :, :m])
        if self.use_adapter:
            hidden = self.compl_mul1d(raw, self.adapter_a_real[:, :, :m], self.adapter_a_imag[:, :, :m])
            out = out + self.compl_mul1d(hidden, self.adapter_b_real[:, :, :m], self.adapter_b_imag[:, :, :m])
        out_ft[:, :, :m] = out
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, in_dim=3, width=32, modes=16, depth=4, gate_variant=None,
                 gate_learnable=True, gate_strength=1.0, gate_alpha=2.0, gate_p=2.0,
                 highfreq_only=False, low_keep=4, gate_mode="legacy", residual_beta=0.2,
                 use_dynamic_score=True, use_model_score=True, use_adapter=False, adapter_rank=4):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, width)
        self.gates, self.spec_layers, self.w_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(depth):
            gate = None
            if gate_variant is not None:
                gate = DualDrivenSpectralGate1d(
                    modes=modes, variant=gate_variant, learnable=gate_learnable,
                    gate_strength=gate_strength, alpha=gate_alpha, p=gate_p,
                    highfreq_only=highfreq_only, low_keep=low_keep, gate_mode=gate_mode,
                    residual_beta=residual_beta, use_dynamic_score=use_dynamic_score,
                    use_model_score=use_model_score)
                self.gates.append(gate)
            self.spec_layers.append(SpectralConv1d(width, modes, gate, use_adapter, adapter_rank))
            self.w_layers.append(nn.Conv1d(width, width, 1))
        self.fc1, self.fc2 = nn.Linear(width, 64), nn.Linear(64, 1)

    @torch.no_grad()
    def fit_gates_from_data(self, u_true):
        for gate in self.gates:
            gate.fit_from_data(u_true)

    def forward(self, a):
        x = self.fc0(a).permute(0, 2, 1)
        for spec, pointwise in zip(self.spec_layers, self.w_layers):
            x = F.gelu(spec(x) + pointwise(x))
        x = F.gelu(self.fc1(x.permute(0, 2, 1)))
        return self.fc2(x).squeeze(-1)


class SpectralConv2d(nn.Module):
    """2-D spectral convolution with an optional band-shared complex adapter."""

    def __init__(self, channels, modes1, modes2, gate=None, use_adapter=False,
                 adapter_rank=4, n_bands=4):
        super().__init__()
        self.channels, self.modes1, self.modes2 = channels, modes1, modes2
        self.gate, self.use_adapter = gate, use_adapter
        self.adapter_rank, self.n_bands = adapter_rank, n_bands
        self.adapter_enabled = True
        self.register_buffer("band_index", self._make_band_index(bottom=False))
        self.register_buffer("band_index_bottom", self._make_band_index(bottom=True))
        scale = 1.0 / (channels * channels)
        self.weight1_real = nn.Parameter(scale * torch.randn(channels, channels, modes1, modes2))
        self.weight1_imag = nn.Parameter(scale * torch.randn(channels, channels, modes1, modes2))
        self.weight2_real = nn.Parameter(scale * torch.randn(channels, channels, modes1, modes2))
        self.weight2_imag = nn.Parameter(scale * torch.randn(channels, channels, modes1, modes2))
        if use_adapter:
            adapter_scale = 1.0 / max(1, channels * adapter_rank)
            for branch in (1, 2):
                self.register_parameter(
                    f"adapter{branch}_a_real",
                    nn.Parameter(adapter_scale * torch.randn(n_bands, adapter_rank, channels)))
                self.register_parameter(
                    f"adapter{branch}_a_imag",
                    nn.Parameter(adapter_scale * torch.randn(n_bands, adapter_rank, channels)))
                self.register_parameter(
                    f"adapter{branch}_b_real", nn.Parameter(torch.zeros(n_bands, channels, adapter_rank)))
                self.register_parameter(
                    f"adapter{branch}_b_imag", nn.Parameter(torch.zeros(n_bands, channels, adapter_rank)))

    def _make_band_index(self, bottom=False):
        k1 = (torch.arange(self.modes1, 0, -1, dtype=torch.float32)
              if bottom else torch.arange(self.modes1, dtype=torch.float32))
        k2 = torch.arange(self.modes2, dtype=torch.float32)
        k1 = k1 / max(1.0, float(self.modes1))
        k2 = k2 / max(1.0, float(self.modes2 - 1))
        k1, k2 = torch.meshgrid(k1, k2, indexing="ij")
        radius = torch.sqrt(k1.square() + k2.square()) / (2.0**0.5)
        return (radius * self.n_bands).long().clamp(max=self.n_bands - 1)

    @staticmethod
    def compl_mul2d(x_hat, w_real, w_imag):
        xr, xi = x_hat.real, x_hat.imag
        out_real = torch.einsum("bcxy,ocxy->boxy", xr, w_real) - torch.einsum("bcxy,ocxy->boxy", xi, w_imag)
        out_imag = torch.einsum("bcxy,ocxy->boxy", xr, w_imag) + torch.einsum("bcxy,ocxy->boxy", xi, w_real)
        return torch.complex(out_real, out_imag)

    @staticmethod
    def band_adapter_mul2d(x_hat, band_index, a_real, a_imag, b_real, b_imag):
        output = torch.zeros_like(x_hat)
        for band in range(a_real.shape[0]):
            mask = (band_index == band).view(1, 1, *band_index.shape)
            band_input = x_hat * mask
            xr, xi = band_input.real, band_input.imag
            ar, ai = a_real[band], a_imag[band]
            hidden_real = torch.einsum("bcxy,rc->brxy", xr, ar) - torch.einsum("bcxy,rc->brxy", xi, ai)
            hidden_imag = torch.einsum("bcxy,rc->brxy", xr, ai) + torch.einsum("bcxy,rc->brxy", xi, ar)
            br, bi = b_real[band], b_imag[band]
            out_real = torch.einsum("brxy,cr->bcxy", hidden_real, br) - torch.einsum("brxy,cr->bcxy", hidden_imag, bi)
            out_imag = torch.einsum("brxy,cr->bcxy", hidden_real, bi) + torch.einsum("brxy,cr->bcxy", hidden_imag, br)
            output = output + torch.complex(out_real, out_imag)
        return output

    def _branch_forward(self, raw, branch, w_real, w_imag, a_real=None, a_imag=None,
                        b_real=None, b_imag=None):
        gated = raw
        if self.gate is not None:
            gated = self.gate(raw.permute(0, 2, 3, 1), branch=branch).permute(0, 3, 1, 2)
        main = self.compl_mul2d(gated, w_real, w_imag)
        if not self.use_adapter or not self.adapter_enabled:
            return main
        full_index = self.band_index_bottom if branch == "bottom" else self.band_index
        band_index = full_index[:raw.shape[-2], :raw.shape[-1]]
        return main + self.band_adapter_mul2d(raw, band_index, a_real, a_imag, b_real, b_imag)

    def forward(self, x):
        _, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        m1, m2 = min(self.modes1, h // 2), min(self.modes2, x_ft.shape[-1])
        out_ft = torch.zeros_like(x_ft)
        top = x_ft[:, :, :m1, :m2]
        bottom = x_ft[:, :, -m1:, :m2]
        adapter1 = ([self.adapter1_a_real, self.adapter1_a_imag,
                     self.adapter1_b_real, self.adapter1_b_imag] if self.use_adapter else [None] * 4)
        adapter2 = ([self.adapter2_a_real, self.adapter2_a_imag,
                     self.adapter2_b_real, self.adapter2_b_imag] if self.use_adapter else [None] * 4)
        out_ft[:, :, :m1, :m2] = self._branch_forward(
            top, "top", self.weight1_real[:, :, :m1, :m2], self.weight1_imag[:, :, :m1, :m2], *adapter1)
        out_ft[:, :, -m1:, :m2] = self._branch_forward(
            bottom, "bottom", self.weight2_real[:, :, :m1, :m2], self.weight2_imag[:, :, :m1, :m2], *adapter2)
        return torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))


class FNO2d(nn.Module):
    def __init__(self, in_dim=3, width=32, modes1=12, modes2=12, depth=4,
                 gate_variant=None, gate_learnable=False, gate_strength=0.2,
                 gate_alpha=2.0, gate_p=2.0, highfreq_only=False,
                 low_keep1=4, low_keep2=4, gate_mode="legacy", residual_beta=0.2,
                 use_dynamic_score=True, use_model_score=True, n_bands=4,
                 use_adapter=False, adapter_rank=4):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, width)
        self.gates, self.spec_layers, self.w_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(depth):
            gate = None
            if gate_variant is not None:
                gate = DualDrivenSpectralGate2d(
                    modes1=modes1, modes2=modes2, variant=gate_variant,
                    learnable=gate_learnable, gate_strength=gate_strength,
                    alpha=gate_alpha, p=gate_p, highfreq_only=highfreq_only,
                    low_keep1=low_keep1, low_keep2=low_keep2, gate_mode=gate_mode,
                    residual_beta=residual_beta, use_dynamic_score=use_dynamic_score,
                    use_model_score=use_model_score)
                self.gates.append(gate)
            self.spec_layers.append(SpectralConv2d(
                width, modes1, modes2, gate=gate, use_adapter=use_adapter,
                adapter_rank=adapter_rank, n_bands=n_bands))
            self.w_layers.append(nn.Conv2d(width, width, 1))
        self.fc1, self.fc2 = nn.Linear(width, 64), nn.Linear(64, 1)

    @torch.no_grad()
    def fit_gates_from_data(self, u_true):
        for gate in self.gates:
            gate.fit_from_data(u_true)

    def set_adapter_enabled(self, enabled):
        for layer in self.spec_layers:
            layer.adapter_enabled = bool(enabled)

    def parameter_breakdown(self):
        groups = {"backbone": 0, "gate": 0, "adapter": 0}
        for name, parameter in self.named_parameters():
            if "adapter" in name:
                group = "adapter"
            elif "gates" in name or ".gate." in name:
                group = "gate"
            else:
                group = "backbone"
            groups[group] += parameter.numel()
        groups["total"] = sum(groups.values())
        return groups

    def forward(self, a):
        x = self.fc0(a).permute(0, 3, 1, 2)
        for spec, pointwise in zip(self.spec_layers, self.w_layers):
            x = F.gelu(spec(x) + pointwise(x))
        x = F.gelu(self.fc1(x.permute(0, 2, 3, 1)))
        return self.fc2(x).squeeze(-1)
