import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(x):
    x = torch.as_tensor(float(x))
    return torch.log(torch.expm1(x))


def _zscore(x, dims, eps):
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True).clamp_min(eps)
    return (x - mean) / std


class DualDrivenSpectralGate1d(nn.Module):
    """Original 1-D gate, kept for the wave experiments."""

    def __init__(self, modes, alpha=2.0, p=2.0, eps=1e-6, variant="data_model",
                 learnable=True, gate_strength=1.0, highfreq_only=False, low_keep=4,
                 gate_mode="legacy", residual_beta=0.2, use_dynamic_score=True,
                 use_model_score=True, init_w_data=1.0, init_w_model=1.0):
        super().__init__()
        self.modes, self.alpha, self.p, self.eps = modes, alpha, p, eps
        self.variant, self.gate_strength = variant, gate_strength
        self.highfreq_only, self.low_keep = highfreq_only, low_keep
        self.gate_mode, self.residual_beta = gate_mode, residual_beta
        self.use_dynamic_score, self.use_model_score = use_dynamic_score, use_model_score
        self.register_buffer("base_logits", torch.zeros(modes))
        self.register_buffer("model_score", torch.zeros(modes))
        self.delta_logits = nn.Parameter(torch.zeros(modes), requires_grad=learnable)
        self.raw_w_data = nn.Parameter(_inv_softplus(init_w_data).clone())
        self.raw_w_model = nn.Parameter(_inv_softplus(init_w_model).clone())
        self.bias = nn.Parameter(torch.zeros(()))

    @torch.no_grad()
    def fit_from_data(self, u_true):
        u_hat = torch.fft.rfft(u_true, dim=-1, norm="forward")
        power = u_hat.abs().square().mean(dim=0)
        m = min(self.modes, power.shape[0])
        power = power[:m].clamp_min(self.eps)
        k = torch.arange(m, device=u_true.device, dtype=torch.float32)
        prior = (1.0 + (k / max(1.0, float(m - 1))) ** 2).pow(-self.p).clamp_min(self.eps)
        self.model_score.zero_()
        self.model_score[:m].copy_(_zscore(torch.log(prior), dims=-1, eps=self.eps))
        if self.variant == "data_only":
            score = torch.log(power)
        elif self.variant == "model_only":
            score = torch.log(prior)
        else:
            score = torch.log(power) - torch.log(prior)
        gate = torch.sigmoid(self.alpha * score).clamp(self.eps, 1.0 - self.eps)
        self.base_logits.zero_()
        self.base_logits[:m].copy_(torch.logit(gate))

    def _sample_score(self, x_hat_modes):
        power = x_hat_modes.abs().square().mean(dim=-1)
        relative_power = power / (power.sum(dim=-1, keepdim=True) + self.eps)
        return _zscore(torch.log(relative_power + self.eps), dims=-1, eps=self.eps)

    def current_gate(self, m=None):
        m = self.modes if m is None else m
        gate = torch.sigmoid(self.base_logits[:m] + self.delta_logits[:m])
        if self.highfreq_only:
            gate = gate.clone()
            gate[: min(self.low_keep, m)] = 1.0
        return (1.0 - self.gate_strength) + self.gate_strength * gate

    def forward(self, x_hat_modes):
        m = x_hat_modes.shape[1]
        if self.gate_mode == "legacy":
            gate = self.current_gate(m).view(1, m, 1)
        else:
            score = self.bias
            if self.use_dynamic_score:
                score = score + F.softplus(self.raw_w_data) * self._sample_score(x_hat_modes)
            if self.use_model_score:
                score = score + F.softplus(self.raw_w_model) * self.model_score[:m]
            gate = 1.0 + self.residual_beta * torch.tanh(score)
            if self.highfreq_only:
                gate = gate.clone()
                gate[..., : min(self.low_keep, m)] = 1.0
            gate = gate.unsqueeze(-1) if gate.dim() > 1 else gate.view(1, m, 1)
        return x_hat_modes * gate


class DualDrivenSpectralGate2d(nn.Module):
    """Mode-wise dual data/model residual gate for 2-D spectra."""

    VALID_MODES = {"legacy", "dual"}

    def __init__(self, modes1, modes2, alpha=2.0, p=2.0, eps=1e-6,
                 variant="data_model", learnable=True, gate_strength=0.2,
                 highfreq_only=False, low_keep1=4, low_keep2=4, gate_mode="legacy",
                 residual_beta=0.2, use_dynamic_score=True, use_model_score=True,
                 init_w_data=1.0, init_w_model=1.0):
        super().__init__()
        if gate_mode not in self.VALID_MODES:
            raise ValueError(f"gate_mode must be one of {sorted(self.VALID_MODES)}, got {gate_mode!r}")
        self.modes1, self.modes2 = modes1, modes2
        self.alpha, self.p, self.eps = alpha, p, eps
        self.variant, self.gate_strength = variant, gate_strength
        self.highfreq_only, self.low_keep1, self.low_keep2 = highfreq_only, low_keep1, low_keep2
        self.gate_mode, self.residual_beta = gate_mode, residual_beta
        self.use_dynamic_score, self.use_model_score = use_dynamic_score, use_model_score
        self.register_buffer("base_logits", torch.zeros(modes1, modes2))
        self.register_buffer("model_score", torch.zeros(modes1, modes2))
        self.register_buffer("model_score_bottom", torch.zeros(modes1, modes2))
        if gate_mode == "legacy":
            self.delta_logits = nn.Parameter(torch.zeros(modes1, modes2), requires_grad=learnable)
        else:
            self.register_buffer("delta_logits", torch.zeros(modes1, modes2))
        self.raw_w_data = nn.Parameter(_inv_softplus(init_w_data).clone())
        self.raw_w_model = nn.Parameter(_inv_softplus(init_w_model).clone())
        self.bias = nn.Parameter(torch.zeros(()))
        self._initialize_model_prior()

    def _frequency_grid(self, bottom=False, device=None):
        k1 = (torch.arange(self.modes1, 0, -1, device=device, dtype=torch.float32)
              if bottom else torch.arange(self.modes1, device=device, dtype=torch.float32))
        k2 = torch.arange(self.modes2, device=device, dtype=torch.float32)
        return torch.meshgrid(k1 / max(1.0, float(self.modes1)),
                              k2 / max(1.0, float(self.modes2 - 1)), indexing="ij")

    @torch.no_grad()
    def _initialize_model_prior(self):
        for bottom, target in ((False, self.model_score), (True, self.model_score_bottom)):
            k1, k2 = self._frequency_grid(bottom=bottom, device=target.device)
            prior = (1.0 + k1.square() + k2.square()).pow(-self.p).clamp_min(self.eps)
            target.copy_(_zscore(torch.log(prior), dims=(-2, -1), eps=self.eps))

    @torch.no_grad()
    def fit_from_data(self, u_true):
        self._initialize_model_prior()
        if self.gate_mode != "legacy":
            return
        u_hat = torch.fft.rfft2(u_true, dim=(-2, -1), norm="forward")
        power = u_hat.abs().square().mean(dim=0)
        m1, m2 = min(self.modes1, power.shape[0] // 2), min(self.modes2, power.shape[1])
        used = 0.5 * (power[:m1, :m2] + power[-m1:, :m2].flip(0))
        prior = self.model_score[:m1, :m2]
        if self.variant == "data_only":
            score = torch.log(used.clamp_min(self.eps))
        elif self.variant == "model_only":
            score = prior
        else:
            score = torch.log(used.clamp_min(self.eps)) - prior
        gate = torch.sigmoid(self.alpha * score).clamp(self.eps, 1.0 - self.eps)
        self.base_logits.zero_()
        self.base_logits[:m1, :m2].copy_(torch.logit(gate))

    def _model_score(self, branch, m1, m2):
        if branch == "bottom":
            return self.model_score_bottom[:m1, :m2]
        return self.model_score[:m1, :m2]

    def _sample_score(self, x_hat_modes):
        power = x_hat_modes.abs().square().mean(dim=-1)
        relative_power = power / (power.sum(dim=(-2, -1), keepdim=True) + self.eps)
        return _zscore(torch.log(relative_power + self.eps), dims=(-2, -1), eps=self.eps)

    def current_gate(self, m1=None, m2=None):
        m1 = self.modes1 if m1 is None else m1
        m2 = self.modes2 if m2 is None else m2
        gate = torch.sigmoid(self.base_logits[:m1, :m2] + self.delta_logits[:m1, :m2])
        return (1.0 - self.gate_strength) + self.gate_strength * gate

    def forward(self, x_hat_modes, branch="top"):
        m1, m2 = x_hat_modes.shape[1:3]
        if self.gate_mode == "legacy":
            gate = self.current_gate(m1, m2).view(1, m1, m2, 1)
        else:
            model_score = self._model_score(branch, m1, m2)
            score = self.bias
            if self.use_dynamic_score:
                score = score + F.softplus(self.raw_w_data) * self._sample_score(x_hat_modes)
            if self.use_model_score:
                score = score + F.softplus(self.raw_w_model) * model_score
            gate = 1.0 + self.residual_beta * torch.tanh(score)
            if self.highfreq_only:
                gate = gate.clone()
                gate[..., :min(self.low_keep1, m1), :min(self.low_keep2, m2)] = 1.0
            gate = gate.unsqueeze(-1)
        return x_hat_modes * gate
