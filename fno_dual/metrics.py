import math

import torch

from .data import add_grid_channels_2d


def _radial_bins(height, width, n_bins, device):
    ky = torch.fft.fftfreq(height, device=device).abs()
    kx = torch.fft.fftfreq(width, device=device).abs()
    radius = torch.sqrt(ky[:, None].square() + kx[None, :].square())
    radius = radius / radius.max().clamp_min(1e-12)
    return (radius * n_bins).long().clamp(max=n_bins - 1)


def radial_power_spectrum(fields, n_bins=None):
    """Return per-sample radial mean power for real fields [B,H,W]."""
    height, width = fields.shape[-2:]
    n_bins = n_bins or min(height, width) // 2
    power = torch.fft.fft2(fields, dim=(-2, -1), norm="ortho").abs().square()
    bins = _radial_bins(height, width, n_bins, fields.device)
    spectrum = fields.new_zeros(fields.shape[0], n_bins)
    for band in range(n_bins):
        mask = bins == band
        if mask.any():
            spectrum[:, band] = power[:, mask].mean(dim=-1)
    return spectrum


def psre(prediction, target, n_bins=None):
    pred_power = radial_power_spectrum(prediction, n_bins=n_bins)
    true_power = radial_power_spectrum(target, n_bins=n_bins)
    numerator = torch.linalg.vector_norm(pred_power - true_power, dim=-1)
    denominator = torch.linalg.vector_norm(true_power, dim=-1).clamp_min(1e-12)
    return (numerator / denominator).mean().item()


def high_frequency_relative_error(prediction, target, high_fraction=0.25):
    height, width = prediction.shape[-2:]
    n_bins = max(2, min(height, width) // 2)
    bins = _radial_bins(height, width, n_bins, prediction.device)
    first_high_bin = max(0, math.floor((1.0 - high_fraction) * n_bins))
    mask = bins >= first_high_bin
    pred_hat = torch.fft.fft2(prediction, dim=(-2, -1), norm="ortho")
    true_hat = torch.fft.fft2(target, dim=(-2, -1), norm="ortho")
    numerator = torch.linalg.vector_norm((pred_hat - true_hat)[:, mask], dim=-1)
    denominator = torch.linalg.vector_norm(true_hat[:, mask], dim=-1).clamp_min(1e-12)
    return (numerator / denominator).mean().item()


def relative_l2(prediction, target):
    error = (prediction - target).reshape(prediction.shape[0], -1)
    truth = target.reshape(target.shape[0], -1)
    return (torch.linalg.vector_norm(error, dim=-1) /
            torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-12)).mean().item()


@torch.no_grad()
def autoregressive_rollout(model, initial_state, n_steps, x_normalizer, y_normalizer):
    model.eval()
    state = initial_state
    predictions = []
    for _ in range(n_steps):
        model_input = add_grid_channels_2d(x_normalizer.encode(state).unsqueeze(-1))
        state = y_normalizer.decode(model(model_input))
        predictions.append(state)
    return torch.stack(predictions, dim=1)


def rollout_metrics(predictions, trajectories, horizons=(1, 5, 10, 20)):
    targets = trajectories[:, 1: predictions.shape[1] + 1]
    metrics = {}
    step_errors = []
    for step in range(predictions.shape[1]):
        step_errors.append(relative_l2(predictions[:, step], targets[:, step]))
    metrics["rollout_rel_l2_average"] = sum(step_errors) / len(step_errors)
    metrics["rollout_rel_l2_final"] = step_errors[-1]
    metrics["rollout_psre_average"] = sum(
        psre(predictions[:, step], targets[:, step]) for step in range(predictions.shape[1])
    ) / predictions.shape[1]
    metrics["rollout_psre_final"] = psre(predictions[:, -1], targets[:, -1])
    metrics["rollout_highfreq_error_final"] = high_frequency_relative_error(
        predictions[:, -1], targets[:, -1]
    )
    for horizon in horizons:
        if horizon <= predictions.shape[1]:
            metrics[f"rel_l2_t{horizon}"] = step_errors[horizon - 1]
            metrics[f"psre_t{horizon}"] = psre(predictions[:, horizon - 1], targets[:, horizon - 1])
    return metrics
