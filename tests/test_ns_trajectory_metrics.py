import torch

from fno_dual.data import generate_navier_stokes_trajectories, trajectory_pairs
from fno_dual.metrics import high_frequency_relative_error, psre, rollout_metrics


def test_navier_stokes_trajectory_shape_and_determinism():
    kwargs = dict(
        n_trajectories=2,
        n_grid=8,
        n_steps=2,
        step_dt=0.01,
        solver_dt=0.01,
        seed=7,
    )
    first = generate_navier_stokes_trajectories(**kwargs)
    second = generate_navier_stokes_trajectories(**kwargs)
    assert first.shape == (2, 3, 8, 8)
    assert torch.equal(first, second)
    inputs, targets = trajectory_pairs(first)
    assert inputs.shape == (4, 8, 8, 1)
    assert targets.shape == (4, 8, 8)


def test_spectral_and_rollout_metrics_are_zero_for_exact_prediction():
    trajectories = torch.randn(3, 5, 8, 8)
    predictions = trajectories[:, 1:].clone()
    assert psre(predictions[:, -1], trajectories[:, -1]) == 0.0
    assert high_frequency_relative_error(predictions[:, -1], trajectories[:, -1]) == 0.0
    metrics = rollout_metrics(predictions, trajectories, horizons=(1, 4))
    assert metrics["rollout_rel_l2_average"] == 0.0
    assert metrics["rollout_rel_l2_final"] == 0.0
