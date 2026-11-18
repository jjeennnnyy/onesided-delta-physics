"""Three-seed one-sided Delta-Physics experiment on viscous Burgers."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fno_dual.data import add_grid_channel_1d, generate_burgers_trajectories
from fno_dual.fno1d import FNO1d
from fno_dual.metrics import relative_l2
from fno_dual.normalizer import UnitGaussianNormalizer
from fno_dual.physics import relative_diffusion_correction_1d
from fno_dual.train import set_seed
from run_ffno_high_viscosity_3seed import (
    HORIZONS,
    METHODS,
    aggregate_rows,
    improvement_rows,
    plot_improvements,
    write_csv,
)


def flatten_pairs(trajectories):
    return (
        trajectories[:, :-1].reshape(-1, trajectories.shape[-1]),
        trajectories[:, 1:].reshape(-1, trajectories.shape[-1]),
    )


def model_inputs(states, input_normalizer):
    normalized = input_normalizer.encode(states.unsqueeze(-1))
    return add_grid_channel_1d(normalized)


def psre_1d(prediction, target):
    prediction_power = torch.fft.rfft(
        prediction, dim=-1, norm="ortho"
    ).abs().square()
    target_power = torch.fft.rfft(target, dim=-1, norm="ortho").abs().square()
    numerator = torch.linalg.vector_norm(prediction_power - target_power, dim=-1)
    denominator = torch.linalg.vector_norm(target_power, dim=-1).clamp_min(1e-12)
    return (numerator / denominator).mean().item()


def predict_next(
    model,
    states,
    method,
    viscosity,
    reference_viscosity,
    delta_t,
    input_normalizer,
    increment_normalizer,
):
    increment = increment_normalizer.decode(model(model_inputs(states, input_normalizer)))
    prediction = states + increment
    if method == "delta_physics":
        prediction = prediction + relative_diffusion_correction_1d(
            states, viscosity, reference_viscosity, delta_t
        )
    return prediction


@torch.no_grad()
def rollout(
    model,
    initial_state,
    steps,
    method,
    viscosity,
    args,
    input_normalizer,
    increment_normalizer,
):
    model.eval()
    state = initial_state
    predictions = []
    for _ in range(steps):
        state = predict_next(
            model, state, method, viscosity, args.reference_viscosity,
            args.step_dt, input_normalizer, increment_normalizer,
        )
        predictions.append(state)
    return torch.stack(predictions, dim=1)


def rollout_metrics_1d(predictions, trajectories):
    targets = trajectories[:, 1 : predictions.shape[1] + 1]
    rel_errors = [
        relative_l2(predictions[:, step], targets[:, step])
        for step in range(predictions.shape[1])
    ]
    spectral_errors = [
        psre_1d(predictions[:, step], targets[:, step])
        for step in range(predictions.shape[1])
    ]
    metrics = {
        "rollout_rel_l2_average": sum(rel_errors) / len(rel_errors),
        "rollout_rel_l2_average_t20": sum(rel_errors[:20]) / 20,
        "rollout_rel_l2_final": rel_errors[-1],
        "rollout_psre_average": sum(spectral_errors) / len(spectral_errors),
        "rollout_psre_average_t20": sum(spectral_errors[:20]) / 20,
        "rollout_psre_final": spectral_errors[-1],
    }
    for horizon in HORIZONS:
        metrics[f"rel_l2_t{horizon}"] = rel_errors[horizon - 1]
        metrics[f"psre_t{horizon}"] = spectral_errors[horizon - 1]
    return metrics


def train_increment(
    model,
    train_trajectories,
    val_trajectories,
    input_normalizer,
    increment_normalizer,
    epochs,
    batch_size,
):
    train_states, train_next = flatten_pairs(train_trajectories)
    val_states, val_next = flatten_pairs(val_trajectories)
    train_inputs = model_inputs(train_states, input_normalizer)
    train_targets = increment_normalizer.encode(train_next - train_states)
    val_inputs = model_inputs(val_states, input_normalizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, epochs // 2), gamma=0.5
    )
    best_rel_l2, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
        total_loss = 0.0
        for start in range(0, train_inputs.shape[0], batch_size):
            index = permutation[start : start + batch_size]
            loss = F.mse_loss(model(train_inputs[index]), train_targets[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * index.numel()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            prediction = val_states + increment_normalizer.decode(model(val_inputs))
            val_rel_l2 = relative_l2(prediction, val_next)
        if val_rel_l2 < best_rel_l2:
            best_rel_l2 = val_rel_l2
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % max(1, epochs // 3) == 0 or epoch == epochs:
            print(
                f"  epoch={epoch:03d} train_mse_norm={total_loss / train_inputs.shape[0]:.4e} "
                f"val_rel_l2={val_rel_l2:.4e}",
                flush=True,
            )
    model.load_state_dict(best_state)
    return best_rel_l2, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/burgers_high_viscosity_3seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--step-dt", type=float, default=0.02)
    parser.add_argument("--solver-dt", type=float, default=0.002)
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-val", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--reference-viscosity", type=float, default=1e-2)
    parser.add_argument(
        "--viscosity-ratios", nargs="+", type=float,
        default=[1.0, 1.25, 1.5, 1.75, 2.0],
    )
    args = parser.parse_args()
    if args.eval_steps < max(HORIZONS):
        raise ValueError(f"eval-steps must be at least {max(HORIZONS)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {**vars(args), "methods": METHODS, "horizons": HORIZONS, "device": device}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True)

    rows = []
    for seed in args.seeds:
        print({"seed": seed, "stage": "data"}, flush=True)
        data_start = time.perf_counter()
        train_pool = generate_burgers_trajectories(
            args.n_train + args.n_val, args.grid, args.reference_viscosity,
            args.train_steps, args.step_dt, args.solver_dt, seed, device,
        )
        train_trajectories = train_pool[: args.n_train]
        val_trajectories = train_pool[args.n_train :]
        test_sets = {
            ratio: generate_burgers_trajectories(
                args.n_test, args.grid, ratio * args.reference_viscosity,
                args.eval_steps, args.step_dt, args.solver_dt, 1000 + seed, device,
            )
            for ratio in args.viscosity_ratios
        }
        data_seconds = time.perf_counter() - data_start
        train_states, train_next = flatten_pairs(train_trajectories)
        input_normalizer = UnitGaussianNormalizer(train_states.unsqueeze(-1))
        increment_normalizer = UnitGaussianNormalizer(train_next - train_states)

        set_seed(seed)
        model = FNO1d(width=args.width, modes=args.modes, depth=args.depth).to(device)
        set_seed(seed + 10000)
        print({"seed": seed, "stage": "train"}, flush=True)
        train_start = time.perf_counter()
        best_val, best_epoch = train_increment(
            model, train_trajectories, val_trajectories,
            input_normalizer, increment_normalizer, args.epochs, args.batch_size,
        )
        train_seconds = time.perf_counter() - train_start
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), seed_dir / "increment.pt")

        print({"seed": seed, "stage": "evaluate"}, flush=True)
        evaluation_start = time.perf_counter()
        for ratio, trajectories in test_sets.items():
            predictions = {}
            for method in METHODS:
                prediction = rollout(
                    model, trajectories[:, 0], args.eval_steps, method,
                    ratio * args.reference_viscosity, args,
                    input_normalizer, increment_normalizer,
                )
                predictions[method] = prediction
                rows.append({
                    "seed": seed,
                    "method": method,
                    "viscosity_ratio": ratio,
                    "viscosity": ratio * args.reference_viscosity,
                    "reference_viscosity": args.reference_viscosity,
                    "best_val_rel_l2": best_val,
                    "best_val_epoch": best_epoch,
                    "data_seconds": data_seconds,
                    "train_seconds": train_seconds,
                    **rollout_metrics_1d(prediction, trajectories),
                })
            difference = (predictions["increment"] - predictions["delta_physics"]).abs().max().item()
            print({"seed": seed, "ratio": ratio, "max_difference": difference}, flush=True)
        evaluation_seconds = time.perf_counter() - evaluation_start
        for row in rows:
            if row["seed"] == seed:
                row["evaluation_seconds"] = evaluation_seconds
        write_csv(out_dir / "seed_results.csv", rows)
        (out_dir / "seed_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    aggregates = aggregate_rows(rows)
    improvements = improvement_rows(rows)
    write_csv(out_dir / "aggregate_metrics.csv", aggregates)
    write_csv(out_dir / "aggregate_improvements.csv", improvements)
    (out_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregates, indent=2), encoding="utf-8"
    )
    (out_dir / "aggregate_improvements.json").write_text(
        json.dumps(improvements, indent=2), encoding="utf-8"
    )
    plot_improvements(
        out_dir / "burgers_high_viscosity_3seed_curve.png",
        improvements,
        title="Viscous Burgers: one-sided Delta-Physics (3 seeds)",
    )
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
