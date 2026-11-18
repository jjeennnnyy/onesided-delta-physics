"""No-retraining dense viscosity scan for Increment F-FNO plus Delta-Physics."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from fno_dual.data import add_grid_channels_2d, generate_navier_stokes_trajectories
from fno_dual.ffno import FFNO2d
from fno_dual.metrics import psre, relative_l2, rollout_metrics
from fno_dual.normalizer import UnitGaussianNormalizer
from fno_dual.physics import relative_diffusion_correction
from fno_dual.train import set_seed


METHODS = ("increment", "delta_physics")
HORIZONS = (1, 5, 10, 20, 50)


def flatten_pairs(trajectories):
    return (
        trajectories[:, :-1].reshape(-1, *trajectories.shape[2:]),
        trajectories[:, 1:].reshape(-1, *trajectories.shape[2:]),
    )


def model_inputs(states, input_normalizer):
    return add_grid_channels_2d(input_normalizer.encode(states).unsqueeze(-1))


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
        prediction = prediction + relative_diffusion_correction(
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


def extended_metrics(predictions, trajectories):
    metrics = rollout_metrics(predictions, trajectories, horizons=HORIZONS)
    targets = trajectories[:, 1 : predictions.shape[1] + 1]
    step_errors = [
        relative_l2(predictions[:, step], targets[:, step])
        for step in range(predictions.shape[1])
    ]
    metrics["rollout_rel_l2_average_t20"] = sum(step_errors[:20]) / 20
    metrics["rollout_psre_average_t20"] = sum(
        psre(predictions[:, step], targets[:, step]) for step in range(20)
    ) / 20
    return metrics


def improvement_rows(result_rows, reference_viscosity):
    output = []
    for ratio in sorted({row["viscosity_ratio"] for row in result_rows}):
        baseline = next(
            row for row in result_rows
            if row["viscosity_ratio"] == ratio and row["method"] == "increment"
        )
        corrected = next(
            row for row in result_rows
            if row["viscosity_ratio"] == ratio and row["method"] == "delta_physics"
        )
        row = {
            "viscosity_ratio": ratio,
            "viscosity": ratio * reference_viscosity,
        }
        metric_names = [f"rel_l2_t{horizon}" for horizon in HORIZONS]
        metric_names += [
            "rollout_rel_l2_average_t20",
            "rollout_rel_l2_average",
            "rollout_psre_average_t20",
            "rollout_psre_average",
            "rollout_psre_final",
        ]
        for name in metric_names:
            base_value = baseline[name]
            corrected_value = corrected[name]
            row[f"increment_{name}"] = base_value
            row[f"delta_{name}"] = corrected_value
            row[f"improvement_{name}_percent"] = (
                100.0 * (base_value - corrected_value) / base_value
            )
        output.append(row)
    return output


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_improvements(path, rows):
    import matplotlib.pyplot as plt

    ratios = [row["viscosity_ratio"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True)
    for axis in axes:
        for horizon in HORIZONS:
            values = [row[f"improvement_rel_l2_t{horizon}_percent"] for row in rows]
            axis.plot(ratios, values, marker="o", linewidth=1.8, label=f"step {horizon}")
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.axvline(1.0, color="gray", linewidth=1.0, linestyle="--")
        axis.set_xlabel(r"Viscosity ratio $\nu/\nu_0$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Delta-Physics improvement over Increment (%)")
    axes[0].set_title("Full range")
    axes[1].set_ylim(-20, 30)
    axes[1].set_title("Zoomed regime boundary")
    axes[1].legend(ncol=2)
    fig.suptitle("Dense viscosity scan (single seed, no retraining)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/ffno_dense_viscosity_scan")
    parser.add_argument(
        "--checkpoint", default="results/ffno_physics_split_screen/increment.pt"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--normalizer-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--step-dt", type=float, default=0.05)
    parser.add_argument("--solver-dt", type=float, default=0.005)
    parser.add_argument("--n-reference-train", type=int, default=24)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--reference-viscosity", type=float, default=1e-3)
    parser.add_argument(
        "--viscosity-ratios", nargs="+", type=float,
        default=[0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0],
    )
    args = parser.parse_args()
    if args.steps < max(HORIZONS):
        raise ValueError(f"steps must be at least {max(HORIZONS)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {**vars(args), "methods": METHODS, "horizons": HORIZONS, "device": device}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True)

    data_start = time.perf_counter()
    reference_train = generate_navier_stokes_trajectories(
        args.n_reference_train, args.grid, args.reference_viscosity,
        args.normalizer_steps, args.step_dt, args.solver_dt, args.seed, device,
    )
    test_sets = {
        ratio: generate_navier_stokes_trajectories(
            args.n_test, args.grid, ratio * args.reference_viscosity,
            args.steps, args.step_dt, args.solver_dt, 1000 + args.seed, device,
        )
        for ratio in args.viscosity_ratios
    }
    data_seconds = time.perf_counter() - data_start

    reference_states, reference_next = flatten_pairs(reference_train)
    input_normalizer = UnitGaussianNormalizer(reference_states.unsqueeze(-1))
    increment_normalizer = UnitGaussianNormalizer(reference_next - reference_states)
    set_seed(args.seed)
    model = FFNO2d(
        width=args.width, modes=args.modes, depth=args.depth, factor=args.factor
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    evaluation_start = time.perf_counter()
    for ratio, trajectories in test_sets.items():
        predictions = {}
        for method in METHODS:
            prediction = rollout(
                model, trajectories[:, 0], args.steps, method,
                ratio * args.reference_viscosity, args,
                input_normalizer, increment_normalizer,
            )
            predictions[method] = prediction
            rows.append({
                "method": method,
                "viscosity_ratio": ratio,
                "viscosity": ratio * args.reference_viscosity,
                "data_seconds": data_seconds,
                **extended_metrics(prediction, trajectories),
            })
        max_difference = (predictions["increment"] - predictions["delta_physics"]).abs().max().item()
        print({"viscosity_ratio": ratio, "rollout_max_difference": max_difference}, flush=True)
    evaluation_seconds = time.perf_counter() - evaluation_start
    for row in rows:
        row["evaluation_seconds"] = evaluation_seconds

    improvements = improvement_rows(rows, args.reference_viscosity)
    write_csv(out_dir / "results.csv", rows)
    write_csv(out_dir / "improvements.csv", improvements)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "improvements.json").write_text(
        json.dumps(improvements, indent=2), encoding="utf-8"
    )
    plot_improvements(out_dir / "viscosity_improvement_curve.png", improvements)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
