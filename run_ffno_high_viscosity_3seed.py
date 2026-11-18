"""Three-seed training and high-viscosity evaluation for one-sided Delta-Physics."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fno_dual.data import generate_navier_stokes_trajectories
from fno_dual.ffno import FFNO2d
from fno_dual.metrics import relative_l2
from fno_dual.normalizer import UnitGaussianNormalizer
from fno_dual.train import set_seed
from run_ffno_dense_viscosity_scan import (
    HORIZONS,
    METHODS,
    extended_metrics,
    flatten_pairs,
    model_inputs,
    rollout,
)


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
    best_rel_l2 = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
        total_loss = 0.0
        for start in range(0, train_inputs.shape[0], batch_size):
            index = permutation[start : start + batch_size]
            prediction = model(train_inputs[index])
            loss = F.mse_loss(prediction, train_targets[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * index.numel()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_increment = increment_normalizer.decode(model(val_inputs))
            val_prediction = val_states + val_increment
            val_rel_l2 = relative_l2(val_prediction, val_next)
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


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(seed_rows):
    excluded = {"seed", "method", "viscosity_ratio", "viscosity"}
    metric_names = [
        key for key, value in seed_rows[0].items()
        if key not in excluded and isinstance(value, (int, float))
        and key not in {"data_seconds", "train_seconds", "evaluation_seconds"}
    ]
    aggregates = []
    for ratio in sorted({row["viscosity_ratio"] for row in seed_rows}):
        for method in METHODS:
            selected = [
                row for row in seed_rows
                if row["viscosity_ratio"] == ratio and row["method"] == method
            ]
            output = {
                "method": method,
                "viscosity_ratio": ratio,
                "viscosity": selected[0]["viscosity"],
                "n_seeds": len(selected),
            }
            for name in metric_names:
                values = torch.tensor([row[name] for row in selected], dtype=torch.float64)
                output[f"{name}_mean"] = values.mean().item()
                output[f"{name}_std"] = values.std(unbiased=True).item()
            aggregates.append(output)
    return aggregates


def improvement_rows(seed_rows):
    metric_names = [f"rel_l2_t{horizon}" for horizon in HORIZONS]
    metric_names += [
        "rollout_rel_l2_average_t20",
        "rollout_rel_l2_average",
        "rollout_psre_average_t20",
        "rollout_psre_average",
        "rollout_psre_final",
    ]
    outputs = []
    for ratio in sorted({row["viscosity_ratio"] for row in seed_rows}):
        per_seed = []
        for seed in sorted({row["seed"] for row in seed_rows}):
            baseline = next(
                row for row in seed_rows
                if row["seed"] == seed and row["viscosity_ratio"] == ratio
                and row["method"] == "increment"
            )
            corrected = next(
                row for row in seed_rows
                if row["seed"] == seed and row["viscosity_ratio"] == ratio
                and row["method"] == "delta_physics"
            )
            values = {"seed": seed}
            for name in metric_names:
                values[name] = 100.0 * (baseline[name] - corrected[name]) / baseline[name]
            per_seed.append(values)
        output = {
            "viscosity_ratio": ratio,
            "viscosity": ratio * seed_rows[0]["reference_viscosity"],
            "n_seeds": len(per_seed),
        }
        for name in metric_names:
            values = torch.tensor([row[name] for row in per_seed], dtype=torch.float64)
            output[f"improvement_{name}_percent_mean"] = values.mean().item()
            output[f"improvement_{name}_percent_std"] = values.std(unbiased=True).item()
        outputs.append(output)
    return outputs


def plot_improvements(
    path,
    rows,
    title="One-sided Delta-Physics: mean ± std over 3 seeds",
):
    import matplotlib.pyplot as plt

    ratios = [row["viscosity_ratio"] for row in rows]
    fig, axis = plt.subplots(figsize=(8.5, 5.3))
    for horizon in HORIZONS:
        means = [row[f"improvement_rel_l2_t{horizon}_percent_mean"] for row in rows]
        stds = [row[f"improvement_rel_l2_t{horizon}_percent_std"] for row in rows]
        axis.errorbar(
            ratios, means, yerr=stds, marker="o", linewidth=1.8,
            capsize=3, label=f"step {horizon}",
        )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.axvline(1.0, color="gray", linewidth=1.0, linestyle="--")
    axis.set_xlabel(r"Viscosity ratio $\nu/\nu_0$")
    axis.set_ylabel("Improvement over Increment (%)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/ffno_high_viscosity_3seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--step-dt", type=float, default=0.05)
    parser.add_argument("--solver-dt", type=float, default=0.005)
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-val", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--reference-viscosity", type=float, default=1e-3)
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
        train_pool = generate_navier_stokes_trajectories(
            args.n_train + args.n_val, args.grid, args.reference_viscosity,
            args.train_steps, args.step_dt, args.solver_dt, seed, device,
        )
        train_trajectories = train_pool[: args.n_train]
        val_trajectories = train_pool[args.n_train :]
        test_sets = {
            ratio: generate_navier_stokes_trajectories(
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
        model = FFNO2d(
            width=args.width, modes=args.modes, depth=args.depth, factor=args.factor
        ).to(device)
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
        torch.save(
            {
                "input_mean": input_normalizer.mean,
                "input_std": input_normalizer.std,
                "increment_mean": increment_normalizer.mean,
                "increment_std": increment_normalizer.std,
            },
            seed_dir / "normalizers.pt",
        )

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
                    **extended_metrics(prediction, trajectories),
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
    plot_improvements(out_dir / "high_viscosity_3seed_curve.png", improvements)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
