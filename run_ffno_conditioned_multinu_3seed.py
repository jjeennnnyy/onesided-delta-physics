"""Strong multi-viscosity conditioned F-FNO baseline at 64 x 64 resolution."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fno_dual.data import add_grid_channels_2d, generate_navier_stokes_trajectories
from fno_dual.ffno import FFNO2d
from fno_dual.metrics import relative_l2
from fno_dual.normalizer import UnitGaussianNormalizer
from fno_dual.train import set_seed
from run_ffno_dense_viscosity_scan import extended_metrics, flatten_pairs, rollout


HORIZONS = (1, 10, 20, 50)
METHODS = ("single_nu_delta_physics", "multi_nu_conditioned")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def conditioned_inputs(states, ratios, input_normalizer, q_mean, q_std):
    state = input_normalizer.encode(states).unsqueeze(-1)
    q = (ratios.log() - q_mean) / q_std
    q = q.reshape(-1, 1, 1, 1).expand(-1, states.shape[-2], states.shape[-1], 1)
    return add_grid_channels_2d(torch.cat((state, q), dim=-1))


def flatten_multinu(trajectory_sets):
    states, next_states, ratios = [], [], []
    for ratio, trajectories in trajectory_sets.items():
        current, following = flatten_pairs(trajectories)
        states.append(current)
        next_states.append(following)
        ratios.append(current.new_full((current.shape[0],), ratio))
    return torch.cat(states), torch.cat(next_states), torch.cat(ratios)


def train_conditioned(
    model,
    train_sets,
    val_sets,
    input_normalizer,
    increment_normalizer,
    q_mean,
    q_std,
    epochs,
    batch_size,
):
    train_states, train_next, train_ratios = flatten_multinu(train_sets)
    val_states, val_next, val_ratios = flatten_multinu(val_sets)
    train_inputs = conditioned_inputs(
        train_states, train_ratios, input_normalizer, q_mean, q_std
    )
    train_targets = increment_normalizer.encode(train_next - train_states)
    val_inputs = conditioned_inputs(
        val_states, val_ratios, input_normalizer, q_mean, q_std
    )

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
            prediction = model(train_inputs[index])
            loss = F.mse_loss(prediction, train_targets[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * index.numel()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            increment = increment_normalizer.decode(model(val_inputs))
            val_rel_l2 = relative_l2(val_states + increment, val_next)
        if val_rel_l2 < best_rel_l2:
            best_rel_l2 = val_rel_l2
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % max(1, epochs // 3) == 0 or epoch == epochs:
            print(
                f"  epoch={epoch:03d} train_mse_norm="
                f"{total_loss / train_inputs.shape[0]:.4e} "
                f"val_rel_l2={val_rel_l2:.4e}",
                flush=True,
            )
    model.load_state_dict(best_state)
    return best_rel_l2, best_epoch


@torch.no_grad()
def conditioned_rollout(
    model,
    initial_state,
    steps,
    ratio,
    input_normalizer,
    increment_normalizer,
    q_mean,
    q_std,
):
    model.eval()
    state = initial_state
    ratios = state.new_full((state.shape[0],), ratio)
    predictions = []
    for _ in range(steps):
        inputs = conditioned_inputs(state, ratios, input_normalizer, q_mean, q_std)
        increment = increment_normalizer.decode(model(inputs))
        state = state + increment
        predictions.append(state)
    return torch.stack(predictions, dim=1)


def aggregate(seed_rows):
    excluded = {
        "seed", "method", "viscosity_ratio", "viscosity", "parameter_count",
        "training_viscosity_count", "training_trajectories", "training_pairs",
        "train_seconds", "inference_seconds", "best_val_rel_l2", "best_val_epoch",
    }
    metric_names = [
        key for key, value in seed_rows[0].items()
        if key not in excluded and isinstance(value, (int, float))
    ]
    output = []
    for ratio in sorted({row["viscosity_ratio"] for row in seed_rows}):
        for method in METHODS:
            selected = [
                row for row in seed_rows
                if row["viscosity_ratio"] == ratio and row["method"] == method
            ]
            row = {"method": method, "viscosity_ratio": ratio, "n_seeds": len(selected)}
            for name in metric_names:
                values = torch.tensor([item[name] for item in selected], dtype=torch.float64)
                row[f"{name}_mean"] = values.mean().item()
                row[f"{name}_std"] = values.std(unbiased=True).item()
            output.append(row)
    return output


def paired_comparison(seed_rows):
    metric_names = [f"rel_l2_t{step}" for step in HORIZONS] + [
        "rollout_rel_l2_average", "rollout_psre_average", "rollout_psre_final"
    ]
    output = []
    for ratio in sorted({row["viscosity_ratio"] for row in seed_rows}):
        comparisons = []
        for seed in sorted({row["seed"] for row in seed_rows}):
            delta = next(
                row for row in seed_rows if row["seed"] == seed
                and row["viscosity_ratio"] == ratio
                and row["method"] == "single_nu_delta_physics"
            )
            conditioned = next(
                row for row in seed_rows if row["seed"] == seed
                and row["viscosity_ratio"] == ratio
                and row["method"] == "multi_nu_conditioned"
            )
            comparisons.append({
                name: 100.0 * (conditioned[name] - delta[name]) / conditioned[name]
                for name in metric_names
            })
        row = {"viscosity_ratio": ratio, "n_seeds": len(comparisons)}
        for name in metric_names:
            values = torch.tensor([item[name] for item in comparisons], dtype=torch.float64)
            row[f"delta_improvement_over_conditioned_{name}_percent_mean"] = values.mean().item()
            row[f"delta_improvement_over_conditioned_{name}_percent_std"] = values.std(unbiased=True).item()
        output.append(row)
    return output


def plot_comparison(path, aggregate_rows):
    import matplotlib.pyplot as plt

    ratios = sorted({row["viscosity_ratio"] for row in aggregate_rows})
    labels = {
        "single_nu_delta_physics": "Single-nu + Delta-Physics",
        "multi_nu_conditioned": "Multi-nu conditioned",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for method in METHODS:
        selected = [
            next(row for row in aggregate_rows
                 if row["method"] == method and row["viscosity_ratio"] == ratio)
            for ratio in ratios
        ]
        for axis, metric, title in zip(
            axes,
            ("rollout_rel_l2_average", "rel_l2_t50"),
            ("Average 50-step RelL2", "Step-50 RelL2"),
        ):
            axis.errorbar(
                ratios,
                [row[f"{metric}_mean"] for row in selected],
                yerr=[row[f"{metric}_std"] for row in selected],
                marker="o", linewidth=2, capsize=3, label=labels[method],
            )
            axis.set_title(title)
            axis.set_xlabel(r"Viscosity ratio $\nu/\nu_0$")
            axis.grid(alpha=0.25)
    axes[0].set_ylabel("RelL2")
    axes[0].legend()
    fig.suptitle("64 x 64 Navier-Stokes, mean +/- std over 3 seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/ffno_ns64_conditioned_multinu_3seed")
    parser.add_argument("--reference-dir", default="results/ffno_ns64_high_viscosity_3seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--step-dt", type=float, default=0.05)
    parser.add_argument("--solver-dt", type=float, default=0.005)
    parser.add_argument("--n-train-per-viscosity", type=int, default=24)
    parser.add_argument("--n-val-per-viscosity", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--reference-viscosity", type=float, default=1e-3)
    parser.add_argument(
        "--train-ratios", nargs="+", type=float,
        default=[1.0, 1.25, 1.5, 1.75, 2.0],
    )
    parser.add_argument(
        "--test-ratios", nargs="+", type=float, default=[1.0, 1.5, 1.75, 2.0]
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(args.reference_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {**vars(args), "device": device, "methods": METHODS}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True)

    log_ratios = torch.tensor(args.train_ratios, device=device).log()
    q_mean = log_ratios.mean()
    q_std = log_ratios.std(unbiased=False).clamp_min(1e-6)
    rows = []
    cost_rows = []
    for seed in args.seeds:
        print({"seed": seed, "stage": "data"}, flush=True)
        data_start = time.perf_counter()
        train_sets, val_sets = {}, {}
        for index, ratio in enumerate(args.train_ratios):
            pool = generate_navier_stokes_trajectories(
                args.n_train_per_viscosity + args.n_val_per_viscosity,
                args.grid, ratio * args.reference_viscosity, args.train_steps,
                args.step_dt, args.solver_dt, seed + 100 * index, device,
            )
            train_sets[ratio] = pool[: args.n_train_per_viscosity]
            val_sets[ratio] = pool[args.n_train_per_viscosity :]
        test_sets = {
            ratio: generate_navier_stokes_trajectories(
                args.n_test, args.grid, ratio * args.reference_viscosity,
                args.eval_steps, args.step_dt, args.solver_dt, 1000 + seed, device,
            )
            for ratio in args.test_ratios
        }
        data_seconds = time.perf_counter() - data_start

        train_states, train_next, _ = flatten_multinu(train_sets)
        input_normalizer = UnitGaussianNormalizer(train_states.unsqueeze(-1))
        increment_normalizer = UnitGaussianNormalizer(train_next - train_states)
        set_seed(seed)
        conditioned = FFNO2d(
            in_dim=4, width=args.width, modes=args.modes,
            depth=args.depth, factor=args.factor,
        ).to(device)
        set_seed(seed + 10000)
        print({"seed": seed, "stage": "train"}, flush=True)
        train_start = time.perf_counter()
        best_val, best_epoch = train_conditioned(
            conditioned, train_sets, val_sets, input_normalizer,
            increment_normalizer, q_mean, q_std, args.epochs, args.batch_size,
        )
        train_seconds = time.perf_counter() - train_start

        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        torch.save(conditioned.state_dict(), seed_dir / "conditioned.pt")
        torch.save({
            "input_mean": input_normalizer.mean,
            "input_std": input_normalizer.std,
            "increment_mean": increment_normalizer.mean,
            "increment_std": increment_normalizer.std,
            "q_mean": q_mean,
            "q_std": q_std,
        }, seed_dir / "normalizers.pt")

        reference = FFNO2d(
            width=args.width, modes=args.modes, depth=args.depth, factor=args.factor
        ).to(device)
        reference.load_state_dict(torch.load(
            reference_dir / f"seed_{seed}" / "increment.pt",
            map_location=device, weights_only=True,
        ))
        reference_stats = torch.load(
            reference_dir / f"seed_{seed}" / "normalizers.pt",
            map_location=device, weights_only=True,
        )
        reference_input_normalizer = UnitGaussianNormalizer(torch.zeros(2, device=device))
        reference_increment_normalizer = UnitGaussianNormalizer(torch.zeros(2, device=device))
        reference_input_normalizer.mean = reference_stats["input_mean"]
        reference_input_normalizer.std = reference_stats["input_std"]
        reference_increment_normalizer.mean = reference_stats["increment_mean"]
        reference_increment_normalizer.std = reference_stats["increment_std"]

        method_seconds = {}
        predictions = {method: {} for method in METHODS}
        start = time.perf_counter()
        for ratio, trajectories in test_sets.items():
            predictions["single_nu_delta_physics"][ratio] = rollout(
                reference, trajectories[:, 0], args.eval_steps, "delta_physics",
                ratio * args.reference_viscosity, args,
                reference_input_normalizer, reference_increment_normalizer,
            )
        method_seconds["single_nu_delta_physics"] = time.perf_counter() - start
        start = time.perf_counter()
        for ratio, trajectories in test_sets.items():
            predictions["multi_nu_conditioned"][ratio] = conditioned_rollout(
                conditioned, trajectories[:, 0], args.eval_steps, ratio,
                input_normalizer, increment_normalizer, q_mean, q_std,
            )
        method_seconds["multi_nu_conditioned"] = time.perf_counter() - start

        parameter_counts = {
            "single_nu_delta_physics": sum(p.numel() for p in reference.parameters()),
            "multi_nu_conditioned": sum(p.numel() for p in conditioned.parameters()),
        }
        for method in METHODS:
            for ratio, trajectories in test_sets.items():
                rows.append({
                    "seed": seed,
                    "method": method,
                    "viscosity_ratio": ratio,
                    "viscosity": ratio * args.reference_viscosity,
                    "parameter_count": parameter_counts[method],
                    "training_viscosity_count": 1 if method.startswith("single") else len(args.train_ratios),
                    "training_trajectories": args.n_train_per_viscosity if method.startswith("single") else args.n_train_per_viscosity * len(args.train_ratios),
                    "training_pairs": args.n_train_per_viscosity * args.train_steps if method.startswith("single") else args.n_train_per_viscosity * len(args.train_ratios) * args.train_steps,
                    "train_seconds": train_seconds if method == "multi_nu_conditioned" else float("nan"),
                    "inference_seconds": method_seconds[method],
                    "best_val_rel_l2": best_val if method == "multi_nu_conditioned" else float("nan"),
                    "best_val_epoch": best_epoch if method == "multi_nu_conditioned" else float("nan"),
                    **extended_metrics(predictions[method][ratio], trajectories),
                })
        cost_rows.append({
            "seed": seed,
            "data_seconds_conditioned": data_seconds,
            "train_seconds_conditioned": train_seconds,
            "inference_seconds_delta_all_tests": method_seconds["single_nu_delta_physics"],
            "inference_seconds_conditioned_all_tests": method_seconds["multi_nu_conditioned"],
        })
        write_csv(out_dir / "seed_results.csv", rows)
        write_csv(out_dir / "costs.csv", cost_rows)

    aggregates = aggregate(rows)
    comparisons = paired_comparison(rows)
    write_csv(out_dir / "aggregate_metrics.csv", aggregates)
    write_csv(out_dir / "paired_comparison.csv", comparisons)
    plot_comparison(out_dir / "conditioned_vs_delta.png", aggregates)
    (out_dir / "seed_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "paired_comparison.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
