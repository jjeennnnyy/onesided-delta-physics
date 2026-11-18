"""Iterative equation-recast baseline for frozen Increment F-FNO checkpoints."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from fno_dual.data import generate_navier_stokes_trajectories
from fno_dual.ffno import FFNO2d
from fno_dual.normalizer import UnitGaussianNormalizer
from fno_dual.physics import spectral_laplacian
from run_ffno_dense_viscosity_scan import (
    extended_metrics,
    model_inputs,
    predict_next,
)


METHODS = ("increment", "delta_physics", "equation_recast")
HORIZONS = (1, 10, 20, 50)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reference_step(model, state, input_normalizer, increment_normalizer):
    increment = increment_normalizer.decode(
        model(model_inputs(state, input_normalizer))
    )
    return state + increment


@torch.no_grad()
def recast_predict_next(
    model,
    state,
    viscosity,
    reference_viscosity,
    delta_t,
    input_normalizer,
    increment_normalizer,
    tolerance=1e-5,
    max_iterations=30,
):
    """Solve y=T0(x+dt*(nu-nu0)*Laplacian(y)) with vector Aitken relaxation."""
    model.eval()
    batch = state.shape[0]
    prediction = reference_step(
        model, state, input_normalizer, increment_normalizer
    )
    nn_evaluations = torch.ones(batch, dtype=torch.int64, device=state.device)
    iterations = torch.zeros(batch, dtype=torch.int64, device=state.device)
    converged = torch.ones(batch, dtype=torch.bool, device=state.device)
    failed_nonfinite = torch.zeros(batch, dtype=torch.bool, device=state.device)
    final_residual = torch.zeros(batch, dtype=state.dtype, device=state.device)
    final_alpha = torch.ones(batch, dtype=state.dtype, device=state.device)

    if viscosity == reference_viscosity:
        return prediction, {
            "iterations": iterations,
            "nn_evaluations": nn_evaluations,
            "converged": converged,
            "failed_nonfinite": failed_nonfinite,
            "relative_residual": final_residual,
            "aitken_alpha": final_alpha,
        }

    converged.zero_()
    active = torch.ones(batch, dtype=torch.bool, device=state.device)
    previous_residual = torch.zeros_like(state)
    has_previous = torch.zeros(batch, dtype=torch.bool, device=state.device)
    alpha = torch.ones(batch, dtype=state.dtype, device=state.device)
    viscosity_shift = delta_t * (viscosity - reference_viscosity)
    tiny = torch.finfo(torch.float64).eps

    for iteration in range(1, max_iterations + 1):
        active_index = active.nonzero(as_tuple=False).squeeze(-1)
        if active_index.numel() == 0:
            break
        current = prediction[active_index]
        effective_source = (
            state[active_index]
            + viscosity_shift * spectral_laplacian(current)
        )
        proposal = reference_step(
            model, effective_source, input_normalizer, increment_normalizer
        )
        nn_evaluations[active_index] += 1
        residual = proposal - current
        residual_norm = torch.linalg.vector_norm(
            residual.reshape(residual.shape[0], -1).double(), dim=1
        )
        proposal_norm = torch.linalg.vector_norm(
            proposal.reshape(proposal.shape[0], -1).double(), dim=1
        )
        relative = residual_norm / (proposal_norm + 1e-12)
        finite = torch.isfinite(relative)
        local_converged = finite & (relative <= tolerance)

        global_converged = active_index[local_converged]
        if global_converged.numel():
            prediction[global_converged] = proposal[local_converged]
            iterations[global_converged] = iteration
            converged[global_converged] = True
            final_residual[global_converged] = relative[local_converged].to(state.dtype)
            active[global_converged] = False

        global_nonfinite = active_index[~finite]
        if global_nonfinite.numel():
            iterations[global_nonfinite] = iteration
            failed_nonfinite[global_nonfinite] = True
            final_residual[global_nonfinite] = float("inf")
            active[global_nonfinite] = False

        update_mask = finite & ~local_converged
        update_index = active_index[update_mask]
        if update_index.numel() == 0:
            continue
        update_residual = residual[update_mask]
        update_alpha = alpha[update_index]
        with_previous = has_previous[update_index]
        if with_previous.any():
            selected = with_previous.nonzero(as_tuple=False).squeeze(-1)
            global_selected = update_index[selected]
            old_residual = previous_residual[global_selected]
            new_residual = update_residual[selected]
            difference = new_residual - old_residual
            old_flat = old_residual.reshape(old_residual.shape[0], -1).double()
            difference_flat = difference.reshape(difference.shape[0], -1).double()
            denominator = (difference_flat * difference_flat).sum(dim=1)
            numerator = (old_flat * difference_flat).sum(dim=1)
            valid_denominator = denominator > tiny
            candidate = update_alpha[selected].double()
            candidate[valid_denominator] = (
                -candidate[valid_denominator]
                * numerator[valid_denominator]
                / denominator[valid_denominator]
            )
            update_alpha[selected] = candidate.to(state.dtype)

        finite_alpha = torch.isfinite(update_alpha)
        bad_alpha_index = update_index[~finite_alpha]
        if bad_alpha_index.numel():
            iterations[bad_alpha_index] = iteration
            failed_nonfinite[bad_alpha_index] = True
            final_residual[bad_alpha_index] = float("inf")
            active[bad_alpha_index] = False

        good_index = update_index[finite_alpha]
        if good_index.numel():
            good_residual = update_residual[finite_alpha]
            good_alpha = update_alpha[finite_alpha]
            updated = prediction[good_index] + good_alpha[:, None, None] * good_residual
            finite_updated = torch.isfinite(updated.reshape(updated.shape[0], -1)).all(dim=1)
            accepted_index = good_index[finite_updated]
            rejected_index = good_index[~finite_updated]
            if accepted_index.numel():
                accepted_residual = good_residual[finite_updated]
                prediction[accepted_index] = updated[finite_updated]
                previous_residual[accepted_index] = accepted_residual
                has_previous[accepted_index] = True
                alpha[accepted_index] = good_alpha[finite_updated]
                final_alpha[accepted_index] = good_alpha[finite_updated]
                final_residual[accepted_index] = relative[update_mask][finite_alpha][finite_updated].to(state.dtype)
                iterations[accepted_index] = iteration
            if rejected_index.numel():
                iterations[rejected_index] = iteration
                failed_nonfinite[rejected_index] = True
                final_residual[rejected_index] = float("inf")
                active[rejected_index] = False

    return prediction, {
        "iterations": iterations,
        "nn_evaluations": nn_evaluations,
        "converged": converged,
        "failed_nonfinite": failed_nonfinite,
        "relative_residual": final_residual,
        "aitken_alpha": final_alpha,
    }


@torch.no_grad()
def recast_rollout(
    model,
    initial_state,
    steps,
    viscosity,
    args,
    input_normalizer,
    increment_normalizer,
):
    state = initial_state
    predictions = []
    diagnostics = []
    for physical_step in range(1, steps + 1):
        state, step_diagnostics = recast_predict_next(
            model=model,
            state=state,
            viscosity=viscosity,
            reference_viscosity=args.reference_viscosity,
            delta_t=args.step_dt,
            input_normalizer=input_normalizer,
            increment_normalizer=increment_normalizer,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )
        predictions.append(state)
        for sample in range(state.shape[0]):
            diagnostics.append({
                "physical_step": physical_step,
                "sample": sample,
                "iterations": int(step_diagnostics["iterations"][sample].item()),
                "nn_evaluations": int(step_diagnostics["nn_evaluations"][sample].item()),
                "converged": bool(step_diagnostics["converged"][sample].item()),
                "failed_nonfinite": bool(step_diagnostics["failed_nonfinite"][sample].item()),
                "relative_residual": float(step_diagnostics["relative_residual"][sample].item()),
                "aitken_alpha": float(step_diagnostics["aitken_alpha"][sample].item()),
            })
    return torch.stack(predictions, dim=1), diagnostics


def load_reference(seed, checkpoint_dir, args, device):
    model = FFNO2d(
        width=args.width, modes=args.modes, depth=args.depth, factor=args.factor
    ).to(device)
    seed_dir = checkpoint_dir / f"seed_{seed}"
    model.load_state_dict(torch.load(
        seed_dir / "increment.pt", map_location=device, weights_only=True
    ))
    statistics = torch.load(
        seed_dir / "normalizers.pt", map_location=device, weights_only=True
    )
    input_normalizer = UnitGaussianNormalizer(torch.zeros(2, device=device))
    increment_normalizer = UnitGaussianNormalizer(torch.zeros(2, device=device))
    input_normalizer.mean = statistics["input_mean"]
    input_normalizer.std = statistics["input_std"]
    increment_normalizer.mean = statistics["increment_mean"]
    increment_normalizer.std = statistics["increment_std"]
    return model, input_normalizer, increment_normalizer


@torch.no_grad()
def direct_rollout(
    model, initial_state, steps, method, viscosity, args,
    input_normalizer, increment_normalizer,
):
    state = initial_state
    predictions = []
    for _ in range(steps):
        state = predict_next(
            model, state, method, viscosity, args.reference_viscosity,
            args.step_dt, input_normalizer, increment_normalizer,
        )
        predictions.append(state)
    return torch.stack(predictions, dim=1)


def aggregate_rows(seed_rows):
    excluded = {"seed", "method", "viscosity_ratio", "viscosity"}
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


def paired_comparison_rows(seed_rows):
    metric_names = [f"rel_l2_t{step}" for step in HORIZONS] + [
        "rollout_rel_l2_average", "rollout_psre_average", "rollout_psre_final"
    ]
    output = []
    for ratio in sorted({row["viscosity_ratio"] for row in seed_rows}):
        for baseline_name in ("increment", "delta_physics"):
            paired = []
            for seed in sorted({row["seed"] for row in seed_rows}):
                baseline = next(
                    row for row in seed_rows
                    if row["seed"] == seed and row["viscosity_ratio"] == ratio
                    and row["method"] == baseline_name
                )
                recast = next(
                    row for row in seed_rows
                    if row["seed"] == seed and row["viscosity_ratio"] == ratio
                    and row["method"] == "equation_recast"
                )
                paired.append({
                    name: 100.0 * (baseline[name] - recast[name]) / baseline[name]
                    for name in metric_names
                })
            row = {
                "viscosity_ratio": ratio,
                "baseline": baseline_name,
                "n_seeds": len(paired),
            }
            for name in metric_names:
                values = torch.tensor([item[name] for item in paired], dtype=torch.float64)
                row[f"recast_improvement_{name}_percent_mean"] = values.mean().item()
                row[f"recast_improvement_{name}_percent_std"] = values.std(unbiased=True).item()
            output.append(row)
    return output


def plot_comparison(path, aggregate_metrics):
    import matplotlib.pyplot as plt

    labels = {
        "increment": "Increment",
        "delta_physics": "One-sided Delta-Physics",
        "equation_recast": "Iterative Equation Recast",
    }
    ratios = sorted({row["viscosity_ratio"] for row in aggregate_metrics})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for method in METHODS:
        selected = [
            next(row for row in aggregate_metrics
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
    fig.suptitle("64 x 64 Navier-Stokes zero-shot baselines (3 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/ffno_ns64_equation_recast_3seed")
    parser.add_argument("--checkpoint-dir", default="results/ffno_ns64_high_viscosity_3seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--step-dt", type=float, default=0.05)
    parser.add_argument("--solver-dt", type=float, default=0.005)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--reference-viscosity", type=float, default=1e-3)
    parser.add_argument("--viscosity-ratios", nargs="+", type=float, default=[1.0, 1.5, 1.75, 2.0])
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--max-iterations", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {**vars(args), "device": device, "methods": METHODS}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True)

    rows = []
    all_diagnostics = []
    for seed in args.seeds:
        model, input_normalizer, increment_normalizer = load_reference(
            seed, checkpoint_dir, args, device
        )
        test_sets = {
            ratio: generate_navier_stokes_trajectories(
                args.n_test, args.grid, ratio * args.reference_viscosity,
                args.eval_steps, args.step_dt, args.solver_dt, 1000 + seed, device,
            )
            for ratio in args.viscosity_ratios
        }
        for ratio, trajectories in test_sets.items():
            predictions = {}
            for method in METHODS:
                start = time.perf_counter()
                if method == "equation_recast":
                    prediction, diagnostics = recast_rollout(
                        model, trajectories[:, 0], args.eval_steps,
                        ratio * args.reference_viscosity, args,
                        input_normalizer, increment_normalizer,
                    )
                    for item in diagnostics:
                        item.update({"seed": seed, "viscosity_ratio": ratio})
                    all_diagnostics.extend(diagnostics)
                    mean_iterations = sum(item["iterations"] for item in diagnostics) / len(diagnostics)
                    mean_nn_evaluations = sum(item["nn_evaluations"] for item in diagnostics) / len(diagnostics)
                    nonconvergence_count = sum(not item["converged"] for item in diagnostics)
                    nonfinite_count = sum(item["failed_nonfinite"] for item in diagnostics)
                else:
                    prediction = direct_rollout(
                        model, trajectories[:, 0], args.eval_steps, method,
                        ratio * args.reference_viscosity, args,
                        input_normalizer, increment_normalizer,
                    )
                    mean_iterations = 0.0
                    mean_nn_evaluations = 1.0
                    nonconvergence_count = 0
                    nonfinite_count = 0
                wall_clock = time.perf_counter() - start
                predictions[method] = prediction
                rows.append({
                    "seed": seed,
                    "method": method,
                    "viscosity_ratio": ratio,
                    "viscosity": ratio * args.reference_viscosity,
                    "wall_clock_seconds": wall_clock,
                    "mean_iterations_per_physical_step": mean_iterations,
                    "mean_nn_evaluations_per_physical_step": mean_nn_evaluations,
                    "nonconvergence_count": nonconvergence_count,
                    "nonfinite_count": nonfinite_count,
                    **extended_metrics(prediction, trajectories),
                })
            print({
                "seed": seed,
                "ratio": ratio,
                "iid_max_difference": float(
                    (predictions["increment"] - predictions["equation_recast"]).abs().max().item()
                ) if ratio == 1.0 else None,
                "recast_nonconvergence": nonconvergence_count,
                "recast_mean_iterations": mean_iterations,
            }, flush=True)
            write_csv(out_dir / "seed_results.csv", rows)
            write_csv(out_dir / "iteration_diagnostics.csv", all_diagnostics)

    aggregates = aggregate_rows(rows)
    comparisons = paired_comparison_rows(rows)
    write_csv(out_dir / "aggregate_metrics.csv", aggregates)
    write_csv(out_dir / "paired_comparisons.csv", comparisons)
    plot_comparison(out_dir / "equation_recast_comparison.png", aggregates)
    (out_dir / "seed_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
