"""Paired training-budget screening report; does not edit the manuscript."""

import csv
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results/ffno_ns64_high_viscosity_3seed"
OUT = ROOT / "results/ffno_ns64_budget75_seed0"


def main():
    old = json.loads((BASE / "seed_results.json").read_text())
    new = json.loads((OUT / "seed_results.json").read_text())
    old = [r for r in old if r["seed"] == 0]
    norms_old = torch.load(BASE / "seed_0/normalizers.pt", weights_only=True)
    norms_new = torch.load(OUT / "seed_0/normalizers.pt", weights_only=True)
    normalizer_check = {
        key: {"exact_equal": torch.equal(norms_old[key], norms_new[key]),
              "max_abs_difference": (norms_old[key] - norms_new[key]).abs().max().item()}
        for key in norms_old
    }
    metric_names = ["rel_l2_t1", "rel_l2_t10", "rel_l2_t20", "rel_l2_t50",
                    "rollout_rel_l2_average", "rollout_psre_average", "rollout_psre_final"]
    rows = []
    for epochs, source in ((15, old), (75, new)):
        for ratio in (1.0, 1.5, 1.75, 2.0):
            inc = next(r for r in source if r["viscosity_ratio"] == ratio
                       and r["method"] == "increment")
            delta = next(r for r in source if r["viscosity_ratio"] == ratio
                         and r["method"] == "delta_physics")
            for metric in metric_names:
                rows.append(dict(epochs=epochs, seed=0, viscosity_ratio=ratio,
                                 metric=metric, increment=inc[metric],
                                 delta_physics=delta[metric],
                                 improvement_percent=100 * (1 - delta[metric] / inc[metric])))
    with (OUT / "budget_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checks = {"normalizers": normalizer_check,
              "reference_metrics_exact_equal": all(
                  r["increment"] == r["delta_physics"] for r in rows
                  if r["viscosity_ratio"] == 1.0)}
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    lines = ["# Single-seed reference training-budget sensitivity", "",
             "Paired seed 0 screening, not a three-seed significance test. "
             "The 15-epoch results are the original seed-0 results, not their three-seed mean.", "",
             "## Protocol", "",
             "- Same NS64 data-generation configuration and seeds: 24 training and 6 validation trajectories "
             "at nu0=1e-3; 480 training pairs. Test seed 1000, eight trajectories per viscosity, 50 steps.",
             "- Same F-FNO initialization seed 0, shuffle seed 10000, architecture, normalization procedure, "
             "batch size 32, Adam lr=1e-3 and weight_decay=1e-6.",
             "- Fresh 75-epoch training with validation-selected checkpoint, matching the original selection rule.",
             "- Original StepLR policy retained: step_size=epochs//2, gamma=0.5. "
             "Thus decay occurs at epochs 7/14 for the original budget and 37/74 for the long budget. "
             "This tests a longer training budget with its original duration-scaled schedule, "
             "not an isolated epoch-count change at identical learning rates.",
             "- 15 epochs: 225 optimizer updates; 75 epochs: 1,125 updates. "
             "The latter matches 15 epochs on the conditioned model's 2,400 training pairs, "
             "but uses only reference-viscosity data.",
             "- Each selected checkpoint is shared by Increment and the parameter-free correction. "
             "No target-viscosity training data or tuning is used.", "",
             "## Training", "",
             "| Budget | Selected epoch | Validation one-step RelL2 | Training wall-clock (s) |",
             "|---:|---:|---:|---:|"]
    for epochs, source in ((15, old), (75, new)):
        r = source[0]
        lines.append(f"| {epochs} | {r['best_val_epoch']} | {r['best_val_rel_l2']:.6f} | {r['train_seconds']:.2f} |")
    lines += ["", "Wall-clock is descriptive only: the runs occurred separately and CPU thread settings may differ.", ""]
    for metric in metric_names:
        lines += [f"## {metric}", "",
                  "| Epochs | nu/nu0 | Increment | Delta-Physics | Improvement (%) |",
                  "|---:|---:|---:|---:|---:|"]
        for r in rows:
            if r["metric"] == metric:
                lines.append(f"| {r['epochs']} | {r['viscosity_ratio']} | {r['increment']:.6f} | "
                             f"{r['delta_physics']:.6f} | {r['improvement_percent']:+.2f} |")
        lines.append("")
    lines += ["## Checks", "", "```json", json.dumps(checks, indent=2), "```", "",
              "Both normalization standard deviations are bitwise identical. The recomputed means differ "
              "by at most 2.84e-10 (input) and 1.57e-11 (increment), consistent with "
              "floating-point reduction differences; normalization statistics are not bitwise identical across runs.", "",
              "## Interpretation", "",
              "The longer budget improves the reference model substantially: its reference-viscosity "
              "Step-50 RelL2 falls from 0.372824 to 0.094524. Yet the correction remains beneficial at "
              "all three higher viscosities. At ratio 2, Avg-50 improvement rises from 23.94% to 62.70%, "
              "and Step-50 improvement rises from 25.23% to 63.26%. The longer-budget model also gains "
              "83.81% in average PSRE and 84.86% in final PSRE at ratio 2.", "",
              "This screening result argues against the gain being solely an artifact of an undertrained "
              "reference model. It also confirms that the original reference model had substantial room "
              "to improve with additional optimization. It does not establish convergence at 75 epochs "
              "or robustness across seeds. The original single- versus multi-viscosity comparison "
              "therefore mixes training-data coverage and optimization-budget effects.", "",
              "Interpret gains relative to the stronger checkpoint's own Increment baseline. "
              "One seed cannot establish robustness across training randomness; "
              "the new results have not been inserted into the paper."]
    (OUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # The reused runner assumes three seeds in its default plot title. Replace
    # that generated plot with an explicitly single-seed, no-error-bar version.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for epochs, style in ((15, "--"), (75, "-")):
        for metric, color, marker, name in (
            ("rollout_rel_l2_average", "tab:blue", "o", "Avg-50"),
            ("rel_l2_t50", "tab:orange", "s", "Step-50"),
        ):
            selected = [r for r in rows if r["epochs"] == epochs and r["metric"] == metric]
            ax.plot([r["viscosity_ratio"] for r in selected],
                    [r["improvement_percent"] for r in selected], linestyle=style,
                    marker=marker, color=color, label=f"{epochs} epochs, {name}")
    ax.set(xlabel="Viscosity ratio", ylabel="Improvement over paired Increment (%)",
           title="Training-budget sensitivity: seed 0 only")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "budget_improvement.png", dpi=180)
    fig.savefig(OUT / "high_viscosity_3seed_curve.png", dpi=180)
    plt.close(fig)
    print(json.dumps(checks, indent=2))
    for r in rows:
        if r["metric"] in ("rel_l2_t50", "rollout_rel_l2_average"):
            print(r)


if __name__ == "__main__":
    main()
