"""Materialize synthetic data using the frozen experiment configurations.

These are regenerated arrays; original experiments generated data in memory.
Run from the repository root. Each NPZ is one split/viscosity/seed, below the
GitHub single-file limit. Metadata records RNG seeds and solver settings.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fno_dual.data import generate_burgers_trajectories, generate_navier_stokes_trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/trajectories')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    entries = []

    def export(name, generator, count, grid, nu, steps, dt, solver_dt, seed, train_count=None):
        path = out / (name + '.npz')
        if not path.exists():
            values = generator(count, grid, nu, steps, dt, solver_dt, seed, 'cpu').numpy()
            if train_count is None:
                np.savez_compressed(path, trajectories=values)
            else:
                np.savez_compressed(path, train=values[:train_count], validation=values[train_count:])
        entries.append(dict(file=path.name, n_trajectories=count, grid=grid,
                            viscosity=nu, steps=steps, step_dt=dt, solver_dt=solver_dt,
                            rng_seed=seed, train_count=train_count,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (out / 'manifest.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
        print(name, flush=True)

    ns = json.loads(Path('results/ffno_ns64_conditioned_multinu_3seed/config.json').read_text())
    for seed in ns['seeds']:
        for index, ratio in enumerate(ns['train_ratios']):
            export(f'ns64_train_seed{seed}_ratio{ratio:g}', generate_navier_stokes_trajectories,
                   ns['n_train_per_viscosity'] + ns['n_val_per_viscosity'], ns['grid'],
                   ratio * ns['reference_viscosity'], ns['train_steps'], ns['step_dt'],
                   ns['solver_dt'], seed + 100 * index, ns['n_train_per_viscosity'])
        for ratio in ns['test_ratios']:
            export(f'ns64_test_seed{seed}_ratio{ratio:g}', generate_navier_stokes_trajectories,
                   ns['n_test'], ns['grid'], ratio * ns['reference_viscosity'],
                   ns['eval_steps'], ns['step_dt'], ns['solver_dt'], 1000 + seed)
    burgers = json.loads(Path('results/burgers_high_viscosity_3seed/config.json').read_text())
    for seed in burgers['seeds']:
        export(f'burgers_train_seed{seed}', generate_burgers_trajectories,
               burgers['n_train'] + burgers['n_val'], burgers['grid'], burgers['reference_viscosity'],
               burgers['train_steps'], burgers['step_dt'], burgers['solver_dt'], seed, burgers['n_train'])
        for ratio in burgers['viscosity_ratios']:
            export(f'burgers_test_seed{seed}_ratio{ratio:g}', generate_burgers_trajectories,
                   burgers['n_test'], burgers['grid'], ratio * burgers['reference_viscosity'],
                   burgers['eval_steps'], burgers['step_dt'], burgers['solver_dt'], 1000 + seed)
    dense = json.loads(Path('results/ffno_dense_viscosity_scan/config.json').read_text())
    export('dense_ns24_reference', generate_navier_stokes_trajectories,
           dense['n_reference_train'], dense['grid'], dense['reference_viscosity'],
           dense['normalizer_steps'], dense['step_dt'], dense['solver_dt'], dense['seed'])
    for ratio in dense['viscosity_ratios']:
        export(f'dense_ns24_test_ratio{ratio:g}', generate_navier_stokes_trajectories,
               dense['n_test'], dense['grid'], ratio * dense['reference_viscosity'],
               dense['steps'], dense['step_dt'], dense['solver_dt'], 1000 + dense['seed'])


if __name__ == '__main__':
    main()
