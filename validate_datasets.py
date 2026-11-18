"""Validate exported arrays and compare NS reference normalization statistics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    torch.set_num_threads(4)
    root = Path('data/trajectories')
    rows = json.loads((root / 'manifest.json').read_text())
    for row in rows:
        path = root / row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
        with np.load(path) as archive:
            count = 0
            for key in archive.files:
                values = archive[key]
                assert np.isfinite(values).all(), (path, key)
                assert values.shape[1] == row['steps'] + 1, path
                assert values.shape[2] == row['grid'], path
                count += values.shape[0]
            assert count == row['n_trajectories'], path
    differences = {}
    for seed in (0, 1, 2):
        train = torch.from_numpy(np.load(root / f'ns64_train_seed{seed}_ratio1.npz')['train'])
        states = train[:, :-1].reshape(-1, 64, 64)
        target = train[:, 1:].reshape(-1, 64, 64)
        stats = torch.load(f'results/ffno_ns64_high_viscosity_3seed/seed_{seed}/normalizers.pt', weights_only=True)
        observed = dict(input_mean=states.mean(), input_std=states.std(),
                        increment_mean=(target-states).mean(), increment_std=(target-states).std())
        differences[str(seed)] = {key: float(abs(observed[key] - stats[key])) for key in observed}
        assert all(torch.allclose(observed[key], stats[key], atol=1e-7, rtol=1e-6) for key in observed)
    report = dict(archives_checked=len(rows), all_finite=True,
                  ns_reference_normalizer_absolute_differences=differences,
                  note='Small reduction-order roundoff is permitted; this is not a comparison to unavailable original cached trajectories.')
    Path('data/validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
