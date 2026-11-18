# Synthetic trajectory archive

`python export_datasets.py` regenerates the data using the committed experiment configurations. Files in `trajectories/` are regenerated during publication export, not original cached training files. Each array includes the initial state and all subsequent frames; shapes are `(trajectory, time, x, y)` for NS and `(trajectory, time, x)` for Burgers. Training NPZ files have `train` and `validation` keys; test files have `trajectories`.

`trajectories/manifest.json` records the solver settings, RNG seed and SHA-256 of every NPZ. NS64 ratio-1 training data are shared by the single-viscosity and conditioned experiments and the extra training-budget check. All NS64 methods share the test sets. The dense NS24 data belong only to the supplementary scan.

The original runners generate these data internally; they do not read NPZ caches. Exporting the arrays does not alter the frozen models or original metrics. Floating-point differences can occur with different NumPy/library versions. These synthetic benchmark data contain no external/private dataset.
