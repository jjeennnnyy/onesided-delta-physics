# Publication export validation

- All 16 bundled tests passed (`python -m pytest tests -q`). The only warning was the existing PyTorch `weight_norm` deprecation warning.
- The copied manuscript compiled successfully with both pdfLaTeX and XeLaTeX, with BibTeX and two subsequent LaTeX passes. The final XeLaTeX log had no undefined or multiply defined references. The original compiled manuscript PDF is retained rather than replaced by the validation build.
- Original model implementations, experiment runners, checkpoints, normalization files, result tables and figures were copied from the research workspace; no training was performed for this export.
- Raw trajectories were not cached by the original runners. Publication-export arrays are regenerated from the same solvers, configurations and random seeds. Their integrity and normalization checks are recorded separately in `data/validation.json`.
- `MANIFEST.sha256` inventories the released files. Run `python verify_artifacts.py` after downloading a complete snapshot. Text files use LF line endings for portable checksum verification.

This validates artifact packaging and reproducibility dependencies, not scientific novelty or independent replication of the reported experiments.
