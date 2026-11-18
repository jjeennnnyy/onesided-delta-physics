# Paper-ready figures and tables

All figures are provided as 320-DPI PNG, vector PDF, and editable SVG.
Tables are provided as CSV, Markdown, and LaTeX (booktabs + graphicx).

## Main-text figures

1. `figure_1_method`: One-sided Delta-Physics overview. The policy depends only
   on known `(nu, nu0)`; the current state separately enters the frozen
   time-stepper and analytical correction.
2. `figure_2_cross_pde_improvement`: NS/Burgers improvement curves with separate
   y-axis ranges, mean +/- sample standard deviation over three seeds.
3. `figure_3_ns_spectral_mechanism`: Step-50 radial spectrum and spectral-error
   contribution for 64 x 64 NS at `nu/nu0=2` (24 trajectories pooled across
   three seeds).
4. `figure_4_accuracy_cost`: accuracy versus training trajectories and NN
   evaluations per physical step at `nu/nu0=2`.

## Main-text tables

1. `table_1_ns64_main`: compact NS results grouped by single-viscosity zero-shot
   versus multi-viscosity supervised regimes.
2. `table_2_burgers_main`: compact Burgers results.
3. `table_3_data_compute`: target-parameter supervision, data, optimizer time,
   parameters, NN evaluations, and CPU inference time.

## Supplementary assets

- `figure_s1_dense_two_sided_ablation`: full-range instability and boundary zoom
  explaining the one-sided switch.
- `table_1_ns64_absolute` and `table_2_burgers_absolute`: full horizons.
- `table_s1_recast_convergence`: fixed-point convergence diagnostics.
- `figure_3_spectral_data.csv`: cached per-bin statistics used by Figure 3.

Recommended first-use name: **One-sided Delta-Physics, a zero-shot analytical
viscosity correction.** In LaTeX prose, render Delta as `\Delta`.

All reported uncertainty is sample standard deviation over three seeds.
