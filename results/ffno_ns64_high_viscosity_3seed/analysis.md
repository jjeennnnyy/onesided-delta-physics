# 64 x 64 Navier--Stokes: frozen one-sided Delta-Physics

## Protocol

- PDE: forced two-dimensional Navier--Stokes in vorticity form.
- Resolution: `64 x 64`; reference viscosity: `nu0=1e-3`.
- Test ratios: `nu/nu0={1.0, 1.5, 1.75, 2.0}`.
- Seeds 0, 1, and 2 independently regenerate data and train an Increment
  F-FNO checkpoint.
- Per seed: 24 training, 6 validation, and 8 independent test trajectories per
  viscosity; 20 training steps and 50-step free-running evaluation.
- Frozen configuration: width 16, 8 modes, 8 F-FNO blocks, factor 4, 15 epochs.
- The same checkpoint and test trajectories are evaluated as Increment and
  one-sided Delta-Physics. There is no new learned parameter or retraining.
- Improvements are paired within seed, then reported as mean +/- sample
  standard deviation over three seeds.

Selected validation RelL2 values were `0.008603`, `0.007232`, and `0.008025`.
All best checkpoints were at epoch 15.

## Paired RelL2 improvement over Increment

| nu/nu0 | Step 1 | Step 5 | Step 10 | Step 20 | Step 50 | Avg 20 | Avg 50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 |
| 1.5 | +2.35+/-1.34 | +4.50+/-1.16 | +6.62+/-0.98 | +9.13+/-0.83 | +12.46+/-1.44 | +7.47+/-0.86 | +10.52+/-1.01 |
| 1.75 | +8.31+/-1.66 | +10.86+/-1.48 | +13.06+/-1.38 | +15.17+/-1.42 | +17.49+/-2.17 | +13.74+/-1.32 | +15.97+/-1.67 |
| 2.0 | **+15.71+/-1.72** | **+18.28+/-1.58** | **+20.14+/-1.66** | **+21.31+/-1.94** | **+22.19+/-2.82** | **+20.46+/-1.69** | **+21.33+/-2.27** |

Both requested trends hold without exception in the aggregate results:

1. At every fixed horizon, improvement increases as the viscosity shift grows.
2. At every non-reference viscosity, improvement increases from Step 1 through
   Step 50.

## Absolute RelL2 means

| nu/nu0 | Method | Step 1 | Step 20 | Step 50 | Avg 50 |
|---:|---|---:|---:|---:|---:|
| 1.0 | Increment / Delta | 0.00898 | 0.14993 | 0.40913 | 0.19647 |
| 1.5 | Increment | 0.00911 | 0.15581 | 0.43308 | 0.20605 |
|  | Delta-Physics | 0.00890 | 0.14157 | 0.37956 | 0.18445 |
| 1.75 | Increment | 0.00966 | 0.16382 | 0.45106 | 0.21566 |
|  | Delta-Physics | 0.00887 | 0.13895 | 0.37283 | 0.18135 |
| 2.0 | Increment | 0.01047 | 0.17414 | 0.47211 | 0.22754 |
|  | Delta-Physics | 0.00884 | 0.13697 | 0.36816 | 0.17915 |

Increment error grows steadily with viscosity. In contrast, the corrected
Step-1, Step-20, Step-50, and average-50 errors all decrease slightly from
ratio 1.5 to 2.0. This indicates that the analytic term removes the dominant
viscosity-dependent dissipative shift rather than acting as a generic rollout
regularizer.

## Spectral error improvement

| nu/nu0 | Avg first 20 PSRE | Avg 50 PSRE | Final PSRE |
|---:|---:|---:|---:|
| 1.0 | 0.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 |
| 1.5 | +28.05+/-11.58 | +36.07+/-15.34 | +43.13+/-22.13 |
| 1.75 | +50.65+/-4.96 | +57.73+/-6.30 | +62.95+/-11.52 |
| 2.0 | **+63.34+/-3.40** | **+68.97+/-3.21** | **+73.01+/-7.38** |

At ratio 2, average-50 PSRE falls from `0.12306` to `0.03757`, and final-step
PSRE falls from `0.25576` to `0.06598`.

## IID identity and conclusion

For every seed, the maximum elementwise difference between Increment and
Delta-Physics at `nu=nu0` is exactly `0.0` over the complete 50-step rollout.

The frozen method preserves the reference operator exactly while recovering
both target signatures at 64 x 64 resolution: gains grow monotonically with
the viscosity shift and with rollout horizon. Together with the earlier
low-resolution Navier--Stokes and viscous Burgers experiments, this supports
one-sided Delta-Physics as a zero-parameter, no-retraining correction for
higher-dissipation zero-shot transfer.
