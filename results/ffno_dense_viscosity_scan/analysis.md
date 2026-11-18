# Dense viscosity scan: Increment F-FNO vs Delta-Physics

## Protocol

- One frozen Increment F-FNO checkpoint; no retraining or parameter changes.
- Delta-Physics uses the fixed additive correction
  `(S_nu - S_nu0) omega_t` at every rollout step.
- Reference viscosity `nu0 = 0.001`.
- Ratios `nu/nu0 = {0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.5, 1.75, 2}`.
- Eight test trajectories per viscosity, common initial-condition seed.
- 50-step free-running rollout. The checkpoint was trained with 20-step
  trajectories; step 50 is an equal long-horizon stress test for both methods.
- Single seed.

Positive percentages below mean lower error for Delta-Physics relative to
Increment.

## Relative-L2 improvement curve

| nu/nu0 | Step 1 | Step 5 | Step 10 | Step 20 | Step 50 | Avg 20 | Avg 50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | +10.53% | +8.23% | +5.28% | -1.77% | **-296.76%** | +2.94% | -73.21% |
| 0.625 | +7.08% | +5.26% | +2.99% | -1.74% | **-52.29%** | +1.40% | -17.60% |
| 0.75 | +4.05% | +2.78% | +1.25% | -1.65% | -14.53% | +0.27% | -6.74% |
| 0.875 | +1.63% | +0.97% | +0.21% | -1.12% | -5.06% | -0.24% | -2.83% |
| 1.0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 1.25 | -0.42% | +0.89% | +2.24% | +4.12% | +7.57% | +2.87% | +5.68% |
| 1.5 | +2.97% | +5.30% | +7.51% | +10.15% | +14.21% | +8.38% | +11.94% |
| 1.75 | +9.15% | +12.06% | +14.58% | +17.05% | +20.37% | +15.36% | +18.42% |
| 2.0 | +16.65% | +19.77% | +22.17% | +23.97% | **+26.07%** | +22.68% | +24.71% |

IID rollouts are elementwise identical, with maximum difference exactly zero.

## Spectral-error averages

| nu/nu0 | Avg-20 PSRE improvement | Avg-50 PSRE improvement |
|---:|---:|---:|
| 0.5 | +23.10% | -9.72% |
| 0.625 | +13.59% | +14.40% |
| 0.75 | +4.38% | +4.41% |
| 0.875 | -1.00% | -3.21% |
| 1.0 | 0.00% | 0.00% |
| 1.25 | +20.89% | +24.91% |
| 1.5 | +43.15% | +50.17% |
| 1.75 | +57.34% | +64.31% |
| 2.0 | +66.47% | +72.72% |

## Regime map

Delta-Physics is strongly asymmetric.

- **Higher viscosity (`nu > nu0`)**: the correction is consistently useful.
  Except for a negligible -0.42% one-step change at ratio 1.25, gains increase
  with both viscosity distance and rollout horizon. At ratio 2, final-step gain
  grows from 16.65% at step 1 to 26.07% at step 50.
- **Reference viscosity (`nu = nu0`)**: strict equivalence to Increment.
- **Lower viscosity (`nu < nu0`)**: the correction is only a short-horizon
  aid. Every sampled lower viscosity is worse by step 20, and degradation grows
  rapidly thereafter. Ratio 0.5 becomes unstable relative to Increment by step
  50, with a 296.76% final-step regression.

The useful boundary is therefore not simply viscosity distance. It is the sign
of the shift and the rollout horizon. Analytically adding missing dissipation
is stable and increasingly valuable; analytically restoring dissipation that
the reference model has implicitly learned behaves like accumulated
anti-diffusive compensation and eventually destabilizes the recurrent state.

## Decision

The scan does not support presenting Delta-Physics as a two-sided viscosity OOD
solution. It supports a narrower and stronger claim: **Frozen Increment F-FNO
plus Delta-Physics is a zero-parameter high-viscosity adaptation and long-horizon
stabilization method.**

For `nu < nu0`, Increment should remain the default unless a separate method is
developed and validated. A practical one-sided policy—apply Delta-Physics only
when `nu > nu0`, otherwise use Increment—preserves IID/lower-viscosity behavior
and captures the demonstrated higher-viscosity gains without training or new
parameters. Multi-seed validation should test this one-sided formulation.
