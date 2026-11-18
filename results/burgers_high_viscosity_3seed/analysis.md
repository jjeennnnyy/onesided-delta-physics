# Viscous Burgers: three-seed high-viscosity evaluation

## Protocol

- PDE: periodic one-dimensional viscous Burgers equation.
- Reference viscosity `nu0=0.01`.
- Seeds 0, 1, 2; each seed independently regenerates data and trains a new
  72,033-parameter Increment FNO.
- 24 train and 6 validation trajectories, 20 training steps, 15 epochs.
- Eight independent 50-step test trajectories per viscosity and seed.
- Test ratios `nu/nu0={1, 1.25, 1.5, 1.75, 2}`.
- The same checkpoint is evaluated as Increment and as one-sided Delta-Physics
  with the exact 1-D correction `(S_nu-S_nu0)u_t`.
- Improvements are paired within seed and reported as mean ± sample standard
  deviation over three seeds.

Best validation Increment RelL2 values were `0.004774`, `0.003101`, and
`0.003683`; all selected checkpoints were epoch 15.

## Paired RelL2 improvement over Increment

| nu/nu0 | Step 1 | Step 5 | Step 10 | Step 20 | Step 50 | Avg 20 | Avg 50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
| 1.25 | +46.56±5.24 | +63.98±2.65 | +70.09±0.78 | +74.23±3.44 | +68.51±9.29 | +70.33±1.28 | +71.20±6.09 |
| 1.5 | +63.67±4.33 | +77.79±1.04 | +83.31±1.42 | +86.82±2.51 | +83.89±4.26 | +83.51±1.63 | +85.07±3.38 |
| 1.75 | +69.02±3.69 | +82.44±0.83 | +87.74±1.34 | +91.06±1.85 | +89.30±2.05 | +87.93±1.44 | +89.87±2.04 |
| 2.0 | **+71.19±3.26** | **+84.97±0.78** | **+90.08±1.22** | **+93.18±1.41** | **+91.87±0.92** | **+90.22±1.26** | **+92.23±1.28** |

## Absolute RelL2, mean ± std

| nu/nu0 | Method | Step 1 | Step 5 | Step 10 | Step 20 | Step 50 | Avg 50 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1.0 | Increment / Delta | .00647±.00050 | .01483±.00132 | .01961±.00133 | .02566±.00220 | .05173±.01146 | .03053±.00381 |
| 1.25 | Increment | .01197±.00046 | .03853±.00352 | .05735±.00489 | .08260±.01168 | .13899±.03389 | .08981±.01618 |
|  | Delta | .00638±.00048 | .01382±.00060 | .01718±.00188 | .02114±.00289 | .04250±.01252 | .02550±.00528 |
| 1.5 | Increment | .02037±.00078 | .07028±.00735 | .10787±.01018 | .15639±.02190 | .26181±.06694 | .16909±.03056 |
|  | Delta | .00738±.00062 | .01556±.00086 | .01797±.00196 | .02036±.00306 | .04091±.00983 | .02475±.00466 |
| 1.75 | Increment | .02910±.00120 | .10137±.01119 | .15729±.01606 | .23149±.03398 | .39896±.11298 | .25287±.04943 |
|  | Delta | .00899±.00077 | .01774±.00111 | .01919±.00176 | .02036±.00270 | .04174±.00892 | .02506±.00350 |
| 2.0 | Increment | .03781±.00166 | .13151±.01504 | .20551±.02241 | .30780±.04867 | .55071±.17208 | .34107±.07368 |
|  | Delta | .01086±.00088 | .01970±.00128 | .02023±.00151 | .02058±.00204 | .04437±.01199 | .02593±.00262 |

## PSRE improvement

| nu/nu0 | Average first 20 | Average 50 | Final step |
|---:|---:|---:|---:|
| 1.0 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
| 1.25 | +77.41±6.60 | +75.06±12.21 | +70.74±16.20 |
| 1.5 | +89.49±2.69 | +88.46±5.54 | +86.84±8.05 |
| 1.75 | +93.17±1.50 | +92.80±3.27 | +92.25±4.77 |
| 2.0 | **+94.90±1.09** | **+94.81±2.29** | **+94.70±3.09** |

At ratio 2, average-50 PSRE is `0.5608±0.2912` for Increment and
`0.02484±0.00492` for Delta-Physics; final PSRE is `1.1645±0.8532` versus
`0.04655±0.01401`.

## Interpretation

The Navier--Stokes pattern transfers to a second dissipative PDE and is much
stronger on viscous Burgers:

1. IID forecasts remain exactly unchanged.
2. Every high-viscosity ratio improves at every reported horizon.
3. Improvement generally increases with viscosity shift.
4. Long-horizon gains exceed short-horizon gains; at ratio 2, Step-1 improves
   `71.19±3.26%` and Step-50 improves `91.87±0.92%`.
5. Spectral errors fall by roughly 95% at ratio 2.

The absolute corrected errors are nearly viscosity invariant. Delta average-50
RelL2 stays around `0.025--0.026` across ratios 1.25--2, while Increment grows
from `0.0898` to `0.3411`. This is direct evidence that the analytic correction
removes a parameter-dependent dissipative shift that a reference-viscosity
neural time-stepper otherwise cannot represent zero-shot.

## Conclusion

The mechanism is not specific to 2-D Navier--Stokes. Across NS and viscous
Burgers, the same zero-parameter, no-retraining correction preserves reference
behavior exactly and improves unseen higher-viscosity rollouts increasingly
with parameter shift and horizon. The supported claim is therefore a class of
one-sided zero-shot corrections for dissipative neural time-steppers with a
known linear semigroup.
