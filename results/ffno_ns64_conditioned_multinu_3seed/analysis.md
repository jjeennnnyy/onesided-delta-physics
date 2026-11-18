# Multi-viscosity conditioned F-FNO baseline

## Experimental protocol

- PDE/resolution: forced Navier--Stokes, `64 x 64`.
- Three independent seeds and 50-step free-running rollouts.
- Test ratios: `nu/nu0={1.0, 1.5, 1.75, 2.0}` with `nu0=1e-3`.
- Single-viscosity method: Increment F-FNO trained only at `nu0`, followed by
  frozen one-sided Delta-Physics at inference.
- Conditioned baseline: the same F-FNO width, modes, depth, optimizer, epochs,
  and per-viscosity trajectory count, with standardized `log(nu/nu0)` as one
  additional constant input channel.
- Conditioned training ratios:
  `nu/nu0={1.0, 1.25, 1.5, 1.75, 2.0}`.
- Consequently, all reported conditioned test viscosities were covered during
  training. This is a deliberately favorable, fully supervised parameter
  baseline rather than a zero-shot viscosity test.
- Both methods are evaluated on identical test trajectories. Values below are
  mean +/- sample standard deviation over three seeds.

## RelL2 results

| nu/nu0 | Method | Step 1 | Step 10 | Step 20 | Step 50 | Avg 50 |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | Single-nu Delta | .00898+/-.00120 | .07923+/-.01009 | .14993+/-.01717 | .40913+/-.06111 | .19647+/-.02527 |
|  | Multi-nu conditioned | **.00226+/-.00029** | **.01804+/-.00240** | **.03051+/-.00410** | **.06163+/-.00547** | **.03504+/-.00394** |
| 1.5 | Single-nu Delta | .00890+/-.00118 | .07579+/-.00900 | .14157+/-.01511 | .37956+/-.06122 | .18445+/-.02419 |
|  | Multi-nu conditioned | **.00205+/-.00024** | **.01477+/-.00157** | **.02357+/-.00234** | **.04922+/-.00592** | **.02733+/-.00287** |
| 1.75 | Single-nu Delta | .00887+/-.00117 | .07446+/-.00856 | .13895+/-.01451 | .37283+/-.06213 | .18135+/-.02412 |
|  | Multi-nu conditioned | **.00209+/-.00020** | **.01454+/-.00119** | **.02268+/-.00157** | **.04798+/-.00383** | **.02637+/-.00189** |
| 2.0 | Single-nu Delta | .00884+/-.00116 | .07332+/-.00818 | .13697+/-.01411 | .36816+/-.06317 | .17915+/-.02419 |
|  | Multi-nu conditioned | **.00223+/-.00017** | **.01523+/-.00092** | **.02341+/-.00093** | **.04941+/-.00109** | **.02716+/-.00076** |

The multi-viscosity baseline is decisively more accurate. Relative to
single-viscosity Delta-Physics, its average-50 RelL2 is lower by
`82.14+/-0.76%`, `85.12+/-1.37%`, `85.36+/-1.33%`, and `84.69+/-1.70%` at
ratios 1.0, 1.5, 1.75, and 2.0 respectively. At ratio 2, Step-50 RelL2 is lower
by `86.36+/-1.89%`.

## Spectral results

| nu/nu0 | Method | Avg-50 PSRE | Final PSRE |
|---:|---|---:|---:|
| 1.0 | Single-nu Delta | .04083+/-.01086 | .07224+/-.01139 |
|  | Multi-nu conditioned | **.01019+/-.00372** | **.02129+/-.00744** |
| 1.5 | Single-nu Delta | .03872+/-.00822 | .06802+/-.00626 |
|  | Multi-nu conditioned | **.00793+/-.00242** | **.01652+/-.00624** |
| 1.75 | Single-nu Delta | .03804+/-.00722 | .06674+/-.00454 |
|  | Multi-nu conditioned | **.00699+/-.00091** | **.01560+/-.00420** |
| 2.0 | Single-nu Delta | .03757+/-.00636 | .06598+/-.00310 |
|  | Multi-nu conditioned | **.00663+/-.00034** | **.01505+/-.00269** |

At ratio 2, conditioned F-FNO reduces average PSRE by `81.89+/-4.05%` and
final PSRE by `77.14+/-4.42%` relative to Delta-Physics.

## Data, parameter, and cost comparison

| Quantity | Single-nu Delta-Physics | Multi-nu conditioned |
|---|---:|---:|
| Training viscosities | 1 | 5 |
| Training trajectories | 24 | 120 |
| One-step training pairs | 480 | 2,400 |
| Parameters | 28,370 | 28,386 |
| Added parameters | 0 | 16 (`+0.056%`) |
| CPU optimizer time / seed | 280.19+/-2.92 s | 1,275.70+/-86.78 s |
| Optimizer-time multiplier | 1.00x | 4.55x |
| All-test inference time / seed | 29.81+/-9.54 s | 29.52+/-1.27 s |

The inference measurement covers four viscosities, eight trajectories per
viscosity, and 50 autoregressive steps. The timings are effectively tied on
this CPU; Delta-Physics adds an analytic FFT correction, while conditioning
adds only one input feature. Training-data generation for the conditioned run
took `245.40+/-107.70 s` per seed and is reported separately from optimizer
time.

## Interpretation

The hoped-for accuracy parity does not occur: when viscosity-specific
trajectories at every test parameter are available, conditioned F-FNO is much
more accurate at every viscosity, horizon, and spectral metric. The comparison
therefore rules out a claim that analytic zero-shot correction replaces
multi-viscosity supervision at equal accuracy.

What remains supported is a different and still useful claim. Delta-Physics
needs only one viscosity, one fifth of the training trajectories/pairs, no
additional learned parameters, and no retraining for a new higher viscosity.
It improves the corresponding single-viscosity Increment model reliably, but
there is a clear accuracy--data tradeoff relative to fully supervised
multi-viscosity conditioning.

For a paper, the two methods should be described as different operating
regimes:

- multi-viscosity conditioned F-FNO is the accuracy-oriented supervised
  baseline when target-parameter simulations are available;
- single-viscosity Delta-Physics is the data-efficient zero-shot alternative
  when only reference-viscosity training data are available.
