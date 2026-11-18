# Iterative Equation-Recast baseline for the frozen Increment F-FNO

## Dynamic one-step formulation

The frozen reference time-stepper is

`T0(x) = x + Decode[F_theta(Encode(x), grid)]`.

It is a Markov flow map rather than a steady source-to-solution operator. To
adapt equation recast to this setting, one backward-Euler step is written as

`O_nu[y] = y - dt*nu*Laplacian(y) - dt*N(y) = x`.

The analytical operator difference is

`O_nu[y] - O_nu0[y] = -dt*(nu-nu0)*Laplacian(y)`.

Treating `T0` as an approximation to the inverse reference discrete operator
gives the fixed-point map

`G_nu(y;x) = T0(x + dt*(nu-nu0)*Laplacian(y))`.

The initial iterate is `y0=T0(x)`. Subsequent updates use un-clipped vector
Aitken relaxation. Convergence is checked independently for every trajectory:

`||G_nu(y)-y||_2 / (||G_nu(y)||_2 + eps) <= 1e-5`.

The maximum number of recast iterations is 30. No learned parameter, model
update, fallback to Delta-Physics, or hidden relaxation clamp is used.

## Protocol

- Forced two-dimensional Navier--Stokes at `64 x 64`.
- Existing single-`nu0` Increment F-FNO checkpoints for seeds 0, 1, and 2.
- `nu0=1e-3`; ratios `{1.0, 1.5, 1.75, 2.0}`.
- Eight deterministic test trajectories per viscosity and seed.
- 50-step free-running rollout.
- Identical checkpoints, trajectories, normalization, RelL2, and PSRE code for
  Increment, one-sided Delta-Physics, and iterative Equation Recast.
- Values are mean +/- sample standard deviation over three seeds.

## RelL2 results

| nu/nu0 | Method | Step 1 | Step 10 | Step 20 | Step 50 | Avg 50 |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | Increment / Delta / Recast | .00898+/-.00120 | .07923+/-.01009 | .14993+/-.01717 | .40913+/-.06111 | .19647+/-.02527 |
| 1.5 | Increment | .00911+/-.00111 | .08115+/-.00937 | .15581+/-.01667 | .43308+/-.06382 | .20605+/-.02567 |
|  | Delta-Physics | **.00890+/-.00118** | .07579+/-.00900 | .14157+/-.01511 | .37956+/-.06122 | .18445+/-.02419 |
|  | Equation Recast | .00892+/-.00117 | **.07578+/-.00895** | **.14148+/-.01503** | **.37802+/-.06129** | **.18408+/-.02420** |
| 1.75 | Increment | .00966+/-.00113 | .08564+/-.00956 | .16382+/-.01722 | .45106+/-.06568 | .21566+/-.02640 |
|  | Delta-Physics | **.00887+/-.00117** | .07446+/-.00856 | .13895+/-.01451 | .37283+/-.06213 | .18135+/-.02412 |
|  | Equation Recast | .00889+/-.00116 | **.07444+/-.00848** | **.13883+/-.01443** | **.37093+/-.06211** | **.18089+/-.02412** |
| 2.0 | Increment | .01047+/-.00119 | .09182+/-.01005 | .17414+/-.01820 | .47211+/-.06797 | .22754+/-.02750 |
|  | Delta-Physics | **.00884+/-.00116** | .07332+/-.00818 | .13697+/-.01411 | .36816+/-.06317 | .17915+/-.02419 |
|  | Equation Recast | .00887+/-.00115 | **.07329+/-.00809** | **.13679+/-.01404** | **.36589+/-.06302** | **.17859+/-.02418** |

Relative to Delta-Physics, Equation Recast changes the metrics only marginally:

| nu/nu0 | Step 1 | Step 10 | Step 20 | Step 50 | Avg 50 |
|---:|---:|---:|---:|---:|---:|
| 1.5 | -0.17+/-0.52% | +0.01+/-0.26% | +0.06+/-0.12% | +0.41+/-0.11% | +0.21+/-0.08% |
| 1.75 | -0.25+/-0.77% | +0.01+/-0.36% | +0.08+/-0.14% | +0.52+/-0.12% | +0.25+/-0.10% |
| 2.0 | -0.32+/-1.02% | +0.03+/-0.44% | +0.13+/-0.15% | +0.62+/-0.10% | +0.31+/-0.09% |

The iterative method is slightly worse at Step 1 and slightly better late in
the rollout. The late improvement grows with viscosity shift, but remains
below one percent.

## Spectral results

| nu/nu0 | Method | Avg-50 PSRE | Final PSRE |
|---:|---|---:|---:|
| 1.5 | Increment | .06463+/-.02655 | .13174+/-.04764 |
|  | Delta-Physics | .03872+/-.00822 | .06802+/-.00626 |
|  | Equation Recast | **.03849+/-.00800** | **.06749+/-.00605** |
| 1.75 | Increment | .09274+/-.02886 | .19137+/-.05505 |
|  | Delta-Physics | .03804+/-.00722 | .06674+/-.00454 |
|  | Equation Recast | **.03773+/-.00693** | **.06603+/-.00443** |
| 2.0 | Increment | .12306+/-.03099 | .25576+/-.06247 |
|  | Delta-Physics | .03757+/-.00636 | .06598+/-.00310 |
|  | Equation Recast | **.03720+/-.00604** | **.06520+/-.00309** |

At ratio 2, Recast improves average PSRE by `0.91+/-0.72%` and final PSRE by
`1.19+/-0.89%` relative to Delta-Physics.

## Convergence and inference cost

| nu/nu0 | Mean iterations/step | Max | P95 | Mean NN evals/step | Non-converged | Wall-clock / seed | Recast / Delta time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.00 | 0 | 0 | 1.00 | 0 / 1,200 | 7.32+/-1.52 s | about 1.0x |
| 1.5 | 5.86+/-0.05 | 6 | 6 | 6.86 | 0 / 1,200 | 48.22+/-3.29 s | 7.08x |
| 1.75 | 6.41+/-0.10 | 8 | 7 | 7.41 | 0 / 1,200 | 61.20+/-5.16 s | 8.22x |
| 2.0 | 7.81+/-0.09 | 9 | 8 | 8.81 | 0 / 1,200 | 70.08+/-8.50 s | 8.69x |

All `3,600` OOD trajectory-steps converged, with no NaN/Inf and no use of the
30-iteration limit. The largest accepted relative residual was below `1e-5`.
Aitken coefficients remained finite without clipping:

- ratio 1.5: `0.363--0.525`;
- ratio 1.75: `0.255--0.516`;
- ratio 2.0: `0.212--0.527`.

At `nu=nu0`, all three seeds produce a complete 50-step Recast rollout with
maximum elementwise difference exactly `0.0` from Increment. The implementation
uses one NN evaluation and zero recast iterations per physical step.

## Conclusion

The dynamic Equation-Recast fixed point is numerically valid for this learned
time-stepper: it converges reliably, preserves IID behavior, and gives a small
late-rollout and spectral improvement over one-sided Delta-Physics. However,
the gain is only about `0.2--0.6%` in rollout RelL2 and about `0.5--1.2%` in
PSRE, while inference is `7--9x` slower.

Therefore this is an important strong baseline but not the preferred method in
the current regime. The single-pass one-sided Delta-Physics captures nearly all
of the available zero-shot benefit at one NN evaluation per step. Equation
Recast adds monitorable convergence and slightly better asymptotic accuracy,
but its cost--accuracy tradeoff is unfavorable for these viscosity shifts.
