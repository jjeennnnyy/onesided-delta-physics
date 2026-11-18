# Single-seed reference training-budget sensitivity

Paired seed 0 screening, not a three-seed significance test. The 15-epoch results are the original seed-0 results, not their three-seed mean.

## Protocol

- Same NS64 data-generation configuration and seeds: 24 training and 6 validation trajectories at nu0=1e-3; 480 training pairs. Test seed 1000, eight trajectories per viscosity, 50 steps.
- Same F-FNO initialization seed 0, shuffle seed 10000, architecture, normalization procedure, batch size 32, Adam lr=1e-3 and weight_decay=1e-6.
- Fresh 75-epoch training with validation-selected checkpoint, matching the original selection rule.
- Original StepLR policy retained: step_size=epochs//2, gamma=0.5. Thus decay occurs at epochs 7/14 for the original budget and 37/74 for the long budget. This tests a longer training budget with its original duration-scaled schedule, not an isolated epoch-count change at identical learning rates.
- 15 epochs: 225 optimizer updates; 75 epochs: 1,125 updates. The latter matches 15 epochs on the conditioned model's 2,400 training pairs, but uses only reference-viscosity data.
- Each selected checkpoint is shared by Increment and the parameter-free correction. No target-viscosity training data or tuning is used.

## Training

| Budget | Selected epoch | Validation one-step RelL2 | Training wall-clock (s) |
|---:|---:|---:|---:|
| 15 | 15 | 0.008603 | 278.43 |
| 75 | 75 | 0.003579 | 1678.70 |

Wall-clock is descriptive only: the runs occurred separately and CPU thread settings may differ.

## rel_l2_t1

| Epochs | nu/nu0 | Increment | Delta-Physics | Improvement (%) |
|---:|---:|---:|---:|---:|
| 15 | 1.0 | 0.008949 | 0.008949 | +0.00 |
| 15 | 1.5 | 0.009208 | 0.008905 | +3.29 |
| 15 | 1.75 | 0.009805 | 0.008887 | +9.36 |
| 15 | 2.0 | 0.010647 | 0.008872 | +16.67 |
| 75 | 1.0 | 0.003774 | 0.003774 | +0.00 |
| 75 | 1.5 | 0.004801 | 0.003756 | +21.77 |
| 75 | 1.75 | 0.006023 | 0.003756 | +37.64 |
| 75 | 2.0 | 0.007440 | 0.003762 | +49.43 |

## rel_l2_t10

| Epochs | nu/nu0 | Increment | Delta-Physics | Improvement (%) |
|---:|---:|---:|---:|---:|
| 15 | 1.0 | 0.079191 | 0.079191 | +0.00 |
| 15 | 1.5 | 0.082240 | 0.075921 | +7.68 |
| 15 | 1.75 | 0.087387 | 0.074620 | +14.61 |
| 15 | 2.0 | 0.094253 | 0.073491 | +22.03 |
| 75 | 1.0 | 0.030725 | 0.030725 | +0.00 |
| 75 | 1.5 | 0.040282 | 0.028074 | +30.31 |
| 75 | 1.75 | 0.050753 | 0.027070 | +46.66 |
| 75 | 2.0 | 0.062468 | 0.026229 | +58.01 |

## rel_l2_t20

| Epochs | nu/nu0 | Increment | Delta-Physics | Improvement (%) |
|---:|---:|---:|---:|---:|
| 15 | 1.0 | 0.146501 | 0.146501 | +0.00 |
| 15 | 1.5 | 0.153679 | 0.138221 | +10.06 |
| 15 | 1.75 | 0.162845 | 0.135548 | +16.76 |
| 15 | 2.0 | 0.174518 | 0.133501 | +23.50 |
| 75 | 1.0 | 0.050254 | 0.050254 | +0.00 |
| 75 | 1.5 | 0.069466 | 0.043989 | +36.68 |
| 75 | 1.75 | 0.088108 | 0.042150 | +52.16 |
| 75 | 2.0 | 0.108300 | 0.040843 | +62.29 |

## rel_l2_t50

| Epochs | nu/nu0 | Increment | Delta-Physics | Improvement (%) |
|---:|---:|---:|---:|---:|
| 15 | 1.0 | 0.372824 | 0.372824 | +0.00 |
| 15 | 1.5 | 0.401424 | 0.345324 | +13.98 |
| 15 | 1.75 | 0.422786 | 0.339000 | +19.82 |
| 15 | 2.0 | 0.447546 | 0.334616 | +25.23 |
| 75 | 1.0 | 0.094524 | 0.094524 | +0.00 |
| 75 | 1.5 | 0.138992 | 0.082323 | +40.77 |
| 75 | 1.75 | 0.174653 | 0.079847 | +54.28 |
| 75 | 2.0 | 0.212827 | 0.078190 | +63.26 |

## rollout_rel_l2_average

| Epochs | nu/nu0 | Increment | Delta-Physics | Improvement (%) |
|---:|---:|---:|---:|---:|
| 15 | 1.0 | 0.184046 | 0.184046 | +0.00 |
| 15 | 1.5 | 0.195485 | 0.172673 | +11.67 |
| 15 | 1.75 | 0.206661 | 0.169705 | +17.88 |
| 15 | 2.0 | 0.220357 | 0.167612 | +23.94 |
| 75 | 1.0 | 0.056092 | 0.056092 | +0.00 |
| 75 | 1.5 | 0.080284 | 0.049214 | +38.70 |
| 75 | 1.75 | 0.101453 | 0.047530 | +53.15 |
| 75 | 2.0 | 0.124232 | 0.046339 | +62.70 |

## rollout_psre_average

| Epochs | nu/nu0 | Increment | Delta-Physics | Improvement (%) |
|---:|---:|---:|---:|---:|
| 15 | 1.0 | 0.052060 | 0.052060 | +0.00 |
| 15 | 1.5 | 0.091128 | 0.047275 | +48.12 |
| 15 | 1.75 | 0.120334 | 0.045470 | +62.21 |
| 15 | 2.0 | 0.151171 | 0.043932 | +70.94 |
| 75 | 1.0 | 0.027047 | 0.027047 | +0.00 |
| 75 | 1.5 | 0.076780 | 0.024190 | +68.49 |
| 75 | 1.75 | 0.106928 | 0.023200 | +78.30 |
| 75 | 2.0 | 0.138124 | 0.022367 | +83.81 |

## rollout_psre_final

| Epochs | nu/nu0 | Increment | Delta-Physics | Improvement (%) |
|---:|---:|---:|---:|---:|
| 15 | 1.0 | 0.084261 | 0.084261 | +0.00 |
| 15 | 1.5 | 0.164335 | 0.075156 | +54.27 |
| 15 | 1.75 | 0.226368 | 0.071977 | +68.20 |
| 15 | 2.0 | 0.292056 | 0.069441 | +76.22 |
| 75 | 1.0 | 0.053322 | 0.053322 | +0.00 |
| 75 | 1.5 | 0.155883 | 0.047076 | +69.80 |
| 75 | 1.75 | 0.218443 | 0.044919 | +79.44 |
| 75 | 2.0 | 0.284153 | 0.043013 | +84.86 |

## Checks

```json
{
  "normalizers": {
    "input_mean": {
      "exact_equal": false,
      "max_abs_difference": 2.832772905136949e-10
    },
    "input_std": {
      "exact_equal": true,
      "max_abs_difference": 0.0
    },
    "increment_mean": {
      "exact_equal": false,
      "max_abs_difference": 1.5643309564383756e-11
    },
    "increment_std": {
      "exact_equal": true,
      "max_abs_difference": 0.0
    }
  },
  "reference_metrics_exact_equal": true
}
```

Both normalization standard deviations are bitwise identical. The recomputed means differ by at most 2.84e-10 (input) and 1.57e-11 (increment), consistent with floating-point reduction differences; normalization statistics are not bitwise identical across runs.

## Interpretation

The longer budget improves the reference model substantially: its reference-viscosity Step-50 RelL2 falls from 0.372824 to 0.094524. Yet the correction remains beneficial at all three higher viscosities. At ratio 2, Avg-50 improvement rises from 23.94% to 62.70%, and Step-50 improvement rises from 25.23% to 63.26%. The longer-budget model also gains 83.81% in average PSRE and 84.86% in final PSRE at ratio 2.

This screening result argues against the gain being solely an artifact of an undertrained reference model. It also confirms that the original reference model had substantial room to improve with additional optimization. It does not establish convergence at 75 epochs or robustness across seeds. The original single- versus multi-viscosity comparison therefore mixes training-data coverage and optimization-budget effects.

Interpret gains relative to the stronger checkpoint's own Increment baseline. One seed cannot establish robustness across training randomness; the new results have not been inserted into the paper.
