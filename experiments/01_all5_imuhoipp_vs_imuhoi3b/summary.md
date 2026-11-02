# IMUHOI++ vs original IMUHOI (three-branch) on five datasets

IMUHOI++ = HP -> VC-v5ar -> OT-v2 (forearm/object rotation + bimanual closure, fixed fusion) -> joint VC+OT (HP frozen).
IMUHOI = HP -> VC-IMUHOI -> OT-IMUHOI with the paper's three candidates (left FK, right FK, IMU integration) -> joint HP+OT (VC frozen).

Shared per dataset: initial HP checkpoint, canonical data, seed 42, Noitom-style per-sequence IMU noise (seed 42),
optimizer-step-scaled schedule and early stopping. Evaluation (checked for every row): full-sequence batch-online,
window 60 / lookahead 10, HP smoothing half-width 7, diff-3 object-velocity init, initial contact 0.

Caveats: single training seed; `best` is selected on test-window loss (no separate validation split) and early
stopping also observes it; IMUHOI's joint stage fine-tunes HP, so its MPJPE/root differ from ours.

## Checkpoint: best

| Dataset | Method | Object (cm) | HOI (cm) | MPJPE (cm) | Root (cm) | L F1 | R F1 | Obj F1 | Obj jitter (mm/f²) |
|---|---|---|---|---|---|---|---|---|---|
| OMOMO | **IMUHOI++ (ours)** | 13.34 | 10.48 | 2.99 | 6.00 | 0.955 | 0.958 | 0.981 | 4.80 |
| OMOMO | IMUHOI (3-branch) | 17.84 | 15.04 | 2.99 | 7.17 | 0.916 | 0.920 | 0.951 | 5.70 |
| BEHAVE | **IMUHOI++ (ours)** | 22.93 | 17.19 | 3.68 | 16.13 | 0.838 | 0.845 | 0.922 | 5.74 |
| BEHAVE | IMUHOI (3-branch) | 95.45 | 79.90 | 3.72 | 26.67 | 0.799 | 0.799 | 0.889 | 3.24 |
| IMHD | **IMUHOI++ (ours)** | 32.69 | 17.23 | 4.22 | 21.42 | 0.877 | 0.965 | 0.708 | 12.88 |
| IMHD | IMUHOI (3-branch) | 103.24 | 97.49 | 4.64 | 21.56 | 0.820 | 0.953 | 0.695 | 8.97 |
| PAHOI | **IMUHOI++ (ours)** | 9.95 | 8.06 | 2.07 | 5.10 | 0.941 | 0.935 | 0.963 | 3.22 |
| PAHOI | IMUHOI (3-branch) | 12.72 | 11.02 | 2.07 | 6.99 | 0.920 | 0.932 | 0.952 | 2.75 |
| HODome | **IMUHOI++ (ours)** | 26.66 | 17.99 | 3.63 | 21.30 | 0.921 | 0.927 | 0.940 | 5.04 |
| HODome | IMUHOI (3-branch) | 81.24 | 58.91 | 3.56 | 39.31 | 0.861 | 0.893 | 0.889 | 3.83 |
| Mean (5) | IMUHOI++ (ours) | 21.11 | 14.19 | 3.32 | 13.99 | 0.906 | 0.926 | 0.903 | 6.34 |
| Mean (5) | IMUHOI (3-branch) | 62.10 | 52.47 | 3.39 | 20.34 | 0.863 | 0.899 | 0.875 | 4.90 |

Relative error reduction, (IMUHOI - ours) / IMUHOI (best):

| Dataset | Object | HOI | Root |
|---|---|---|---|
| OMOMO | 25.2% | 30.3% | 16.4% |
| BEHAVE | 76.0% | 78.5% | 39.5% |
| IMHD | 68.3% | 82.3% | 0.6% |
| PAHOI | 21.8% | 26.9% | 27.1% |
| HODome | 67.2% | 69.5% | 45.8% |

## Checkpoint: final

| Dataset | Method | Object (cm) | HOI (cm) | MPJPE (cm) | Root (cm) | L F1 | R F1 | Obj F1 | Obj jitter (mm/f²) |
|---|---|---|---|---|---|---|---|---|---|
| OMOMO | **IMUHOI++ (ours)** | 13.33 | 10.48 | 2.99 | 5.99 | 0.955 | 0.958 | 0.981 | 4.79 |
| OMOMO | IMUHOI (3-branch) | 17.84 | 15.04 | 2.99 | 7.18 | 0.916 | 0.920 | 0.951 | 5.70 |
| BEHAVE | **IMUHOI++ (ours)** | 22.88 | 17.14 | 3.68 | 16.16 | 0.839 | 0.845 | 0.921 | 5.75 |
| BEHAVE | IMUHOI (3-branch) | 98.05 | 83.37 | 3.71 | 29.30 | 0.799 | 0.797 | 0.889 | 3.18 |
| IMHD | **IMUHOI++ (ours)** | 31.78 | 16.43 | 4.25 | 21.63 | 0.906 | 0.965 | 0.706 | 12.89 |
| IMHD | IMUHOI (3-branch) | 123.35 | 101.39 | 4.16 | 47.46 | 0.820 | 0.953 | 0.695 | 8.81 |
| PAHOI | **IMUHOI++ (ours)** | 9.64 | 8.20 | 2.07 | 5.03 | 0.929 | 0.927 | 0.961 | 3.21 |
| PAHOI | IMUHOI (3-branch) | 12.78 | 10.97 | 2.06 | 6.71 | 0.920 | 0.932 | 0.951 | 2.76 |
| HODome | **IMUHOI++ (ours)** | 25.96 | 17.99 | 3.61 | 20.05 | 0.909 | 0.921 | 0.938 | 5.29 |
| HODome | IMUHOI (3-branch) | 84.00 | 61.65 | 3.55 | 39.53 | 0.861 | 0.893 | 0.889 | 3.83 |
| Mean (5) | IMUHOI++ (ours) | 20.72 | 14.05 | 3.32 | 13.77 | 0.908 | 0.923 | 0.901 | 6.39 |
| Mean (5) | IMUHOI (3-branch) | 67.20 | 54.48 | 3.29 | 26.04 | 0.863 | 0.899 | 0.875 | 4.86 |

Relative error reduction, (IMUHOI - ours) / IMUHOI (final):

| Dataset | Object | HOI | Root |
|---|---|---|---|
| OMOMO | 25.3% | 30.3% | 16.5% |
| BEHAVE | 76.7% | 79.4% | 44.8% |
| IMHD | 74.2% | 83.8% | 54.4% |
| PAHOI | 24.6% | 25.2% | 25.1% |
| HODome | 69.1% | 70.8% | 49.3% |

## Sources

| Dataset | Method | Full-sequence test seqs | Eval JSON (best) |
|---|---|---|---|
| OMOMO | IMUHOI++ (ours) | 1173 | `outputs/IMUHOI++/RNN_cano/vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose/eval_omomo_best.json` |
| OMOMO | IMUHOI (3-branch) | 1173 | `outputs/IMUHOI++/RNN_cano/imuhoi3b_20261012_omomo_unified/vc_imuhoi_ot_3branch/eval_omomo_best.json` |
| BEHAVE | IMUHOI++ (ours) | 59 | `outputs/IMUHOI++/RNN_cano/vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose/eval_behave_best.json` |
| BEHAVE | IMUHOI (3-branch) | 59 | `outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/vc_imuhoi_ot_3branch/eval_behave_best.json` |
| IMHD | IMUHOI++ (ours) | 15 | `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/v2_armrot_biclose/eval_imhd_best.json` |
| IMHD | IMUHOI (3-branch) | 15 | `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/vc_imuhoi_ot_3branch/eval_imhd_best.json` |
| PAHOI | IMUHOI++ (ours) | 113 | `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/v2_armrot_biclose/eval_pahoi_best.json` |
| PAHOI | IMUHOI (3-branch) | 113 | `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/vc_imuhoi_ot_3branch/eval_pahoi_best.json` |
| HODome | IMUHOI++ (ours) | 36 | `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/v2_armrot_biclose/eval_hodome_best.json` |
| HODome | IMUHOI (3-branch) | 36 | `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/vc_imuhoi_ot_3branch/eval_hodome_best.json` |
