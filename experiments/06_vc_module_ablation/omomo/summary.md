# VC module ablation on OMOMO (2026-10-30)

Full = the published IMUHOI++ pipeline (VC v5ar + OT v2_armrot_biclose; training is deterministic, a
re-run on OMOMO reproduced its checkpoints bit for bit). Every other row retrains its own VC (v5ar recipe,
seed 42), OT and joint stage; only the `vc_ablation_*` keys differ (`model/rnn/velocity_contact_v5.py`).
w/o HP prior removes the pose prior together with every other v5ar addition except the conditional
contact factorization (coupling features, bidirectional object-motion head, initializer perturbation),
leaving IMU-HOI's contact evidence plus p_s = p_m * p_s|m; w/o pose prior only (supplementary) removes the
pose prior alone. Body-group rows remove one group from the pose prior (palm dynamics kept, as in A2).
Scored online with the unified protocol (HP smoothing 7, diff-3 velocity init) on the 1173 test
sequences, best joint checkpoint, seed-42 Noitom test noise. F1 at the pipeline thresholds (hand 0.15,
object motion 0.5), per-sequence means; AP pools all published frames; event F1 matches GT and predicted
onsets / releases within ±10 frames in frames [90, T-30). Values in percent except errors (cm) and
over-segmentation. The sequence bootstrap does not cover training-seed variation (one seed per row).

## Main table

| Variant | L-hand F1 | R-hand F1 | Obj-motion F1 | Hand event F1 (±10 f) | Obj event F1 (±10 f) | Object err (cm) | HOI err (cm) |
|---|---|---|---|---|---|---|---|
| Full model (IMUHOI++ VC) | 95.5 | 95.8 | 98.1 | 56.2 | 83.0 | 13.34 | 10.48 |
| w/o HP prior | 92.6 | 93.1 | 97.4 | 40.6 | 71.2 | 14.11 | 10.73 |
| w/o conditional contact prob. | 95.0 | 95.3 | 98.2 | 52.1 | 83.0 | 13.56 | 10.67 |
| w/o left arm | 94.9 | 95.8 | 98.2 | 57.7 | 85.2 | 13.49 | 10.55 |
| w/o right arm | 95.6 | 95.7 | 98.2 | 55.7 | 85.5 | 13.31 | 10.49 |
| w/o torso | 95.4 | 95.8 | 98.2 | 56.9 | 85.0 | 13.29 | 10.57 |
| w/o legs | 95.3 | 95.9 | 98.2 | 56.7 | 84.1 | 13.44 | 10.57 |
| w/o pose prior only (suppl.) | 94.7 | 94.9 | 98.5 | 60.2 | 85.4 | 13.52 | 10.56 |

Difference to the full row, 95 % paired sequence-bootstrap CI (* = excludes 0):

| Variant | L-hand F1 | R-hand F1 | Obj-motion F1 | Hand event F1 (±10 f) | Obj event F1 (±10 f) | Object err (cm) | HOI err (cm) |
|---|---|---|---|---|---|---|---|
| w/o HP prior | -2.9 [-3.6, -2.2]* | -2.6 [-3.5, -1.9]* | -0.7 [-0.9, -0.6]* | -15.7 [-19.4, -12.0]* | -11.8 [-15.7, -8.0]* | +0.77 [+0.47, +1.04]* | +0.25 [+0.01, +0.51]* |
| w/o conditional contact prob. | -0.5 [-0.8, -0.1]* | -0.5 [-0.8, -0.2]* | +0.1 [-0.0, +0.2] | -4.1 [-7.6, -0.5]* | +0.0 [-2.9, +2.6] | +0.22 [+0.03, +0.42]* | +0.19 [-0.04, +0.40] |
| w/o left arm | -0.6 [-1.2, -0.0]* | +0.0 [-0.3, +0.4] | +0.1 [+0.0, +0.2]* | +1.5 [-2.1, +5.0] | +2.2 [-0.1, +4.5] | +0.15 [-0.05, +0.36] | +0.07 [-0.15, +0.31] |
| w/o right arm | +0.2 [-0.2, +0.5] | -0.0 [-0.4, +0.4] | +0.1 [+0.0, +0.2]* | -0.5 [-4.5, +3.3] | +2.5 [+0.1, +4.8]* | -0.03 [-0.25, +0.18] | +0.01 [-0.20, +0.23] |
| w/o torso | -0.0 [-0.4, +0.3] | +0.1 [-0.2, +0.3] | +0.1 [+0.0, +0.2]* | +0.7 [-2.2, +3.8] | +2.0 [-0.2, +4.3] | -0.05 [-0.25, +0.14] | +0.09 [-0.13, +0.29] |
| w/o legs | -0.2 [-0.6, +0.2] | +0.1 [-0.2, +0.4] | +0.0 [-0.1, +0.1] | +0.5 [-2.7, +3.9] | +1.1 [-1.4, +3.5] | +0.10 [-0.17, +0.33] | +0.08 [-0.20, +0.33] |
| w/o pose prior only (suppl.) | -0.8 [-1.3, -0.3]* | -0.9 [-1.6, -0.3]* | +0.3 [+0.2, +0.4]* | +4.0 [-0.4, +8.1] | +2.5 [-0.2, +5.0] | +0.18 [-0.08, +0.45] | +0.08 [-0.17, +0.32] |

## Threshold-free, precision / recall, event details

| Variant | L-hand AP | R-hand AP | Obj-motion AP | L-hand P | L-hand R | R-hand P | R-hand R | Hard-neg FP | Hand onset F1 | Hand release F1 | Obj start F1 | Obj stop F1 | Hand over-seg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full model (IMUHOI++ VC) | 99.2 | 99.3 | 99.9 | 94.9 | 97.0 | 94.9 | 97.5 | 35.2 | 42.0 | 70.4 | 82.4 | 83.6 | 2.911 |
| w/o HP prior | 98.2 | 98.4 | 99.7 | 93.1 | 93.6 | 93.7 | 94.0 | 31.1 | 24.4 | 56.7 | 66.3 | 76.0 | 5.214 |
| w/o conditional contact prob. | 99.1 | 99.1 | 99.9 | 93.1 | 97.8 | 93.0 | 98.4 | 40.2 | 38.8 | 65.4 | 82.4 | 83.6 | 2.732 |
| w/o left arm | 99.1 | 99.3 | 99.9 | 94.7 | 96.3 | 95.4 | 97.0 | 36.2 | 43.5 | 71.9 | 85.3 | 85.1 | 2.732 |
| w/o right arm | 99.2 | 99.1 | 99.9 | 94.8 | 97.3 | 95.4 | 96.8 | 37.2 | 40.2 | 71.3 | 84.7 | 86.2 | 3.089 |
| w/o torso | 99.2 | 99.2 | 99.9 | 95.0 | 96.7 | 94.9 | 97.4 | 35.3 | 42.0 | 71.9 | 84.9 | 85.1 | 2.866 |
| w/o legs | 99.2 | 99.3 | 99.9 | 94.7 | 97.0 | 95.2 | 97.4 | 36.4 | 41.8 | 71.7 | 83.6 | 84.5 | 2.848 |
| w/o pose prior only (suppl.) | 98.9 | 99.0 | 99.9 | 94.9 | 95.5 | 95.4 | 95.4 | 31.9 | 45.9 | 74.5 | 84.4 | 86.4 | 2.384 |

Difference to the full row, 95 % paired sequence-bootstrap CI (* = excludes 0):

| Variant | L-hand AP | R-hand AP | Obj-motion AP | L-hand P | L-hand R | R-hand P | R-hand R | Hard-neg FP | Hand onset F1 | Hand release F1 | Obj start F1 | Obj stop F1 | Hand over-seg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w/o HP prior | -1.0 [-1.3, -0.8]* | -0.9 [-1.1, -0.6]* | -0.2 [-0.2, -0.1]* | -1.9 [-2.5, -1.3]* | -3.4 [-4.3, -2.5]* | -1.2 [-1.9, -0.6]* | -3.5 [-4.3, -2.6]* | -4.1 [-5.9, -2.3]* | -17.6 [-22.3, -12.9]* | -13.7 [-17.4, -10.0]* | -16.1 [-21.7, -11.0]* | -7.5 [-10.7, -4.4]* | +2.304 [+1.518, +3.250]* |
| w/o conditional contact prob. | -0.1 [-0.2, +0.0] | -0.2 [-0.3, -0.1]* | +0.0 [+0.0, +0.0]* | -1.8 [-2.3, -1.4]* | +0.8 [+0.3, +1.3]* | -1.9 [-2.3, -1.6]* | +0.9 [+0.5, +1.2]* | +5.0 [+3.7, +6.5]* | -3.3 [-8.0, +2.0] | -5.0 [-8.1, -1.5]* | +0.0 [-3.8, +3.7] | +0.0 [-2.5, +2.2] | -0.179 [-0.520, +0.171] |
| w/o left arm | -0.1 [-0.3, +0.1] | +0.0 [-0.1, +0.1] | +0.0 [-0.0, +0.0] | -0.2 [-0.6, +0.2] | -0.7 [-1.4, -0.0]* | +0.5 [+0.1, +0.8]* | -0.5 [-0.9, -0.0]* | +1.0 [-0.4, +2.5] | +1.5 [-3.6, +6.6] | +1.5 [-1.4, +4.4] | +2.9 [-0.4, +6.0] | +1.6 [-0.6, +3.6] | -0.179 [-0.520, +0.203] |
| w/o right arm | -0.0 [-0.1, +0.1] | -0.1 [-0.3, +0.0] | +0.0 [+0.0, +0.0]* | -0.1 [-0.5, +0.2] | +0.3 [-0.1, +0.8] | +0.5 [+0.1, +0.9]* | -0.7 [-1.1, -0.2]* | +2.0 [+0.8, +3.2]* | -1.8 [-6.8, +3.6] | +0.8 [-2.7, +4.0] | +2.4 [-1.0, +5.8] | +2.7 [+0.3, +4.9]* | +0.179 [-0.252, +0.667] |
| w/o torso | -0.0 [-0.1, +0.1] | -0.0 [-0.1, +0.1] | +0.0 [+0.0, +0.0]* | +0.1 [-0.3, +0.5] | -0.3 [-0.8, +0.2] | -0.0 [-0.4, +0.4] | -0.1 [-0.4, +0.2] | +0.1 [-1.1, +1.3] | +0.0 [-4.4, +4.6] | +1.4 [-0.8, +3.7] | +2.5 [-0.6, +5.7] | +1.6 [-0.5, +3.8] | -0.045 [-0.345, +0.264] |
| w/o legs | +0.0 [-0.1, +0.2] | +0.0 [-0.1, +0.1] | +0.0 [-0.0, +0.0] | -0.2 [-0.6, +0.1] | +0.0 [-0.5, +0.5] | +0.3 [-0.1, +0.6] | -0.1 [-0.5, +0.3] | +1.2 [-0.0, +2.5] | -0.2 [-4.5, +4.4] | +1.3 [-1.3, +3.9] | +1.3 [-2.1, +4.4] | +1.0 [-1.3, +3.3] | -0.062 [-0.416, +0.288] |
| w/o pose prior only (suppl.) | -0.2 [-0.4, -0.1]* | -0.3 [-0.5, -0.1]* | +0.0 [+0.0, +0.0]* | -0.0 [-0.4, +0.3] | -1.5 [-2.2, -0.8]* | +0.5 [-0.1, +1.0] | -2.1 [-2.9, -1.4]* | -3.3 [-4.8, -1.6]* | +3.9 [-2.1, +9.4] | +4.1 [+0.5, +7.6]* | +2.1 [-1.7, +5.7] | +2.9 [+0.3, +5.4]* | -0.527 [-0.902, -0.128]* |

## Sources

- runs: `outputs/IMUHOI++/RNN_cano/vc_ablation_omomo_20261030/<variant>/` (VC `vc/`, pipeline `pipeline/`, per-frame `probs/`, `records.json`, `eval_dump.json`)
- published pipeline (full row): `outputs/IMUHOI++/RNN_cano/vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose/eval_omomo_best.json`
- script: `scripts/vc_probe/summarize_vc_ablation.py`
