# VC module ablation on BEHAVE (2026-10-31)

Full = the published IMUHOI++ pipeline (VC v5ar + OT v2_armrot_biclose; training is deterministic, a
re-run on OMOMO reproduced its checkpoints bit for bit). Every other row retrains its own VC (v5ar recipe,
seed 42), OT and joint stage; only the `vc_ablation_*` keys differ (`model/rnn/velocity_contact_v5.py`).
w/o HP prior removes the pose prior together with every other v5ar addition except the conditional
contact factorization (coupling features, bidirectional object-motion head, initializer perturbation),
leaving IMU-HOI's contact evidence plus p_s = p_m * p_s|m; w/o pose prior only (supplementary) removes the
pose prior alone. Body-group rows remove one group from the pose prior (palm dynamics kept, as in A2).
Scored online with the unified protocol (HP smoothing 7, diff-3 velocity init) on the 59 test
sequences, best joint checkpoint, seed-42 Noitom test noise. F1 at the pipeline thresholds (hand 0.15,
object motion 0.5), per-sequence means; AP pools all published frames; event F1 matches GT and predicted
onsets / releases within ±10 frames in frames [90, T-30). Values in percent except errors (cm) and
over-segmentation. The sequence bootstrap does not cover training-seed variation (one seed per row).

## Main table

| Variant | L-hand F1 | R-hand F1 | Obj-motion F1 | Hand event F1 (±10 f) | Obj event F1 (±10 f) | Object err (cm) | HOI err (cm) |
|---|---|---|---|---|---|---|---|
| Full model (IMUHOI++ VC) | 83.8 | 84.5 | 92.2 | 29.6 | 51.0 | 22.93 | 17.19 |
| w/o HP prior | 79.7 | 80.1 | 91.2 | 16.3 | 44.5 | 25.11 | 19.23 |
| w/o conditional contact prob. | 82.4 | 82.8 | 91.4 | 23.7 | 52.9 | 24.07 | 17.39 |
| w/o left arm | 83.4 | 83.5 | 92.4 | 25.4 | 57.4 | 23.38 | 17.42 |
| w/o right arm | 83.8 | 82.3 | 91.8 | 26.9 | 53.3 | 23.44 | 17.25 |
| w/o torso | 84.6 | 83.6 | 92.2 | 27.9 | 53.6 | 22.75 | 16.76 |
| w/o legs | 84.6 | 83.6 | 92.0 | 28.1 | 52.1 | 23.58 | 17.87 |
| w/o pose prior only (suppl.) (pending) | – | – | – | – | – | – | – |

Difference to the full row, 95 % paired sequence-bootstrap CI (* = excludes 0):

| Variant | L-hand F1 | R-hand F1 | Obj-motion F1 | Hand event F1 (±10 f) | Obj event F1 (±10 f) | Object err (cm) | HOI err (cm) |
|---|---|---|---|---|---|---|---|
| w/o HP prior | -4.1 [-7.4, -0.6]* | -4.4 [-8.3, -1.0]* | -1.0 [-2.4, +0.4] | -13.3 [-17.4, -9.8]* | -6.4 [-13.0, +1.2] | +2.18 [+1.09, +3.46]* | +2.04 [+0.73, +3.61]* |
| w/o conditional contact prob. | -1.4 [-3.5, +0.9] | -1.7 [-3.1, -0.5]* | -0.7 [-1.8, +0.0] | -5.8 [-10.0, -2.5]* | +1.9 [-2.4, +6.0] | +1.14 [+0.33, +2.01]* | +0.20 [-0.66, +1.13] |
| w/o left arm | -0.4 [-2.4, +1.9] | -0.9 [-2.6, +0.6] | +0.2 [-0.3, +0.8] | -4.2 [-7.9, -0.9]* | +6.4 [+1.9, +10.9]* | +0.44 [-0.50, +1.33] | +0.24 [-0.89, +1.30] |
| w/o right arm | +0.0 [-1.8, +2.3] | -2.2 [-4.5, -0.0]* | -0.4 [-1.2, +0.2] | -2.6 [-6.5, +0.6] | +2.3 [-1.7, +5.9] | +0.51 [-0.35, +1.33] | +0.07 [-0.90, +0.91] |
| w/o torso | +0.9 [-1.0, +3.0] | -0.8 [-3.2, +1.4] | -0.0 [-0.6, +0.6] | -1.7 [-4.5, +1.1] | +2.6 [-1.3, +6.6] | -0.19 [-1.12, +0.59] | -0.43 [-1.33, +0.31] |
| w/o legs | +0.8 [-0.8, +2.7] | -0.9 [-2.6, +0.9] | -0.2 [-1.0, +0.6] | -1.4 [-4.7, +1.6] | +1.2 [-3.5, +5.4] | +0.65 [-0.29, +1.53] | +0.69 [-0.49, +1.97] |

## Threshold-free, precision / recall, event details

| Variant | L-hand AP | R-hand AP | Obj-motion AP | L-hand P | L-hand R | R-hand P | R-hand R | Hard-neg FP | Hand onset F1 | Hand release F1 | Obj start F1 | Obj stop F1 | Hand over-seg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full model (IMUHOI++ VC) | 93.3 | 95.1 | 99.0 | 82.7 | 88.0 | 83.1 | 89.5 | 52.0 | 29.3 | 29.8 | 49.3 | 52.6 | 2.982 |
| w/o HP prior | 91.7 | 94.4 | 98.8 | 80.7 | 81.5 | 79.6 | 82.0 | 51.6 | 18.0 | 14.6 | 47.5 | 41.5 | 7.336 |
| w/o conditional contact prob. | 92.1 | 94.2 | 98.9 | 79.6 | 89.2 | 79.2 | 91.4 | 60.7 | 27.0 | 20.5 | 52.9 | 52.8 | 3.422 |
| w/o left arm | 93.6 | 95.8 | 99.1 | 83.7 | 86.6 | 84.7 | 86.5 | 47.5 | 25.1 | 25.7 | 57.8 | 57.0 | 3.924 |
| w/o right arm | 93.3 | 94.5 | 99.0 | 83.7 | 88.1 | 80.3 | 86.6 | 52.7 | 27.0 | 26.9 | 53.3 | 53.3 | 3.734 |
| w/o torso | 93.7 | 94.9 | 98.9 | 82.9 | 90.1 | 83.8 | 86.4 | 50.1 | 28.5 | 27.3 | 54.4 | 52.8 | 3.685 |
| w/o legs | 92.9 | 95.1 | 98.9 | 81.9 | 90.6 | 82.0 | 88.3 | 55.8 | 30.4 | 25.9 | 53.9 | 50.3 | 3.190 |
| w/o pose prior only (suppl.) (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |

Difference to the full row, 95 % paired sequence-bootstrap CI (* = excludes 0):

| Variant | L-hand AP | R-hand AP | Obj-motion AP | L-hand P | L-hand R | R-hand P | R-hand R | Hard-neg FP | Hand onset F1 | Hand release F1 | Obj start F1 | Obj stop F1 | Hand over-seg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w/o HP prior | -1.6 [-3.4, +0.8] | -0.7 [-2.5, +1.3] | -0.2 [-0.5, +0.2] | -2.0 [-4.8, +1.0] | -6.5 [-11.3, -1.6]* | -3.5 [-7.4, -0.1]* | -7.5 [-12.4, -2.8]* | -0.4 [-7.5, +5.9] | -11.4 [-15.9, -7.6]* | -15.2 [-20.0, -11.4]* | -1.8 [-8.8, +5.6] | -11.1 [-18.1, -2.8]* | +4.355 [+3.301, +5.535]* |
| w/o conditional contact prob. | -1.2 [-2.2, -0.3]* | -0.9 [-2.0, +0.0] | -0.0 [-0.2, +0.1] | -3.2 [-5.6, -0.6]* | +1.1 [-0.7, +3.3] | -3.9 [-6.0, -2.1]* | +1.9 [-0.2, +4.3] | +8.7 [+1.9, +16.3]* | -2.4 [-6.7, +1.2] | -9.3 [-13.6, -5.8]* | +3.6 [-0.6, +7.6] | +0.2 [-5.0, +5.2] | +0.440 [+0.087, +0.801]* |
| w/o left arm | +0.2 [-1.4, +2.5] | +0.7 [-0.2, +1.9] | +0.1 [-0.0, +0.2] | +1.0 [-1.9, +5.5] | -1.4 [-3.4, +0.6] | +1.6 [+0.1, +3.3]* | -3.0 [-5.1, -1.0]* | -4.5 [-7.5, -1.5]* | -4.2 [-8.4, -0.2]* | -4.1 [-8.1, -0.8]* | +8.5 [+3.3, +13.6]* | +4.4 [-0.1, +9.2] | +0.942 [+0.395, +1.552]* |
| w/o right arm | +0.0 [-0.9, +1.1] | -0.6 [-2.1, +0.8] | +0.1 [-0.1, +0.2] | +1.0 [-1.8, +5.3] | +0.0 [-2.4, +2.6] | -2.8 [-5.9, -0.5]* | -2.9 [-5.5, -0.5]* | +0.6 [-4.2, +5.0] | -2.3 [-6.1, +1.1] | -2.9 [-7.3, +1.1] | +3.9 [-0.3, +8.1] | +0.7 [-3.8, +4.7] | +0.752 [+0.287, +1.249]* |
| w/o torso | +0.3 [-0.3, +0.9] | -0.2 [-1.1, +0.6] | -0.1 [-0.2, +0.1] | +0.1 [-2.8, +3.7] | +2.1 [-0.0, +4.4] | +0.7 [-1.0, +2.4] | -3.1 [-6.2, -0.3]* | -1.9 [-6.6, +2.5] | -0.9 [-3.9, +2.4] | -2.6 [-6.1, +0.6] | +5.1 [+0.8, +9.5]* | +0.2 [-4.8, +5.0] | +0.703 [+0.313, +1.097]* |
| w/o legs | -0.4 [-1.3, +0.5] | +0.0 [-0.6, +0.6] | -0.0 [-0.2, +0.1] | -0.8 [-2.6, +1.5] | +2.6 [+0.6, +4.5]* | -1.1 [-2.6, +0.4] | -1.2 [-3.4, +1.1] | +3.7 [-0.0, +7.6] | +1.0 [-2.3, +4.1] | -3.9 [-7.4, -0.3]* | +4.6 [-0.2, +9.1] | -2.3 [-7.2, +2.2] | +0.208 [-0.125, +0.573] |

## Sources

- runs: `outputs/IMUHOI++/RNN_cano/vc_ablation_behave_20261031/<variant>/` (VC `vc/`, pipeline `pipeline/`, per-frame `probs/`, `records.json`, `eval_dump.json`)
- published pipeline (full row): `outputs/IMUHOI++/RNN_cano/vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose/eval_behave_best.json`
- script: `scripts/vc_probe/summarize_vc_ablation.py`
