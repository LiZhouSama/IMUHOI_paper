# VC module ablation on IMHD (2026-11-01)

Full = the published IMUHOI++ pipeline (VC v5ar + OT v2_armrot_biclose; training is deterministic, a
re-run on OMOMO reproduced its checkpoints bit for bit). Every other row retrains its own VC (the recipe of
the published VC, seed 42), OT and joint stage with the published HP frozen; only the `vc_ablation_*` keys
differ (`model/rnn/velocity_contact_v5.py`).
w/o HP prior removes the pose prior together with every other v5ar addition except the conditional
contact factorization (coupling features, bidirectional object-motion head, initializer perturbation),
leaving IMU-HOI's contact evidence plus p_s = p_m * p_s|m; w/o pose prior only (supplementary) removes the
pose prior alone. Body-group rows remove one group from the pose prior (palm dynamics kept, as in A2).
Scored online with the unified protocol (HP smoothing 7, diff-3 velocity init) on the 15 test
sequences, best joint checkpoint, seed-42 Noitom test noise. F1 at the pipeline thresholds (hand 0.15,
object motion 0.5), per-sequence means; AP pools all published frames; event F1 matches GT and predicted
onsets / releases within ±10 frames in frames [90, T-30). Values in percent except errors (cm) and
over-segmentation. The sequence bootstrap does not cover training-seed variation (one seed per row).

## Main table

| Variant | L-hand F1 | R-hand F1 | Obj-motion F1 | Hand event F1 (±10 f) | Obj event F1 (±10 f) | Object err (cm) | HOI err (cm) |
|---|---|---|---|---|---|---|---|
| Full model (IMUHOI++ VC) | 87.7 | 96.5 | 70.8 | 11.5 | 33.3 | 32.69 | 17.23 |
| w/o HP prior | 86.8 | 92.3 | 70.0 | 2.4 | 4.8 | 33.18 | 17.88 |
| w/o conditional contact prob. | 90.9 | 97.4 | 69.8 | 22.2 | 15.6 | 30.49 | 15.62 |
| w/o left arm (pending) | – | – | – | – | – | – | – |
| w/o right arm (pending) | – | – | – | – | – | – | – |
| w/o torso (pending) | – | – | – | – | – | – | – |
| w/o legs (pending) | – | – | – | – | – | – | – |
| w/o pose prior only (suppl.) (pending) | – | – | – | – | – | – | – |

Difference to the full row, 95 % paired sequence-bootstrap CI (* = excludes 0):

| Variant | L-hand F1 | R-hand F1 | Obj-motion F1 | Hand event F1 (±10 f) | Obj event F1 (±10 f) | Object err (cm) | HOI err (cm) |
|---|---|---|---|---|---|---|---|
| w/o HP prior | -0.8 [-6.3, +5.7] | -4.3 [-9.5, -0.3]* | -0.8 [-1.8, +0.1] | -9.1 [-20.8, -0.4]* | -28.6 [-45.8, +0.0] | +0.49 [-1.93, +2.57] | +0.65 [-1.50, +2.91] |
| w/o conditional contact prob. | +3.2 [+0.2, +7.1]* | +0.9 [-0.2, +3.1] | -1.0 [-2.4, +0.0] | +10.7 [-2.0, +22.0] | -17.7 [-36.4, +0.0] | -2.20 [-5.27, -0.26]* | -1.61 [-3.69, -0.05]* |

## Threshold-free, precision / recall, event details

| Variant | L-hand AP | R-hand AP | Obj-motion AP | L-hand P | L-hand R | R-hand P | R-hand R | Hard-neg FP | Hand onset F1 | Hand release F1 | Obj start F1 | Obj stop F1 | Hand over-seg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full model (IMUHOI++ VC) | 96.9 | 96.6 | 99.7 | 88.9 | 90.0 | 94.4 | 99.2 | 21.6 | 12.1 | 11.0 | 44.4 | 22.2 | 3.459 |
| w/o HP prior | 97.6 | 95.8 | 99.0 | 91.8 | 86.8 | 94.7 | 91.2 | 12.9 | 2.7 | 2.2 | 0.0 | 9.5 | 18.919 |
| w/o conditional contact prob. | 98.2 | 97.0 | 99.7 | 91.0 | 91.3 | 95.9 | 99.1 | 20.5 | 19.2 | 25.2 | 15.4 | 15.8 | 1.811 |
| w/o left arm (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o right arm (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o torso (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o legs (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o pose prior only (suppl.) (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |

Difference to the full row, 95 % paired sequence-bootstrap CI (* = excludes 0):

| Variant | L-hand AP | R-hand AP | Obj-motion AP | L-hand P | L-hand R | R-hand P | R-hand R | Hard-neg FP | Hand onset F1 | Hand release F1 | Obj start F1 | Obj stop F1 | Hand over-seg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w/o HP prior | +0.6 [-1.2, +2.2] | -0.8 [-6.0, +2.3] | -0.7 [-2.9, +0.0] | +2.9 [+0.2, +7.0]* | -3.2 [-10.4, +4.5] | +0.3 [-0.3, +1.1] | -8.0 [-16.9, -0.7]* | -8.8 [-23.5, -0.7]* | -9.4 [-23.8, +1.6] | -8.8 [-19.3, -1.1]* | -44.4 [-66.7, +0.0] | -12.7 [-25.0, +0.0] | +15.459 [+2.533, +52.182]* |
| w/o conditional contact prob. | +1.2 [+0.2, +2.7]* | +0.5 [-0.2, +1.1] | +0.0 [-0.1, +0.3] | +2.1 [+0.1, +4.6]* | +1.3 [-4.6, +7.4] | +1.5 [-0.0, +4.9] | -0.1 [-0.5, +0.3] | -1.2 [-3.6, +0.0] | +7.1 [-3.8, +13.5] | +14.3 [-4.0, +32.2] | -29.1 [-53.0, +0.0] | -6.4 [-19.7, +2.1] | -1.649 [-6.729, +0.200] |

## Sources

- runs: `outputs/IMUHOI++/RNN_cano/vc_ablation_imhd_20261101/<variant>/` (VC `vc/`, pipeline `pipeline/`, per-frame `probs/`, `records.json`, `eval_dump.json`)
- published pipeline (full row): `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/v2_armrot_biclose/eval_imhd_best.json`
- script: `scripts/vc_probe/summarize_vc_ablation.py`
