# VC module ablation on HODome (2026-11-01)

Full = the published IMUHOI++ pipeline (VC v5ar + OT v2_armrot_biclose; training is deterministic, a
re-run on OMOMO reproduced its checkpoints bit for bit). Every other row retrains its own VC (the recipe of
the published VC, seed 42), OT and joint stage with the published HP frozen; only the `vc_ablation_*` keys
differ (`model/rnn/velocity_contact_v5.py`).
w/o HP prior removes the pose prior together with every other v5ar addition except the conditional
contact factorization (coupling features, bidirectional object-motion head, initializer perturbation),
leaving IMU-HOI's contact evidence plus p_s = p_m * p_s|m; w/o pose prior only (supplementary) removes the
pose prior alone. Body-group rows remove one group from the pose prior (palm dynamics kept, as in A2).
Scored online with the unified protocol (HP smoothing 7, diff-3 velocity init) on the 36 test
sequences, best joint checkpoint, seed-42 Noitom test noise. F1 at the pipeline thresholds (hand 0.15,
object motion 0.5), per-sequence means; AP pools all published frames; event F1 matches GT and predicted
onsets / releases within ±10 frames in frames [90, T-30). Values in percent except errors (cm) and
over-segmentation. The sequence bootstrap does not cover training-seed variation (one seed per row).

## Main table

| Variant | L-hand F1 | R-hand F1 | Obj-motion F1 | Hand event F1 (±10 f) | Obj event F1 (±10 f) | Object err (cm) | HOI err (cm) |
|---|---|---|---|---|---|---|---|
| Full model (IMUHOI++ VC) | 92.1 | 92.7 | 94.0 | 36.4 | 48.4 | 26.66 | 17.99 |
| w/o HP prior | 81.6 | 88.1 | 89.0 | 17.6 | 29.7 | 26.89 | 17.84 |
| w/o conditional contact prob. | 90.7 | 90.6 | 93.4 | 30.2 | 46.8 | 26.03 | 17.42 |
| w/o left arm (pending) | – | – | – | – | – | – | – |
| w/o right arm (pending) | – | – | – | – | – | – | – |
| w/o torso (pending) | – | – | – | – | – | – | – |
| w/o legs (pending) | – | – | – | – | – | – | – |
| w/o pose prior only (suppl.) (pending) | – | – | – | – | – | – | – |

Difference to the full row, 95 % paired sequence-bootstrap CI (* = excludes 0):

| Variant | L-hand F1 | R-hand F1 | Obj-motion F1 | Hand event F1 (±10 f) | Obj event F1 (±10 f) | Object err (cm) | HOI err (cm) |
|---|---|---|---|---|---|---|---|
| w/o HP prior | -10.5 [-15.1, -7.2]* | -4.6 [-8.0, -1.9]* | -5.0 [-10.0, -1.5]* | -18.8 [-24.0, -13.8]* | -18.7 [-26.4, -11.0]* | +0.23 [-1.49, +1.85] | -0.16 [-1.54, +1.10] |
| w/o conditional contact prob. | -1.4 [-3.3, +0.2] | -2.1 [-4.4, -0.2]* | -0.6 [-1.4, +0.1] | -6.2 [-11.7, -0.7]* | -1.6 [-6.4, +3.3] | -0.63 [-1.40, +0.02] | -0.57 [-1.60, +0.36] |

## Threshold-free, precision / recall, event details

| Variant | L-hand AP | R-hand AP | Obj-motion AP | L-hand P | L-hand R | R-hand P | R-hand R | Hard-neg FP | Hand onset F1 | Hand release F1 | Obj start F1 | Obj stop F1 | Hand over-seg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full model (IMUHOI++ VC) | 97.9 | 98.8 | 99.7 | 90.2 | 95.2 | 91.1 | 95.6 | 45.2 | 42.2 | 30.7 | 53.6 | 43.2 | 2.513 |
| w/o HP prior | 96.3 | 97.8 | 99.2 | 86.8 | 80.7 | 86.6 | 92.9 | 36.2 | 21.5 | 13.7 | 36.1 | 23.4 | 5.751 |
| w/o conditional contact prob. | 97.5 | 98.4 | 99.7 | 87.5 | 95.7 | 88.0 | 94.9 | 39.9 | 36.9 | 23.5 | 49.7 | 43.9 | 2.959 |
| w/o left arm (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o right arm (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o torso (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o legs (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o pose prior only (suppl.) (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |

Difference to the full row, 95 % paired sequence-bootstrap CI (* = excludes 0):

| Variant | L-hand AP | R-hand AP | Obj-motion AP | L-hand P | L-hand R | R-hand P | R-hand R | Hard-neg FP | Hand onset F1 | Hand release F1 | Obj start F1 | Obj stop F1 | Hand over-seg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w/o HP prior | -1.6 [-3.1, -0.6]* | -1.0 [-1.8, -0.4]* | -0.6 [-1.0, -0.3]* | -3.4 [-8.3, -0.3]* | -14.5 [-20.3, -9.5]* | -4.5 [-9.0, -1.3]* | -2.7 [-5.0, -0.4]* | -9.0 [-21.4, +2.2] | -20.7 [-25.9, -15.9]* | -17.0 [-23.3, -10.6]* | -17.5 [-24.7, -10.2]* | -19.9 [-29.5, -11.0]* | +3.238 [+2.225, +4.474]* |
| w/o conditional contact prob. | -0.4 [-1.0, +0.2] | -0.4 [-1.0, +0.0] | -0.0 [-0.1, +0.0] | -2.7 [-4.9, -1.1]* | +0.5 [-1.8, +3.3] | -3.1 [-6.5, -0.5]* | -0.7 [-2.0, +0.3] | -5.3 [-15.8, +4.4] | -5.3 [-11.8, +1.0] | -7.2 [-12.8, -1.6]* | -3.9 [-9.1, +1.0] | +0.7 [-5.7, +6.8] | +0.446 [-0.007, +1.011] |

## Sources

- runs: `outputs/IMUHOI++/RNN_cano/vc_ablation_hodome_20261101/<variant>/` (VC `vc/`, pipeline `pipeline/`, per-frame `probs/`, `records.json`, `eval_dump.json`)
- published pipeline (full row): `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/v2_armrot_biclose/eval_hodome_best.json`
- script: `scripts/vc_probe/summarize_vc_ablation.py`
