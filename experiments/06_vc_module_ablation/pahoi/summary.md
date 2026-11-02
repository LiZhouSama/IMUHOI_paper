# VC module ablation on PAHOI (2026-11-01)

Full = the published IMUHOI++ pipeline (VC v5ar + OT v2_armrot_biclose; training is deterministic, a
re-run on OMOMO reproduced its checkpoints bit for bit). Every other row retrains its own VC (the recipe of
the published VC, seed 42), OT and joint stage with the published HP frozen; only the `vc_ablation_*` keys
differ (`model/rnn/velocity_contact_v5.py`).
w/o HP prior removes the pose prior together with every other v5ar addition except the conditional
contact factorization (coupling features, bidirectional object-motion head, initializer perturbation),
leaving IMU-HOI's contact evidence plus p_s = p_m * p_s|m; w/o pose prior only (supplementary) removes the
pose prior alone. Body-group rows remove one group from the pose prior (palm dynamics kept, as in A2).
Scored online with the unified protocol (HP smoothing 7, diff-3 velocity init) on the 113 test
sequences, best joint checkpoint, seed-42 Noitom test noise. F1 at the pipeline thresholds (hand 0.15,
object motion 0.5), per-sequence means; AP pools all published frames; event F1 matches GT and predicted
onsets / releases within ±10 frames in frames [90, T-30). Values in percent except errors (cm) and
over-segmentation. The sequence bootstrap does not cover training-seed variation (one seed per row).

## Main table

| Variant | L-hand F1 | R-hand F1 | Obj-motion F1 | Hand event F1 (±10 f) | Obj event F1 (±10 f) | Object err (cm) | HOI err (cm) |
|---|---|---|---|---|---|---|---|
| Full model (IMUHOI++ VC) | 94.1 | 93.5 | 96.3 | 54.6 | 70.7 | 9.95 | 8.06 |
| w/o HP prior | 92.3 | 93.0 | 94.7 | 50.0 | 49.3 | 10.73 | 8.20 |
| w/o conditional contact prob. | 93.5 | 92.2 | 95.9 | 53.7 | 57.4 | 9.80 | 7.68 |
| w/o left arm (pending) | – | – | – | – | – | – | – |
| w/o right arm (pending) | – | – | – | – | – | – | – |
| w/o torso (pending) | – | – | – | – | – | – | – |
| w/o legs (pending) | – | – | – | – | – | – | – |
| w/o pose prior only (suppl.) (pending) | – | – | – | – | – | – | – |

Difference to the full row, 95 % paired sequence-bootstrap CI (* = excludes 0):

| Variant | L-hand F1 | R-hand F1 | Obj-motion F1 | Hand event F1 (±10 f) | Obj event F1 (±10 f) | Object err (cm) | HOI err (cm) |
|---|---|---|---|---|---|---|---|
| w/o HP prior | -1.8 [-5.0, +1.8] | -0.5 [-3.4, +2.7] | -1.6 [-2.5, -0.8]* | -4.6 [-15.7, +9.2] | -21.4 [-36.3, -4.0]* | +0.78 [+0.30, +1.31]* | +0.15 [-0.38, +0.67] |
| w/o conditional contact prob. | -0.7 [-2.7, +2.0] | -1.4 [-5.2, +1.4] | -0.4 [-1.0, +0.2] | -0.9 [-8.8, +9.4] | -13.2 [-22.4, -3.3]* | -0.15 [-0.58, +0.26] | -0.38 [-0.94, +0.15] |

## Threshold-free, precision / recall, event details

| Variant | L-hand AP | R-hand AP | Obj-motion AP | L-hand P | L-hand R | R-hand P | R-hand R | Hard-neg FP | Hand onset F1 | Hand release F1 | Obj start F1 | Obj stop F1 | Hand over-seg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full model (IMUHOI++ VC) | 98.8 | 98.9 | 99.6 | 93.3 | 95.7 | 93.8 | 95.1 | 17.7 | 32.1 | 77.0 | 50.0 | 91.3 | 4.091 |
| w/o HP prior | 94.7 | 94.6 | 99.1 | 90.8 | 95.5 | 90.2 | 96.3 | 21.3 | 29.6 | 70.4 | 10.5 | 88.1 | 3.909 |
| w/o conditional contact prob. | 98.4 | 99.1 | 99.6 | 93.5 | 95.4 | 92.3 | 94.2 | 16.2 | 28.6 | 78.7 | 30.0 | 84.8 | 3.455 |
| w/o left arm (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o right arm (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o torso (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o legs (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| w/o pose prior only (suppl.) (pending) | – | – | – | – | – | – | – | – | – | – | – | – | – |

Difference to the full row, 95 % paired sequence-bootstrap CI (* = excludes 0):

| Variant | L-hand AP | R-hand AP | Obj-motion AP | L-hand P | L-hand R | R-hand P | R-hand R | Hard-neg FP | Hand onset F1 | Hand release F1 | Obj start F1 | Obj stop F1 | Hand over-seg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w/o HP prior | -4.1 [-9.8, +0.1] | -4.3 [-10.2, +0.0] | -0.5 [-0.6, -0.3]* | -2.5 [-5.5, +0.7] | -0.3 [-3.4, +3.6] | -3.5 [-6.4, -1.5]* | +1.2 [-1.6, +4.5] | +3.6 [+0.3, +6.7]* | -2.5 [-18.4, +17.8] | -6.6 [-14.9, +3.4] | -39.5 [-66.7, -6.1]* | -3.3 [-9.6, +3.1] | -0.182 [-3.201, +3.000] |
| w/o conditional contact prob. | -0.5 [-1.7, +0.3] | +0.2 [-0.2, +0.6] | -0.0 [-0.1, +0.0] | +0.2 [-1.8, +3.0] | -0.3 [-2.3, +1.9] | -1.5 [-4.4, +0.3] | -0.9 [-4.3, +1.7] | -1.5 [-4.1, +0.4] | -3.6 [-15.7, +11.9] | +1.7 [-4.2, +8.9] | -20.0 [-36.7, +0.0] | -6.5 [-13.5, +0.2] | -0.636 [-2.501, +1.250] |

## Sources

- runs: `outputs/IMUHOI++/RNN_cano/vc_ablation_pahoi_20261101/<variant>/` (VC `vc/`, pipeline `pipeline/`, per-frame `probs/`, `records.json`, `eval_dump.json`)
- published pipeline (full row): `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/v2_armrot_biclose/eval_pahoi_best.json`
- script: `scripts/vc_probe/summarize_vc_ablation.py`
