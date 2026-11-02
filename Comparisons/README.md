# IMUHOI Comparison Baselines

## Protocol v3 ("paper", default since 2026-10-23)

`train_comparison.py --protocol paper` (the default) trains each method as its
paper and released code define it, on the **same measured IMU stream as the main
IMUHOI pipeline**: the corpus' stored `human_imu`/`obj_imu`, noised once per
sequence exactly like `dataset/dataset_IMUHOI.py` (Noitom model, seed 42 plus a
path hash; verified bit-identical). Only the per-method input layout and
normalization are rebuilt from that stream (`common/paper_protocol.py`), so all
methods and ours see identical measurements. Object IMU inputs and object-velocity
outputs remain project extensions with their own parameters (TIP is the
exception: as before, its single network also reads the object sensor).

| Method | What follows the paper / released code | Known remaining deviations |
| --- | --- | --- |
| `dip18` | 5 root-normalized sensors `[ori45, acc15]`, z-score of inputs and SMPL targets, input dropout 0.2 → Dense 512 → 2×BiLSTM 512 → Dense 256 → Gaussian heads (softplus σ) for 15 joint rotation matrices plus the auxiliary acceleration reconstruction, NLL loss, Adam 1e-4 ×0.96/2000 steps, batch 16, 80 epochs, clip 1, early stopping; online 20 past + 5 future frames; root from the root sensor | windows of 120 frames at 30 fps (DIP: 300 at 60 fps); no DIP-IMU fine-tuning |
| `tip` | 72-D IMU (global root, other sensors in the current root frame) after the 11-frame acceleration moving average, 40-frame acc sums, 18-joint 2-axis state + root velocity + 5 SBPs, causal Transformer+RNN, history dropout 0.8 kept on at test time (as the released code), history noise ±0.15, ×100/×6/×12/jerk/SBP losses, Adam 1e-4 with per-step cosine (T_max = epochs + 850), batch 256, 40-frame windows, 1100 epochs; online: GT first-frame state, measured root fed back, 6-frame output filter, root translation corrected with the foot SBPs (minimal real-time runner) | SBP labels are regenerated from the corpus (`tip/sbp_cache.py`); no Bullet terrain; flat ground at the median foot-SBP height of the training split (OMOMO 0.028 m; `TIP_GROUND_SBP_HEIGHTS` per dataset) |
| `transpose` | 72-D input (acc/30, root last), S1/S2/S3/B1/B2 sizes, input-only dropout, each network trained on noisy GT upstream positions (σ 0.04 / 0.025), L2/BCE/accumulated velocity (n = 1, 3, 9, 27) losses, B2 output = local velocity / 3, contact label 0.48 m/s on the SMPL feet, Adam 1e-3, batch 256; online 20 + 5 frames, B1/B2 fusion with gravity velocity and floor clamp | B2 is trained on all clips (the paper used only clips without stable foot contact, which OMOMO hardly has) |
| `globalpose` | 84-D input (aRB, wRB, RRB, gR0), PL → IK1 → IK2 → VR with RNNWithInit and gravity refinement, L2 losses + BCE stationary + FK position loss on the IK2 pose, 100 teacher-forced epochs then 200 joint epochs, stationary joints = pelvis, feet, hands below TIP's 0.15 m/s; online with first-frame PL init and zero VR init, stationary-weighted velocity fusion, then the released per-frame physics optimizer (`globalpose/physics.py`: `forward_frame` ported line by line on a NumPy port of carticulate's dynamics, which matches the official C++ to float32 precision) | physics armature from the evaluation skeleton (SMPL-X neutral) instead of SMPL male, built with articulate's voxel procedure including its inertia reference point (inflates distal inertias, kept as released); 30 fps frames reach the 1/60 s optimizer in two steps toward causally interpolated targets (released gains diverge at 1/30 s); floor = lowest contact joint of the first frame placed at the GT root; friction-cone QP solved exactly instead of OSQP; sensors at the IMUHOI joints instead of mesh vertices; optimizer of the earlier adaptation (AdamW 1e-4, batch 16) because the paper defers to PNP |

Model selection follows each method's own rule and is written to
`selected.pt`, which the run script evaluates (`checkpoint` in
`PAPER_TRAINING_DEFAULTS`, override with `--checkpoint_policy`):

| Method | Rule | Why |
| --- | --- | --- |
| `dip18` | `best`: lowest total validation loss (NLL) | DIP uses early stopping |
| `tip` | `final`: state after 1100 epochs | released `train_model.py` has no validation and keeps the last state |
| `transpose` | `best_per_network`: S1/S2/S3/B1/B2/object head each at its own lowest validation loss, measured with clean GT upstream inputs | the networks are trained separately and share no parameters |
| `globalpose` | `final`: state after 100 + 200 epochs | fixed schedule; the total validation loss is dominated by the stationary BCE, which overfits while every pose/velocity term keeps improving (on OMOMO `best.pt` fell at epoch 35, inside the teacher-forced phase, and was far worse online) |

The validation set is 5% of the training files (`split_manifest.json`); the
test split is never used for selection.

Other datasets (2026-10-23, `scripts/run_comparison_all4_20261023.sh`, outputs
`outputs/Comparisons/all4_cano_20261023/<dataset>`): `train_comparison.py
--dataset <key> --schedule_scale auto` trains on that dataset's `train_path`
(the main pipeline's training segments) and multiplies epochs, validation,
checkpoint and GlobalPose teacher intervals by 4470 / (training files), so
every run takes the optimizer steps of the OMOMO run; TIP's cosine keeps the
unscaled period. Evaluation uses the dataset's full-sequence test set and its
declared body model (SMPL-H for BEHAVE/IMHD/HODome, as `eval_IMUHOI.py`).
TIP's flat ground uses each dataset's own foot-SBP height (BEHAVE 0.071, IMHD
0.068, PAHOI 0.095, HODome 0.085 m). Segment files store their parent
sequence's full tensor storage (~5x on BEHAVE); the loader compacts them, and
under protocol v3 a file that fails to load is an error instead of being
skipped. `scripts/summarize_comparisons_all5_20261023.py` merges all baselines
with the five-dataset IMUHOI++ vs IMUHOI table into
`outputs/Comparisons/all5_baselines_vs_imuhoi_20261023/summary.md` (all 16
runs finished 2026-10-25; renders of HODome/IMHD predictions in its `vis/`).

Shared evaluation (`eval_comparison.py`, detects v3 checkpoints): canonical
OMOMO test split, fixed per-sequence noise (`--simulate_imu_noise
--simulate_imu_noise_seed 42`), object integration anchored at the GT frame-0
object position. Joints use the body model's native palms (SMPL-X middle1),
matching `eval_IMUHOI.py`; `hoi_err_cm` is the main pipeline's contact-palm HOI,
`hoi_err_egoimu_cm` the EgoIMU variant. Rotation-to-axis-angle conversions run in
float64 (float32 loses ~0.5 deg near pi).

Run one method end to end (train, a rendered subset, full online test):

```bash
scripts/run_comparison_chain_20261023.sh dip18   # or tip / transpose / globalpose
python -m Comparisons.render_comparison_meshes \
  --prediction dip18=outputs/Comparisons/omomo_cano_20261023/dip18/predictions_vis/<seq>.pt \
  --output outputs/Comparisons/omomo_cano_20261023/vis/<seq>.png
```

The sections below describe the earlier protocol v2 (`--protocol legacy`).

---

This directory contains trainable IMUHOI adaptations of four published inertial
motion methods:

| Method | Human branch retained from the reference method | Translation path |
| --- | --- | --- |
| `dip18` | DIP Gaussian BiRNN pose reconstruction | The method has no root-translation estimator; evaluation keeps the first root position fixed |
| `tip` | Causal Transformer and recurrent state prediction | State root velocity integration; Bullet/SBP correction is not run by the unified evaluator |
| `transpose` | S1/S2/S3 pose and B1/B2 contact/velocity fusion | B1/B2 fusion and floor correction |
| `globalpose` | PL, IK and VR recurrent cascade | Neural VR/stationary fusion; the Windows physics stage is exported separately |

The protocol-v2 adapter in `Comparisons/common/adapters.py` follows the sensor
ordering, coordinate frames, reduced joint sets, target scaling and initialization
used by the corresponding public implementations. Human branches consume only
measured human IMU. Object IMU and an object-velocity head are IMUHOI extensions,
not claims about the original papers.

Comparison-specific data protocols are isolated under `Comparisons/common`:
`dataset.py` owns the baseline dataset variants, `imu_noise.py` adapts TIP's
rotation layout, and `globalpose_smpl.py` evaluates GlobalPose's mesh-mounted
sensors. The repository's main dataset, IMU noise, and human-pose modules do
not expose these comparison-only interfaces.

Object velocity is represented in the frame natural to each method and converted
to world coordinates during evaluation:

| Method | Predicted object-velocity frame |
| --- | --- |
| `dip18` | World |
| `tip` | Initial measured root frame |
| `transpose` | World |
| `globalpose` | Current corrected predicted root frame |

Acceleration and orientation alone do not determine absolute object position.
Protocol-v2 evaluation therefore anchors integration at the ground-truth object
position of frame zero. This is explicit in the JSON report and must be held
constant when comparing methods.

## Train

Run from the repository root in the project environment:

```bash
python -m Comparisons.train_comparison \
  --method transpose \
  --save_dir outputs/Comparisons
```

Replace `transpose` with `dip18`, `tip`, or `globalpose`. Method-specific defaults
for optimizer, scheduler, batch size, window size and loss weights are defined in
`Comparisons/train_comparison.py`; CLI arguments override them. A short executable
smoke run is:

```bash
python -m Comparisons.train_comparison \
  --method transpose --debug --epochs 1 \
  --max_train_sequences 1 --max_eval_sequences 1 \
  --max_train_batches 1 --max_eval_batches 1
```

TIP uses 16 attention heads, the published recurrent-state feedback loop, a
six-frame output filter, and optional SBP labels. The processed IMUHOI files do
not contain terrain SBP labels, so generate a sidecar cache first when those
labels are available:

```bash
python -m Comparisons.tip.generate_sbp_cache --help
```

Missing TIP cache entries are masked and do not become negative targets.

## Checkpoint Evaluation

The unified evaluator loads the trained checkpoint strictly, runs method-specific
full-sequence inference, reconstructs body joints with the configured body model,
and writes aggregate plus per-sequence metrics:

```bash
python -m Comparisons.eval_comparison \
  --method transpose \
  --checkpoint outputs/Comparisons/<run>/best.pt \
  --dataset omomo --split test \
  --output_json outputs/Comparisons/<run>/eval_omomo_test.json
```

To render the two paper tables from a completed set of method/dataset reports:

```bash
python -m Comparisons.generate_comparison_tables \
  --report_root outputs/Comparisons/<run>/evals \
  --output Comparisons/results/comparison_tables.tex \
  --summary_json Comparisons/results/comparison_tables.json
```

Use `--data_dir <processed-directory>` instead of `--dataset` for an explicit
dataset path. `--save_predictions <directory>` additionally stores decoded local
rotations, root translations, human joints and object translations per sequence.

Training checkpoints include `comparison_protocol_version`, object-output
semantics and architecture arguments. Original legacy checkpoints without this
metadata are loaded by strict weight-shape compatibility. `--protocol_version`
and `--object_output_mode` exist only for checkpoints whose metadata is missing
or known to be incorrect.

The default online contexts follow the reference runners: DIP and TransPose use
20 past plus 5 future frames (five-frame delayed output), TIP uses a causal
40-frame history, and GlobalPose carries recurrent state without future input.
The selected context is recorded in `online_window` in every JSON report.

## Metrics

The report contains all requested metrics. Aggregation is a macro mean over
sequences, matching the sequence-level style of `eval_batch_online.py` and
`eval_IMUHOI.py`.

| JSON key | Definition | Unit |
| --- | --- | --- |
| `human_ang_err_deg` | Axis-angle component MAE over the 21 non-root SMPL joints | degree |
| `human_geodesic_err_deg` | SO(3) geodesic rotation error over the 21 non-root joints | degree |
| `human_pos_err_cm` | Root-relative MPJPE over 22 SMPL joints | cm |
| `human_jitter_mm_frame2` | Mean joint second finite difference | mm/frame^2 |
| `root_trans_err_cm` | Root translation error | cm |
| `obj_err_cm` | Object translation error | cm |
| `hoi_err_cm` | Contact-hand error on contact frames and root-relative hand error otherwise | cm |
| `obj_trans_jitter_mm_frame2` | Object translation second finite difference | mm/frame^2 |

Compatibility aliases are also emitted: `ang_err`, `mpjre_angle`, `pos_err`,
`mpjpe`, `human_jitter`, `jitter`, `obj_err`, `obj_trans_err`, `hoi_err`, and
`obj_trans_jitter`.

## GlobalPose Physics Export

Protocol v3 runs the optimizer in `eval_comparison.py` (see above). For
protocol-v2 checkpoints, the neural sequence outputs can still be exported for
the Windows-only released optimizer with:

```bash
python -m Comparisons.export_globalpose_for_physics \
  --checkpoint outputs/Comparisons/<run>/best.pt \
  --dataset omomo --split test \
  --output_dir outputs/Comparisons/globalpose_physics_export/omomo_test
```

Full sequences are exported by default. Use `--windowed --window_size 120` only
when the downstream physics process is intentionally evaluating isolated windows.

## Interpretation Limits

- The object heads are repository-specific extensions. They preserve each human
  method's inputs but are not part of DIP, TIP, TransPose, or GlobalPose papers.
- DIP has no learned global root motion, so its root and absolute human joint
  metrics are expected to degrade whenever the subject translates.
  `--gt_root_trans` evaluates any method with the ground-truth root translation
  (root orientation stays measured); the report marks it as
  `root_translation_oracle: true`.
- TIP unified evaluation does not apply Bullet/SBP root correction unless an
  external implementation is added; its translation mode is recorded as
  `root_velocity_no_bullet_sbp_correction`.
- Protocol v3 GlobalPose evaluation applies the released physics optimizer
  (`globalpose_physics_applied: true`, CPU worker pool `--physics_workers`);
  `--no_globalpose_physics` reports the network output alone. Protocol v2
  evaluation reports neural stationary fusion without physics.
- The corrected GlobalPose adapter uses the released SMPL-male mesh and the exact
  vertex sites `(1961, 5424, 1176, 4662, 411, 3021)` whenever that pickle is
  available. If the reference pickle is absent it falls back to the configured
  body model and records the fallback through the evaluation protocol metadata.
