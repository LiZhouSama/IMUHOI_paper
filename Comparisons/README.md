# IMUHOI Comparison Baselines

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

The public GlobalPose physics optimizer depends on its external dynamics stack
and is not silently approximated in Linux evaluation. Export the neural sequence
outputs for the Windows-side optimizer with:

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
- TIP unified evaluation does not apply Bullet/SBP root correction unless an
  external implementation is added; its translation mode is recorded as
  `root_velocity_no_bullet_sbp_correction`.
- GlobalPose unified evaluation reports neural stationary fusion without the
  physics optimizer. The JSON field `globalpose_physics_applied` is `false`.
- The corrected GlobalPose adapter uses the released SMPL-male mesh and the exact
  vertex sites `(1961, 5424, 1176, 4662, 411, 3021)` whenever that pickle is
  available. If the reference pickle is absent it falls back to the configured
  body model and records the fallback through the evaluation protocol metadata.
