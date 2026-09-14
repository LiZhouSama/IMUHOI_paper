# Neutral Canonical Skeleton Migration

**Date:** 2026-09-07  
**Schema:** `canonical_skeleton_version = 1`

## Contract

IMUHOI no longer uses source gender or source shape coefficients when generating
human training targets. Each dataset keeps its fitted model family, but uses one
neutral, zero-betas template:

| Dataset | Canonical family/model |
| --- | --- |
| OMOMO, PAHOI, InterCap | `SMPLX_NEUTRAL.npz`, zero betas |
| BEHAVE, IMHD, HODome | `smplh/neutral/model.npz`, zero betas |

The legacy AMASS preprocessor in `process/preprocess.py` is currently commented out. Existing AMASS `.pt` files do not satisfy schema v1 and are intentionally rejected when `require_canonical_skeleton=true`; they must not be mixed into a canonical run until that preprocessor is reactivated and migrated.

Every saved sequence contains:

```text
gender = neutral
body_model_type = smplx | smplh
canonical_body_model_gender = neutral
canonical_betas = zero
canonical_skeleton_version = 1
```

Source betas are not saved. Human joints, synthesized human IMU, velocities,
floor/contact labels, hand-object directions and bone lengths are all derived
from the same canonical joints. InterCap stores body-22 plus the two canonical
virtual palm anchors as SMPL-24.

## Training

All source model configs point to neutral SMPL-H/SMPL-X files and require the
canonical schema. Legacy pretrained module paths are cleared because every
position-derived target changed. RNN may train mixed SMPL-H/SMPL-X batches and
selects the family per sample. DiT/Mamba currently own one BodyModel, so their
training helper selects the neutral model from the active dataset family and
rejects mixed-family training rather than silently choosing the wrong model.

Required order after preprocessing:

1. retrain HumanPose;
2. retrain VelocityContact with the new HumanPose;
3. retrain ObjectTrans/refinement;
4. rerun all baselines under the new metric contract.

Old checkpoints and old metrics are not comparable to schema v1 results.

## Evaluation

`eval_IMUHOI.py` and `eval_batch_online.py` do not instantiate a second
BodyModel for metrics. They use:

```text
MPJPE: model pred_joints_global vs dataset position_global
HOI: published/re-fused object + refined predicted palm
     vs dataset object + canonical dataset palm
MPJRE: predicted full rotations vs stored GT pose/rotations
```

The model still loads the selected neutral BodyModel internally to generate
`pred_joints_global`. `--smpl_model_path` is retained as a deprecated no-op for
old command-line scripts.

## Validation

A real OMOMO sequence was processed with the new schema and successfully passed
one-sequence batched online inference without an evaluation BodyModel. Focused
canonical/preprocessing/evaluation tests are maintained under:

- `test/test_preprocess_omomo_canonical.py`
- `test/test_preprocess_smplx_canonical.py`
- `test/test_preprocess_smplh_canonical.py`
- `test/test_canonical_neutral_contract.py`
