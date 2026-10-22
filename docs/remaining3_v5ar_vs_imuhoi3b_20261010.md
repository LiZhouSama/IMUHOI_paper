# Remaining datasets: VC-v5ar + OT-v2-arm-bic vs IMUHOI three-branch

## Execution (2026-10-10)

Experiment root: `outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/`.
Six full domain-trained pipelines are queued serially on the available GPU.
The first run trains IMHD HP from scratch. This document is an execution plan,
not a completed numerical results report. The generated `summary.md`,
`metrics.csv`, and `metrics.json` under the experiment root are the results.
Unavailable metrics remain missing; they are never fabricated or zero-filled.

## Frozen recipes

- Proposed: HP -> VC-v5ar -> OT-v2 with forearm/object relative rotation and
  bimanual closure -> joint VC+OT (HP frozen).
- Original IMUHOI: the same initial HP -> VC-IMUHOI -> original OT-IMUHOI
  with left FK, right FK, and IMU integration -> joint HP+OT (VC frozen).
  This is `object_trans_arch: imuhoi`, **not** the four-head reimplementation.
- Seed: 42 for training and sequence-level simulated IMU noise.
- The configuration starts from the current `configs/IMUHOI_train_rnn.yaml`.
  Exact v5ar switches are copied from the completed OMOMO v5ar recipe:
  initializer noise probability 0.5, object/hand velocity standard deviation
  0.2 m/s, contact-flip probability 0, and IMU heading normalization disabled.
- Shared online settings: HP smoothing half-width 7 / polynomial order 2,
  object velocity initializer backward-difference length 3, the configured
  60-frame window and 10-frame lookahead. v2 uses FK alpha=0.35, beta=0.05.
- Batch size 800, four sampled windows per training sequence, two loader
  workers, no pin memory on WSL, online evaluation batch size 48.
- No debug training, epoch reduction, checkpoint transfer across datasets,
  or training on OMOMO/BEHAVE. Each dataset trains one initial HP which the
  two pipelines share. Baseline joint training can subsequently change HP.
- Each training/evaluation subprocess prints GPU memory availability before
  starting. The outer queue chooses the available GPU with the most free
  memory and runs one pipeline at a time.

## Dataset audit and maximum budgets

Train/test overlap is checked at original-source-sequence level, removing
segment suffixes. All three datasets passed. Canonical sample metadata is
skeleton version 1 and contact postprocess version 3; SMPL-H and SMPL-X
neutral body models are available.

| Dataset | Train samples | Train source sequences | Test samples for selection | Full-sequence test | Steps/epoch | Schedule scale | Stage max epochs | Joint max epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IMHD | 329 | 52 | 114 | 15 | 1 | 23.0 | 6900 | 2300 |
| PAHOI | 447 | 447 | 113 | 113 | 2 | 11.5 | 3450 | 1150 |
| HODome | 1130 | 144 | 285 | 36 | 5 | 4.6 | 1380 | 460 |

The reference is OMOMO's 23 optimizer steps/epoch. Scaling is inherited from
`scripts/run_behave_full_pipeline_trust.py`; all epoch-like schedule entries,
VC phase lengths, evaluation cadence, and early-stopping windows scale
consistently. These are maximum budgets; existing early stopping remains on.

## Evaluation and interpretation

Every pipeline evaluates both joint `best` and `final` checkpoint sets on the
full-sequence test paths using batch-online inference. The report includes
object position error, HOI error, MPJPE, root translation error, left/right/
object contact F1, object jitter, and paired relative object/HOI reductions.

**Existing-protocol limitation:** the trainers use the configured test-window
loader for checkpoint selection and early stopping. There is no independent
validation split. Reporting `final` does not remove that leakage, since early
stopping still observes the test-window loss. These runs preserve comparability
with the current OMOMO/BEHAVE experiments; they must not be represented as an
unbiased held-out benchmark. A publication-quality validation-only selection
protocol would require a separate train/validation split and reruns of all
compared datasets, including OMOMO and BEHAVE.

This is a single-seed **whole-pipeline comparison**, not an isolated OT effect:
VC architecture and joint fine-tuning targets differ. Do not select whichever
of `best` and `final` looks better on the full-sequence test; report both.

## Artifacts and continuation

The outer queue survives the current chat turn. A failed job is recorded and
later datasets are still attempted; no smaller budgets or different recipes
are substituted silently. Completed child stages are reused on resume.

```bash
# WSL: resume unfinished stages; the SAGE Python path is fixed in this launcher.
bash scripts/run_remaining_datasets_comparison.sh

# Refresh a partial table without touching any training process.
nvidia-smi
/home/l/anaconda3/envs/SAGE/bin/python scripts/run_remaining_datasets_comparison.py --summarize_only
```

Files in the experiment root:

- `recipe.yaml`: frozen recipe shared by all six jobs.
- `provenance.json`: original configuration paths, dataset counts, step-scaled
  schedules, and source SHA-256 hashes (including dirty workspace files).
- `queue_status.json`, `queue.log`, `queue.pid`: durable queue state and logs.
- `<dataset>/{proposed,imuhoi3b}/experiment_manifest.json`: stage checkpoints.
- `summary.md`, `metrics.csv`, `metrics.json`: refreshed after each job and at
  queue completion. Failed/incomplete rows are explicitly marked.

## Validation before launch

27 focused tests passed: queue/source-overlap/summary contracts, VC-v5,
OT-v2 arm-rotation+bimanual-closure behavior, and HP-smoothing/velocity-init
behavior. The shell launcher passed `bash -n`. All Python execution used
`/home/l/anaconda3/envs/SAGE/bin/python` under WSL; no Windows Python was used.
