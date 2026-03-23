# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMUHOI estimates human-object interaction (HOI) from sparse IMU signals. Given 6 body IMUs (root, shins, head, forearms), it reconstructs full-body SMPL pose and 6-DoF object motion. The current active architecture is **DiT** (Diffusion Transformer); the **RNN** path is legacy.

## Commands

### Training
```bash
# Stage 1: Human pose from IMU
bash scripts/train.sh hp --cfg configs/IMUHOI_train.yaml

# Stage 2/3: Object interaction (requires Stage 1 checkpoint)
bash scripts/train.sh int --cfg configs/IMUHOI_train.yaml

# Both stages sequentially
bash scripts/train.sh all --cfg configs/IMUHOI_train.yaml

# Common overrides
python train/diffussion/train_human_pose.py --cfg configs/IMUHOI_train.yaml \
    --batch_size 60 --epochs 300 --debug --no_trans
```

### Evaluation & Visualization
```bash
python eval_IMUHOI.py --dataset process/processed_split_data_OMOMO --config configs/IMUHOI_train.yaml

python vis_IMUHOI.py --config configs/IMUHOI_train.yaml \
    --checkpoint outputs/IMUHOI_DiT/human_pose_<timestamp>/best.pt
```

### Demo (real Noitom/STAG device data)
```bash
python demo.py --config configs/IMUHOI_train.yaml \
    --data-dict noitom/7IMU/shoe004.pt \
    --checkpoint outputs/IMUHOI_DiT/human_pose_<timestamp>/best.pt \
    --object-mesh path/to/object.obj
```

## Architecture

### Two-Stage Pipeline
1. **HumanPoseModule** (`model/diffussion/human_pose.py`) — Stage 1. Denoises full-body SMPL pose from 6 IMU signals (acc + 6D rotation = 9D each). Outputs 24 joint rotations + root translation.
2. **InteractionModule** (`model/diffussion/interaction.py`) — Stage 2/3 merged. Takes Stage 1 output + one object IMU to predict 6-DoF object pose. Replaces the old separate VelocityContact + ObjectTrans RNN stages.
3. **IMUHOIModel** (`model/diffussion/imuhoi_model.py`) — Combines both modules end-to-end.

Architecture is selected at runtime via `cfg.model_arch` (`"dit"` or `"rnn"`), resolved in `model/__init__.py`. The env var `IMUHOI_MODEL_ARCH` also overrides this.

### DiT Core (`model/diffussion/base.py`)
Diffusion Transformer with:
- DDIM sampler (recommended, `eta=0.0` for deterministic AR inference)
- Autoregressive inference with GT warmup (`dit_test_use_gt_warmup: true`)
- Root correction via foot contact (`dit_enable_root_correction: true`)
- Prediction type: `x0` (predicts clean signal, not noise)

### Configuration (`configs/`)
- `IMUHOI_train.yaml` — main DiT config (active)
- `IMUHOI_fineTuning.yaml` — fine-tuning with frozen modules
- `IMUHOI_train_rnn.yaml` — legacy RNN config
- `configs/__init__.py` — shared constants (sensor indices, joint names) and `load_config()`

All configs are loaded as `EasyDict` via `configs.load_config(path)`.

### Dataset (`dataset/dataset_IMUHOI.py`)
Supports: `omomo`, `behave`, `imhd`, `amass`. Selected via `train_datasets` in config. Data lives under `process/processed_*_data_*/` (symlinked to parent directory `../../datasets`). Window size set by `train.window` / `test.window` (default 60 frames at 30 FPS).

### Outputs
Saved under `outputs/IMUHOI_DiT/<module>_<timestamp>/`. Each run saves `best.pt` and periodic checkpoints. TensorBoard logs in the same directory.

### Pretrained Module Loading
Set `pretrained_modules.human_pose` and/or `pretrained_modules.interaction` in the YAML to load specific checkpoints before training. The `--hp_ckpt` / `--interaction_ckpt` CLI flags override the YAML values.

## Key Constants (`configs/__init__.py`)
- `FRAME_RATE = 30`
- 6 body IMUs at joint indices `[0, 7, 8, 15, 20, 21]` (root, L/R lower leg, head, L/R forearm)
- 10 "reduced" joints inferred without direct IMU: `[1,2,3,6,9,12,13,14,16,17]`
- 8 ignored joints (hands/feet): `[7,8,10,11,20,21,22,23]`
- IMU feature dim: 9 (3D accel + 6D rotation)

## 第一性原理
请使用第一性原理思考。你不能总是假设我非常清楚自己想要什么和该怎么得到。请保持审慎，从原始需求和问题出发，如果动机和目标不清晰，停下来和我讨论。如果目标清晰但是路径不是最短，告诉我，并且建议更好的办法

## 用户偏好
最好使用简体中文与我对话，但在代码中尽可能使用英文注释