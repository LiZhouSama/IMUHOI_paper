# Repository Guidelines

## Project Structure & Module Organization

This repository contains IMUHOI research code for sparse-IMU human-object interaction estimation. Core model code lives in `model/`, with active DiT components under `model/diffussion/` and legacy RNN code under `model/rnn/`. Training entry points are in `train/diffussion/`, `train/rnn/`, and `train/mamba/`. Dataset loading is centralized in `dataset/dataset_IMUHOI.py`; preprocessing scripts and generated `.pt` datasets are under `process/`. Shared math, parsing, pose, and rotation helpers live in `utils/`. Tests and diagnostics are in `test/`. Runtime artifacts, checkpoints, logs, and TensorBoard files belong in `outputs/`.

## Build, Test, and Development Commands

Use Python from the project environment with PyTorch and the repository's ML dependencies installed. Common commands:

```bash
bash scripts/train.sh hp --cfg configs/IMUHOI_train_dit.yaml
bash scripts/train.sh int --cfg configs/IMUHOI_train_dit.yaml
python eval_IMUHOI.py --dataset process/processed_split_data_OMOMO --config configs/IMUHOI_train_dit.yaml
python vis_IMUHOI.py --config configs/IMUHOI_train_dit.yaml --checkpoint outputs/<run>/best.pt
pytest test
```

`scripts/train.sh` wraps staged DiT training. Use `--debug`, `--batch_size N`, and `--epochs N` for small local runs.

## Coding Style & Naming Conventions

Write Python with 4-space indentation, snake_case functions and variables, and PascalCase classes. Keep configuration in YAML under `configs/`; prefer explicit config keys over hard-coded paths. Follow existing module naming patterns such as `train_human_pose.py`, `train_interaction.py`, and `test_rnn_online_inference.py`. Keep comments concise and use English in code comments unless documenting existing Chinese-facing script output.

## Testing Guidelines

Tests use `pytest` and are named `test_*.py` in `test/`. Add focused tests for dataset shape contracts, model pipeline changes, and inference behavior. For expensive GPU paths, include a small debug configuration or fixture so `pytest test/<file>.py` can run quickly. Do not rely on large generated outputs unless the test clearly documents the required fixture path.

## Commit & Pull Request Guidelines

This checkout does not expose Git history, so use clear, imperative commit messages such as `Add DiT interaction test` or `Fix dataset window padding`. Pull requests should describe the changed training or inference behavior, list the config and dataset used for validation, link related issues, and include screenshots or sample visualizations when changing `vis_IMUHOI.py` or demo output.

## Data & Artifact Hygiene

Avoid committing new checkpoints, TensorBoard logs, or bulk processed `.pt` data from `outputs/` or `process/processed_*` unless explicitly required. Keep machine-specific paths out of configs and scripts; prefer paths relative to the repository root.


## Custom instructions
有你认为不合理或不够优的点需要先提出来，不要完全按照我的指令修改，没有的话直接修改
请使用第一性原理思考。你不能总是假设我非常清楚自己想要什么和该怎么得到。请保持审慎，从原始需求和问题出发，如果动机和目标不清晰，停下来和我讨论。如果目标清晰但是路径不是最短，告诉我，并且建议更好的办法

当前已处于docker环境，用下面环境：
conda activate /workspace/envs/IMUHOI
否则尝试：
conda SAGE


每次运行代码前先看下各个gpu的显存余量再选择空的gpu跑