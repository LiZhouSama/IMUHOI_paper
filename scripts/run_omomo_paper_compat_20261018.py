#!/usr/bin/env python3
"""Run the authorized W120/velocity-weight10 baseline and IMUHOI++ comparison."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_egoimu_vc_recipe as previous

OUT = ROOT / "outputs/IMUHOI++/paper_compat_w120_v10_seed10_omomo_20261018"
EXP, VC_RUN, PYTHON = previous.EXP, previous.OUT, previous.PYTHON
METHODS = ("baseline", "proposed")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    previous.write_json(path, value)


def sha(path):
    return previous.sha(path)


def source_paths():
    result = []
    for folder in ("model/rnn", "train/rnn", "dataset", "utils", "configs"):
        result.extend(sorted((ROOT / folder).rglob("*.py")))
    result += [ROOT / "eval_IMUHOI.py", ROOT / "eval_batch_online.py"]
    return result


def verify(protocol, data_hash=False):
    for name, digest in protocol["production_source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError(f"Production source changed: {name}")
    for name, digest in protocol["frozen_inputs_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError(f"Frozen input changed: {name}")
    for split, rows in protocol["data"].items():
        for row in rows:
            path = EXP / "data" / split / row["name"]
            stat = path.stat()
            if [stat.st_size, stat.st_mtime_ns] != row["stat"] or (data_hash and sha(path) != row["sha256"]):
                raise RuntimeError(f"Data changed: {path}")


def prepare():
    import yaml
    if (OUT / "protocol.json").exists():
        verify(read(OUT / "protocol.json"))
        return
    old_protocol = read(VC_RUN / "protocol.json")
    previous.verify_receipt(old_protocol)
    baseline_manifest = read(EXP / "baseline/experiment_manifest.json")
    hp = Path(baseline_manifest["stages"]["human_pose"]["checkpoint"])
    base_vc = VC_RUN / "w120_v10/final.pt"
    base_cfg = yaml.safe_load((VC_RUN / "w120_v10/config.yaml").read_text())
    proposed_manifest = read(EXP / "proposed/experiment_manifest.json")
    proposed_pipeline = next(iter(proposed_manifest["pipelines"].values()))
    proposed_cfg = yaml.safe_load(Path(proposed_pipeline["config"]).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    configs = {}
    for method, source in (("baseline", base_cfg), ("proposed", proposed_cfg)):
        cfg = copy.deepcopy(source)
        cfg.update(seed=10, num_workers=0, batch_size=800, windows_per_sequence=4,
                   train_datasets=["omomo"], datasets=copy.deepcopy(base_cfg["datasets"]),
                   train_data_window=120, test_data_window=60, epochs=300,
                   joint_train_epochs=100, early_stopping_patience_epochs=0,
                   joint_train_early_stopping_patience_epochs=0, deterministic_test_windows=True,
                   eval_interval=5, joint_train_eval_interval=1, use_tensorboard=False,
                   dataloader_pin_memory=False, simulate_imu_noise=False, validation_add_imu_noise=True,
                   simulate_imu_noise_seed=10, eval_hoi_protocol="egoimu", use_multi_gpu=False,
                   gpus=[0], device="cuda:0", no_trans=False, debug=False,
                   resume_dir=None, pretrained_ckpt=None, hp_ckpt=str(hp),
                   save_dir=str(OUT / method), joint_train=True,
                   vc_staged_training=method == "proposed", velocity_contact_arch="imuhoi" if method == "baseline" else "v5",
                   object_trans_arch="imuhoi" if method == "baseline" else "v2",
                   paper_compat_ot_training_anchor="wrists_20_21" if method == "baseline" else "native_palms")
        cfg["train"]["window"] = 120
        cfg["test"].update(window=60, online_lookahead=10)
        for weights in ("imuhoi_loss_weights", "loss_weights"):
            cfg.setdefault(weights, {}).update(obj_vel=10.0, hand_vel=10.0)
        cfg["pretrained_modules"] = {"human_pose": str(hp), "velocity_contact": None, "object_trans": None}
        for key in ("imu_acc_noise_std", "imu_rot_noise_std", "obj_imu_acc_noise_std", "obj_imu_rot_noise_std"):
            cfg[key] = .1
        if method == "proposed":
            cfg.update(v5_heading_normalize_imu=False, v5_init_noise_prob=.5,
                       v5_obj_vel_init_noise_std=.2, v5_hand_vel_init_noise_std=.2,
                       v5_contact_init_flip_prob=0., object_trans_forearm_obj_rotation=True,
                       object_trans_bimanual_closure=True)
            cfg["loss_weights"].update(bimanual_hoi=5., bimanual_prior=1.)
        directory = OUT / method
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "recipe.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs[method] = config_path
    inventory = read(EXP / "data/inventory.json")
    members = {row["file"] for row in old_protocol["training_files"]}
    data = {}
    for split, rows in inventory.items():
        selected = [row for row in rows if split == "test" or row["name"] in members]
        data[split] = [{"name": row["name"], "sha256": row["sha256"],
                        "stat": [(EXP / "data" / split / row["name"]).stat().st_size,
                                 (EXP / "data" / split / row["name"]).stat().st_mtime_ns]} for row in selected]
    if len(data["train"]) != 4115 or len(data["test"]) != 1177:
        raise RuntimeError("Wrong common membership")
    inputs = [hp, base_vc, VC_RUN / "initial.pt", VC_RUN / "protocol.json", *configs.values()]
    protocol = {"created_at": previous.now(), "seed": 10, "methods": {m: str(p) for m, p in configs.items()},
        "shared_hp": str(hp), "baseline_vc": str(base_vc), "data": data,
        "shared_hp_scope": "Same existing pre-joint HP, trained earlier on 4701 eligible members at W60; HP is not retrained in this experiment.",
        "budget": {"train_sequences": 4115, "train_window": 120, "batch_size": 800, "windows_per_sequence": 4,
                   "sampled_windows_per_epoch": 16460, "used_windows_per_epoch": 16000, "steps_per_epoch": 20,
                   "frames_per_epoch": 1920000, "vc_epochs": 300, "ot_epochs": 300, "joint_epochs": 100,
                   "vc_updates": 6000, "ot_updates": 6000, "joint_updates": 2000, "early_stopping": False},
        "method_specific_training": {"baseline": "Reuse audited W120/weight10 final VC; joint trains HP+OT, freezes VC; OT inputs adapted to wrists to match existing wrist losses/inference.",
                                     "proposed": "Retrain VC-v5ar; inherited SUPPORTS_ENDPOINT_FINETUNE_STAGE=False means 120 velocity + 180 joint-contact epochs, no endpoint-only tail; retain AR init augmentation. Joint trains VC+OT and freezes HP."},
        "checkpoint_policy": "Final joint epoch99 is primary, best joint secondary; existing auto-joint starts from window-validation-best OT; proposed VC uses final global epoch299, including its intrinsic curriculum best restores.",
        "evaluation": {"primary": "offline full sequences", "secondary": "batch_online", "online_window": 60,
                       "online_lookahead": 10, "online_chunk": 16, "hoi_metric_protocol": "egoimu", "noise": "clean"},
        "production_source_sha256": {str(path): sha(path) for path in source_paths()},
        "frozen_inputs_sha256": {str(path): sha(path) for path in inputs}, "initial_wrapper_sha256": sha(__file__),
        "not_strict_original_training_replay": True,
        "paper_better_explanation": "Prior controlled original-context VC replacement also improves Obj; training sampling, initialization, canonical supervision and checkpoint selection differ from original published run, not all separated."}
    verify(protocol, data_hash=True)
    write(OUT / "protocol.json", protocol)
    write(OUT / "status.json", {"status": "prepared", "updated_at": previous.now(), "stages": {}})
    print("PREPARED: common4115/test1177, W120, weight10, OT300+joint100, seed10", flush=True)


def state_steps(optimizer):
    result = {}
    for group in optimizer.param_groups:
        values = [int(optimizer.state[p]["step"]) for p in group["params"] if "step" in optimizer.state.get(p, {})]
        result[group.get("group_name", group.get("name", "all"))] = {"min": min(values) if values else 0,
                                                                 "max": max(values) if values else 0}
    return result


def worker(method, role):
    import torch
    import yaml
    from dataset.dataset_IMUHOI import IMUDataset
    import train.rnn.train_utils as training
    from utils.utils import setup_seed

    protocol = read(OUT / "protocol.json")
    verify(protocol)
    config_path = Path(protocol["methods"][method])
    directory = OUT / method / role
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    names = [row["file"] for row in read(VC_RUN / "protocol.json")["training_files"]]
    paths = [str(EXP / "data/train" / name) for name in names]

    class CommonDataset(IMUDataset):
        def __init__(self, *args, **kwargs):
            is_train = any(Path(p).resolve() == (EXP / "data/train").resolve() for p in kwargs.get("data_dir", []))
            if is_train:
                kwargs["sequence_paths"] = paths
            super().__init__(*args, **kwargs)
            if is_train and len(self) != 4115:
                raise RuntimeError(f"Wrong training membership: {len(self)}")

    training.IMUDataset = CommonDataset
    progress = {"method": method, "role": role, "status": "initializing", "pid": os.getpid(),
                "wrapper_sha256": sha(__file__), "started_at": previous.now()}

    def save_progress(phase, trainer, epoch, loss, components, validation=False):
        path = directory / f"progress_{phase}.json"
        current = read(path) if path.exists() else dict(progress)
        current.update(phase=phase, status="training", updated_at=previous.now(), epoch_budget=int(trainer.cfg.epochs),
                       optimizer_steps=trainer.n_iter, parameter_group_steps=state_steps(trainer.optimizer),
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        if validation:
            current.update(validation_epoch=epoch, validation_loss=loss, validation_components=components)
        else:
            if not math.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {method}/{phase}/{epoch}")
            current.update(epoch=epoch, epochs_completed=epoch + 1, train_loss=loss, train_components=components)
        write(path, current)

    def fixed_save(cfg, _module_name):
        cfg.save_dir = str(directory)
        return str(directory)

    def common_loader(cfg):
        loaders = training.create_dataloaders(cfg)
        if len(loaders[0]) != 20 or len(loaders[0].dataset) != 4115:
            raise RuntimeError("Wrong optimizer or membership budget")
        write(directory / "loader_receipt.json", {"train_sequences": len(loaders[0].dataset), "validation_sequences": len(loaders[1].dataset),
              "steps_per_epoch": len(loaders[0]), "train_window": cfg.train_data_window, "test_window": cfg.test_data_window,
              "obj_velocity_weight": 10, "hand_velocity_weight": 10, "updated_at": previous.now()})
        return loaders

    arguments = [__file__, "--cfg", str(config_path), "--seed", "10", "--num_workers", "0",
                 "--train_datasets", "omomo", "--velocity_contact_arch", "imuhoi" if method == "baseline" else "v5"]
    if (directory / "last.pt").exists():
        arguments += ["--resume_dir", str(directory)]
    if role == "vc":
        import train.rnn.train_velocity_contact as entry
        original_trainer = entry.VelocityContactTrainer
        original_build = entry.build_velocity_contact_module

        def build(cfg):
            setup_seed(10)
            model = original_build(cfg)
            shared = torch.load(VC_RUN / "initial.pt", map_location="cpu", weights_only=False)["model_state_dict"]
            copied = []
            for name in ("hand_vel_net", "obj_vel_net"):
                subset = {key[len(name)+1:]: value for key, value in shared.items() if key.startswith(name + ".")}
                getattr(model, name).load_state_dict(subset, strict=True)
                copied.extend(name + "." + key for key in subset)
            initial_path = directory / "initial.pt"
            if initial_path.exists():
                model.load_state_dict(torch.load(initial_path, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
            else:
                torch.save({"model_state_dict": model.state_dict(), "seed": 10, "shared_velocity_keys": copied}, initial_path)
            write(directory / "initialization_receipt.json", {"shared_velocity_key_count": len(copied), "strict_submodule_load": True,
                  "baseline_initial_sha256": sha(VC_RUN / "initial.pt"), "full_initial_sha256": sha(initial_path)})
            return model

        class ReportingVC(original_trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if self.loss_fn.weights["obj_vel"] != 10 or self.loss_fn.weights["hand_vel"] != 10:
                    raise RuntimeError("VC effective loss weights are not 10")
            def train_epoch(self, epoch):
                result = super().train_epoch(epoch)
                save_progress("vc", self, epoch, *result)
                return result
            def evaluate_all_losses(self, epoch):
                result = super().evaluate_all_losses(epoch)
                save_progress("vc", self, epoch, result[0], result[2], validation=True)
                return result
        entry.build_velocity_contact_module = build
        entry.VelocityContactTrainer = ReportingVC
        arguments += ["--hp_ckpt", protocol["shared_hp"]]
    else:
        import train.rnn.train_object_trans as entry
        original_trainer = entry.Stage3JointTrainer
        original_load = entry.load_checkpoint

        def strict_load(model, path, device, strict=True):
            return original_load(model, path, device, strict=True)
        entry.load_checkpoint = strict_load
        ot_class_name = "IMUHOIObjectTransModule" if method == "baseline" else "V2ObjectTransModule"
        original_ot = getattr(entry, ot_class_name)
        class SeededOT(original_ot):
            def __init__(self, cfg):
                setup_seed(10)
                super().__init__(cfg)
                path = directory / "initial_object_trans.pt"
                if path.exists():
                    self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
                else:
                    torch.save({"model_state_dict": self.state_dict(), "seed": 10}, path)
        setattr(entry, ot_class_name, SeededOT)

        class ReportingOT(original_trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._anchor_verified = method != "baseline"
                if method == "baseline":
                    if not getattr(self.hp_model, "_paper_compat_wrist_alias", False):
                        original_forward = self.hp_model.forward
                        def wrist_forward(*forward_args, **forward_kwargs):
                            result = dict(original_forward(*forward_args, **forward_kwargs))
                            result["pred_palm_glb_pos"] = result["pred_hand_glb_pos"]
                            return result
                        self.hp_model.forward = wrist_forward
                        self.hp_model._paper_compat_wrist_alias = True
                    # This checks the actual hand tensor consumed by OT, including joint HP gradients.
                    original_hp_forward = self.hp_model.forward
                    def record_forward(*forward_args, **forward_kwargs):
                        result = original_hp_forward(*forward_args, **forward_kwargs)
                        self._expected_wrists = result["pred_hand_glb_pos"]
                        return result
                    self.hp_model.forward = record_forward
                    def check_anchor(_model, forward_args):
                        if not self._anchor_verified:
                            difference = (forward_args[0] - self._expected_wrists).abs().max().item()
                            if difference != 0:
                                raise RuntimeError(f"Baseline OT did not consume wrists: {difference}")
                            self._anchor_verified = True
                            write(directory / ("anchor_joint.json" if self.joint_train else "anchor_ot.json"),
                                  {"training_hand_anchor": "wrists_20_21", "max_abs_difference": difference, "gradients_preserved": True})
                    original_ot_forward = self.ot_model.forward
                    def checked_ot_forward(*forward_args, **forward_kwargs):
                        # The standardized offline inference calls forward directly,
                        # so nn.Module forward-pre-hooks are bypassed.
                        check_anchor(self.ot_model, forward_args)
                        return original_ot_forward(*forward_args, **forward_kwargs)
                    self.ot_model.forward = checked_ot_forward
                if self.vc_loss_fn is not None and any(self.vc_loss_fn.weights[key] != 10 for key in ("obj_vel", "hand_vel")):
                    raise RuntimeError("Joint VC effective loss weights are not 10")
            def train_epoch(self, epoch):
                result = super().train_epoch(epoch)
                if not self._anchor_verified:
                    raise RuntimeError("Unverified baseline training anchor")
                save_progress("joint" if self.joint_train else "ot", self, epoch, *result)
                return result
            def evaluate(self, epoch):
                result = super().evaluate(epoch)
                save_progress("joint" if self.joint_train else "ot", self, epoch, *result, validation=True)
                return result
        entry.Stage3JointTrainer = ReportingOT
        vc = protocol["baseline_vc"] if method == "baseline" else str(OUT / "proposed/vc/final.pt")
        arguments += ["--hp_ckpt", protocol["shared_hp"], "--vc_ckpt", vc,
                      "--object_trans_arch", "imuhoi" if method == "baseline" else "v2", "--joint_train"]
    entry.create_save_dir = fixed_save
    entry.create_dataloaders = common_loader
    sys.argv = arguments
    write(directory / "worker_command.json", arguments)
    entry.main()
    expected = directory / ("final.pt" if role == "vc" else "final_joint.pt")
    payload = torch.load(expected, map_location="cpu", weights_only=False)
    expected_epoch, expected_steps = (299, 6000) if role == "vc" else (99, 2000)
    if payload["epoch"] != expected_epoch or payload["n_iter"] != expected_steps:
        raise RuntimeError(f"Wrong final budget: {expected}")
    verify(protocol)
    write(directory / "completed.json", {"epoch": payload["epoch"], "optimizer_steps": payload["n_iter"],
          "final_sha256": sha(expected), "completed_at": previous.now(), "wrapper_sha256": sha(__file__)})


def eval_command(method, mode, tag):
    protocol = read(OUT / "protocol.json")
    directory = OUT / method / "ot"
    hp = directory / f"{tag}_human_pose.pt" if method == "baseline" else Path(protocol["shared_hp"])
    vc = Path(protocol["baseline_vc"]) if method == "baseline" else directory / f"{tag}_velocity_contact.pt"
    output = OUT / method / f"eval_{mode}_{tag}.json"
    command = [str(PYTHON), str(ROOT / ("eval_IMUHOI.py" if mode == "offline" else "eval_batch_online.py")),
        "--config", protocol["methods"][method], "--dataset", "omomo", "--hp_ckpt", str(hp), "--vc_ckpt", str(vc),
        "--ot_ckpt", str(directory / f"{tag}_object_trans.pt"), "--velocity_contact_arch", "imuhoi" if method == "baseline" else "v5",
        "--object_trans_arch", "imuhoi" if method == "baseline" else "v2", "--seed", "10", "--num_workers", "0", "--output_json", str(output)]
    command += ["--inference_mode", "offline"] if mode == "offline" else ["--online_batch_size", "24", "--online_window", "60", "--online_lookahead", "10", "--online_chunk", "16"]
    return command, output


def report(state):
    lines = ["# OMOMO 120帧/速度权重10：基线与IMUHOI++", "", f'状态：{state["status"]}；更新：{previous.now()}', "",
        "固定seed10、4115训练成员、当前预处理、IMU高斯σ0.1；训练窗口120、窗口验证60。复用共同初始HP及已完成的基线VC。",
        "OT300轮+联合100轮，不早停；末轮为主结果。++保留VC分阶段逻辑。基线OT训练输入在独立适配层对齐手腕；生产源码不改。", "",
        "| 方法/阶段 | 已完成epoch | 预算 | 最近验证损失 |", "|---|---:|---:|---:|"]
    for method, role, phase in (("baseline", "ot", "ot"), ("baseline", "ot", "joint"),
                                ("proposed", "vc", "vc"), ("proposed", "ot", "ot"), ("proposed", "ot", "joint")):
        path = OUT / method / role / f"progress_{phase}.json"
        data = read(path) if path.exists() else {}
        lines.append(f'| {method}/{phase} | {data.get("epochs_completed", 0)} | {data.get("epoch_budget", 100 if phase == "joint" else 300)} | {data.get("validation_loss", "—")} |')
    lines += ["", "| 方法 | 模式 | checkpoint | Obj cm | HOI cm | MPJPE cm | root cm |", "|---|---|---|---:|---:|---:|---:|"]
    for method in METHODS:
        for mode in ("offline", "online"):
            for tag in ("final", "best"):
                path = OUT / method / f"eval_{mode}_{tag}.json"
                if path.exists():
                    result = read(path)["results"]["omomo"]
                    values = " | ".join(f'{result[key]:.4f}' for key in ("obj_trans_err_fusion", "hoi_err_fusion", "mpjpe", "root_trans_err"))
                    lines.append(f"| {method} | {mode} | {tag} | {values} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare()
    protocol = read(OUT / "protocol.json")
    state = read(OUT / "status.json")
    state.update(status="running", pid=os.getpid(), error=None, updated_at=previous.now())
    stages = [("baseline_ot_joint", [str(PYTHON), __file__, "worker", "--method", "baseline", "--role", "ot"], OUT / "baseline/ot/completed.json"),
              ("proposed_vc", [str(PYTHON), __file__, "worker", "--method", "proposed", "--role", "vc"], OUT / "proposed/vc/completed.json"),
              ("proposed_ot_joint", [str(PYTHON), __file__, "worker", "--method", "proposed", "--role", "ot"], OUT / "proposed/ot/completed.json")]
    for method in METHODS:
        for tag in ("final", "best"):
            for mode in ("offline", "online"):
                command, expected = eval_command(method, mode, tag)
                stages.append((f"{method}_{mode}_{tag}", command, expected))
    try:
        for stage, command, expected in stages:
            if expected.exists():
                continue
            verify(protocol)
            while True:
                try:
                    gpu = previous.check_gpu(8000)
                    break
                except RuntimeError as exc:
                    state.update(status="waiting_for_gpu", active_stage=stage, error=str(exc), updated_at=previous.now())
                    write(OUT / "status.json", state)
                    report(state)
                    time.sleep(30)
            state.update(status="running", active_stage=stage, gpu=gpu, updated_at=previous.now())
            state["stages"][stage] = {"status": "running", "started_at": previous.now(), "command": command}
            write(OUT / "status.json", state)
            report(state)
            with (OUT / f"{stage}.log").open("a") as handle:
                handle.write(f"\n{previous.now()} COMMAND {json.dumps(command)}\n")
                handle.flush()
                process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONHASHSEED": "10"})
                state.update(child_pid=process.pid)
                write(OUT / "status.json", state)
                while process.poll() is None:
                    report(state)
                    time.sleep(20)
            if process.returncode or not expected.exists():
                raise RuntimeError(f"{stage} failed, exit={process.returncode}; see {OUT / (stage + '.log')}")
            state["stages"][stage].update(status="complete", finished_at=previous.now())
            write(OUT / "status.json", state)
        verify(protocol)
        state.update(status="complete", active_stage=None, child_pid=None, updated_at=previous.now())
        write(OUT / "status.json", state)
        report(state)
    except Exception as exc:
        state.update(status="failed", error=str(exc), updated_at=previous.now())
        if state.get("active_stage") in state["stages"]:
            state["stages"][state["active_stage"]].update(status="failed", error=str(exc))
        write(OUT / "status.json", state)
        report(state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "run"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--role", choices=("vc", "ot"))
    args = parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("Use WSL native SAGE Python")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(hard, 65536)), hard))
    previous.check_gpu(0 if args.mode == "prepare" else 8000)
    os.chdir(ROOT)
    if args.mode == "worker":
        if not args.method or not args.role:
            parser.error("Worker requires method and role")
        worker(args.method, args.role)
    else:
        {"prepare": prepare, "run": run}[args.mode]()


if __name__ == "__main__":
    main()
