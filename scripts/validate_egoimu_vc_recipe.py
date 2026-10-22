#!/usr/bin/env python3
"""Controlled VC retraining; production code and published runs stay immutable."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
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
EGO = ROOT.parent / "EgoIMU"
EXP = ROOT / "outputs/IMUHOI++/egoimu_split_seed10_omomo_20261016"
BRIDGE = ROOT / "outputs/analysis/egoimu_implementation_bridge_20261017"
OUT = ROOT / "outputs/analysis/egoimu_vc_retrain_20261017"
PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")
SMI = "/usr/lib/wsl/lib/nvidia-smi"
ARMS = ("legacy_recipe", "w60_v1", "w60_v10", "w120_v1", "w120_v10")
sys.path.insert(0, str(ROOT))


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_gpu(minimum=8000):
    output = subprocess.check_output([SMI, "--query-gpu=index,name,memory.free,memory.used", "--format=csv,noheader"], text=True)
    print(output.strip(), flush=True)
    candidates = [(int(row.split(",")[2].strip().split()[0]), int(row.split(",")[0])) for row in output.splitlines()]
    free, index = max(candidates)
    if free < minimum:
        raise RuntimeError(f"Only {free} MiB free; require {minimum} MiB before executing code")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    return {"physical_gpu": index, "free_mib": free, "checked_at": now()}


def source_paths():
    paths = []
    for folder in ("model/rnn", "train/rnn", "dataset", "utils"):
        paths.extend(sorted((ROOT / folder).glob("*.py")))
    paths += [ROOT / "eval_IMUHOI.py", ROOT / "eval_batch_online.py", EXP / "recipe.yaml",
              EGO / "outputs/IMUHOI/omomo/training_config.yaml"]
    return paths


def verify_receipt(protocol):
    for name, expected in protocol["source_sha256"].items():
        if sha(name) != expected:
            raise RuntimeError(f"Source changed since protocol freeze: {name}")
    for row in protocol["training_files"]:
        path = EXP / "data/train" / row["file"]
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
            raise RuntimeError(f"Training data changed: {path}")
    if sha(OUT / "initial.pt") != protocol["initial_checkpoint_sha256"]:
        raise RuntimeError("Shared initialization changed")


def prepare():
    import torch
    import yaml
    from easydict import EasyDict
    from model.rnn.velocity_contact_IMUHOI import IMUHOIVelocityContactModule
    from utils.utils import setup_seed

    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        verify_receipt(json.loads(protocol_path.read_text()))
        print("Existing frozen protocol verified", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/analysis/omomo_preprocessing_audit_20261016/per_sequence.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    names = sorted((r["file"] for r in rows if r["old_split"] == "train" and int(r["old_frames"]) >= 121
                    and float(r["old_object_move_fraction"]) * int(r["old_frames"]) >= 9.99999), key=lambda x: int(Path(x).stem))
    if len(names) != 4115:
        raise RuntimeError(f"Expected common 4115 training sequences, got {len(names)}")
    cfg_raw = yaml.safe_load((EXP / "recipe.yaml").read_text())
    cfg_raw.update(velocity_contact_arch="imuhoi", object_trans_arch="imuhoi", hp_ckpt=None,
                   pretrained_ckpt=None, resume_dir=None, seed=10, epochs=300, num_workers=0,
                   use_multi_gpu=False, gpus=[0], device="cuda:0", no_trans=False,
                   use_tensorboard=False, vc_staged_training=False, dataloader_pin_memory=False,
                   early_stopping_patience_epochs=0, eval_interval=5, test_data_window=60,
                   deterministic_test_windows=True)
    setup_seed(10)
    initial = IMUHOIVelocityContactModule(EasyDict(cfg_raw)).state_dict()
    torch.save({"model_state_dict": initial, "seed": 10, "epoch": -1}, OUT / "initial.pt")
    files = []
    for name in names:
        stat = (EXP / "data/train" / name).stat()
        files.append({"file": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    arms = {}
    for arm in ARMS:
        config = json.loads(json.dumps(cfg_raw))
        legacy = arm == "legacy_recipe"
        window = 120 if legacy or arm.startswith("w120") else 60
        weight = 10.0 if legacy or arm.endswith("v10") else 1.0
        config.update(save_dir=str(OUT / arm), train_data_window=window,
                      batch_size=100 if legacy else 800, windows_per_sequence=1 if legacy else 4)
        config["imuhoi_loss_weights"].update(obj_vel=weight, hand_vel=weight, hand_contact=1.0)
        if legacy:
            config.update(vc_grouped_lr=False, milestones=[50, 150], amp_enabled=False, gradient_clip=1e30)
        path = OUT / arm / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        arms[arm] = {"config": str(path), "train_window": window, "velocity_weight": weight,
                     "batch_size": config["batch_size"], "windows_per_sequence": config["windows_per_sequence"],
                     "steps_per_epoch": len(names) * config["windows_per_sequence"] // config["batch_size"],
                     "recompute_features_after_crop": legacy, "epochs": 300, "status": "queued"}
    protocol = {"created_at": now(), "seed": 10, "arms": arms, "training_files": files,
                "training_membership": "Original eligibility: old_frames>=121 and old obj_contact count>=10; current canonical tensors",
                "paired_window_sampling": "Both lengths draw starts from [1,T-120], using the same seed and common file order",
                "primary_checkpoint": "final.pt at zero-based epoch 299; best.pt secondary, selected by existing window loss",
                "primary_evaluation": "All 1177 native cached test sequences; frozen original HP and OT, EgoIMU HOI GT; VC intervention only",
                "secondary_evaluation": "Current baseline final HP/OT fixed; full current offline and batch-online evaluation, EgoIMU HOI protocol",
                "scope": "VC retraining ablations, not a newly trained complete paper reproduction",
                "legacy_recipe_limit": "Original VC recipe on current canonical body tensors and current noise draw implementation; not exact original data/RNG replay. Validation remains common deterministic W60.",
                "source_sha256": {str(p): sha(p) for p in source_paths()},
                "script_sha256": sha(__file__), "initial_checkpoint_sha256": sha(OUT / "initial.pt"),
                "reset": {"authorized_by_user": True, "outcome": "reset", "idempotency_key": "f5352c12-0f87-4e12-ae23-8100581369a5"}}
    write_json(protocol_path, protocol)
    write_json(OUT / "status.json", {"status": "prepared", "updated_at": now(), "arms": arms})
    print("Prepared five arms, 4115 shared members, seed10, shared strict initialization", flush=True)


def train_arm(arm):
    import fcntl
    arm_lock = (OUT / arm / "train.lock").open("w")
    fcntl.flock(arm_lock, fcntl.LOCK_EX)
    if (OUT / arm / "final.pt").exists():
        print(f"Completed fixed-budget training already exists: {arm}", flush=True)
        return
    import torch
    import yaml
    from easydict import EasyDict
    from dataset.dataset_IMUHOI import IMUDataset
    from model.rnn.velocity_contact_IMUHOI import IMUHOIVelocityContactModule
    from train.rnn.loss import IMUHOIVelocityContactLoss
    import train.rnn.train_utils as training
    from train.rnn.train_velocity_contact import VelocityContactTrainer
    from utils.utils import setup_seed, setup_device

    protocol = json.loads((OUT / "protocol.json").read_text())
    verify_receipt(protocol)
    directory = OUT / arm
    cfg = EasyDict(yaml.safe_load((directory / "config.yaml").read_text()))
    cfg.debug = False
    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    paths = [str(EXP / "data/train" / row["file"]) for row in protocol["training_files"]]
    recompute = protocol["arms"][arm]["recompute_features_after_crop"]

    class ControlledDataset(IMUDataset):
        def __init__(self, *args, **kwargs):
            data_dirs = kwargs.get("data_dir", [])
            is_train = any(Path(p).resolve() == (EXP / "data/train").resolve() for p in data_dirs)
            if is_train:
                kwargs["sequence_paths"] = paths
            super().__init__(*args, **kwargs)
            if is_train:
                if len(self) != 4115:
                    raise RuntimeError(f"Incomplete common training membership: {len(self)}")
                # __getitem__ uses seq_len only to choose the crop bounds.
                # Restrict W60 starts to the same range as W120, without editing tensors.
                for info in self.sequence_info:
                    info["actual_full_frames"] = info["seq_len"]
                    info["seq_len"] -= 120 - self.window_size
                if recompute:
                    keys = ("human_imu", "obj_imu", "ori_root_reduced", "root_vel", "obj_vel",
                            "sensor_vel_root", "sensor_vel_glb", "obj_rot_6d")
                    for sequence in self.loaded_data.values():
                        for key in keys:
                            sequence.pop(key, None)

    training.IMUDataset = ControlledDataset
    train_loader, test_loader = training.create_dataloaders(cfg)
    expected_steps = protocol["arms"][arm]["steps_per_epoch"]
    if len(train_loader) != expected_steps:
        raise RuntimeError(f"Optimizer budget mismatch {len(train_loader)} != {expected_steps}")
    model = IMUHOIVelocityContactModule(cfg).to(cfg.device)
    initial = torch.load(OUT / "initial.pt", map_location=cfg.device, weights_only=False)["model_state_dict"]
    model.load_state_dict(initial, strict=True)
    if not all(torch.equal(value, initial[key]) for key, value in model.state_dict().items()):
        raise RuntimeError("Shared initialization verification failed")
    write_json(directory / "initialization.json", {"initial_checkpoint_sha256": protocol["initial_checkpoint_sha256"],
               "strict_load": True, "train_sequences": len(train_loader.dataset), "validation_sequences": len(test_loader.dataset),
               "steps_per_epoch": len(train_loader), "verified_at": now()})
    progress = {"arm": arm, "status": "training", "epoch_budget": cfg.epochs, "started_at": now()}

    class ReportingTrainer(VelocityContactTrainer):
        def train_epoch(self, epoch):
            started = time.monotonic()
            loss, components = super().train_epoch(epoch)
            if not math.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
            progress.update(epoch=epoch, epochs_completed=epoch + 1, optimizer_steps=self.n_iter,
                            train_loss=loss, train_components=components,
                            epoch_seconds=time.monotonic() - started, updated_at=now())
            write_json(directory / "progress.json", progress)
            return loss, components

        def evaluate_all_losses(self, epoch):
            result = super().evaluate_all_losses(epoch)
            if result[0] is not None and not math.isfinite(result[0]):
                raise RuntimeError(f"Nonfinite validation loss at epoch {epoch}")
            progress.update(last_validation_epoch=epoch, last_validation_loss=result[0],
                            last_validation_components=result[2], updated_at=now())
            write_json(directory / "progress.json", progress)
            return result

    if (directory / "last.pt").exists() and not (directory / "final.pt").exists():
        cfg.resume_dir = str(directory)
    trainer = ReportingTrainer(cfg, model, IMUHOIVelocityContactLoss(weights=cfg.imuhoi_loss_weights),
                               train_loader, test_loader, hp_model=None)
    trainer.train()
    checkpoint = torch.load(directory / "final.pt", map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 299 or checkpoint["n_iter"] != expected_steps * 300:
        raise RuntimeError("Final checkpoint failed fixed-budget verification")
    verify_receipt(protocol)
    progress.update(status="training_complete", completed_at=now(), final_epoch=checkpoint["epoch"])
    write_json(directory / "progress.json", progress)


def evaluate_component(arm):
    import torch
    import yaml
    from easydict import EasyDict
    from model.rnn.velocity_contact_IMUHOI import IMUHOIVelocityContactModule
    from model.rnn.object_trans_IMUHOI import IMUHOIObjectTransModule
    from utils.hoi_metrics import compute_hoi_error_egoimu

    protocol = json.loads((OUT / "protocol.json").read_text())
    verify_receipt(protocol)
    directory = OUT / arm
    cfg = EasyDict(yaml.safe_load((directory / "config.yaml").read_text()))
    device = torch.device("cuda:0")
    vc = IMUHOIVelocityContactModule(cfg).to(device).eval()
    checkpoint = torch.load(directory / "final.pt", map_location=device, weights_only=False)
    vc.load_state_dict(checkpoint["model_state_dict"], strict=True)
    ot = IMUHOIObjectTransModule(cfg).to(device).eval()
    state = torch.load(EGO / "outputs/IMUHOI/omomo/modules/object_trans_best.pt", map_location=device, weights_only=False)
    ot.load_state_dict(state["module_state_dict"], strict=True)
    entries = json.loads((BRIDGE / "native_manifest.json").read_text())["entries"]
    rows = []
    with torch.inference_mode():
        for index, entry in enumerate(entries):
            cache = torch.load(BRIDGE / "native_cache" / entry["file"], map_location="cpu", weights_only=False)
            data = {key: value.to(device) for key, value in cache["data"].items()}
            pred = vc(data)
            velocity = pred["pred_obj_vel"][0]
            target = cache["target"]["obj_trans"][0].to(device)
            error = velocity - cache["target"]["obj_vel"][0].to(device)
            displacement = velocity.cumsum(0) / 30
            integral = target[0] + torch.cat((displacement.new_zeros(1, 3), displacement[:-1]), 0)
            fused = ot(cache["pred"]["pred_hand_glb_pos"].to(device), pred["pred_hand_contact_prob"],
                       obj_trans_init=data["obj_trans_init"], obj_imu=data["obj_imu"], human_imu=data["human_imu"],
                       contact_init=data["contact_init"], has_object_mask=data["has_object"], obj_vel_input=pred["pred_obj_vel"])["pred_obj_trans"][0]
            hoi = compute_hoi_error_egoimu(fused, target, cache["pred_joints"].to(device), cache["gt_joints"].to(device),
                                          cache["target"]["lhand_contact"][0].to(device), cache["target"]["rhand_contact"][0].to(device))
            rows.append({"file": entry["file"], "velocity_rmse_mps": error.square().mean().sqrt().item(),
                         "bias_norm_mps": error.mean(0).norm().item(),
                         "integral_obj_cm": (integral - target).norm(dim=-1).mean().item() * 100,
                         "end_obj_cm": (integral[-1] - target[-1]).norm().item() * 100,
                         "frozen_original_hp_ot_obj_cm": (fused - target).norm(dim=-1).mean().item() * 100,
                         "frozen_original_hp_ot_hoi_cm": float(hoi)})
            if (index + 1) % 100 == 0:
                print(f"COMPONENT EVAL {arm} {index+1}/{len(entries)}", flush=True)
    if len(rows) != 1177:
        raise RuntimeError("Incomplete component evaluation")
    metrics = {key: sum(r[key] for r in rows) / len(rows) for key in rows[0] if key != "file"}
    write_json(directory / "component_final.json", {"arm": arm, "n": len(rows), "epoch": checkpoint["epoch"],
               "metrics": metrics, "scope": protocol["primary_evaluation"], "completed_at": now()})
    write_json(directory / "component_sequences.json", rows)
    print(json.dumps(metrics), flush=True)


def full_eval_command(arm, mode):
    manifest = json.loads((EXP / "baseline/experiment_manifest.json").read_text())
    pipeline = manifest["pipelines"]["vc_imuhoi_ot_3branch"]
    stage = Path(pipeline["stage_dir"])
    entry = "eval_IMUHOI.py" if mode == "offline" else "eval_batch_online.py"
    command = [str(PYTHON), str(ROOT / entry), "--config", pipeline["config"], "--dataset", "omomo",
               "--hp_ckpt", str(stage / "final_human_pose.pt"), "--ot_ckpt", str(stage / "final_object_trans.pt"),
               "--vc_ckpt", str(OUT / arm / "final.pt"), "--velocity_contact_arch", "imuhoi",
               "--object_trans_arch", "imuhoi", "--num_workers", "0", "--seed", "10",
               "--output_json", str(OUT / arm / f"current_pipeline_{mode}_final.json")]
    if mode == "offline":
        command += ["--inference_mode", "offline"]
    else:
        command += ["--online_batch_size", "24"]
    return command


def update_summary(state):
    lines = ["# VC 受控重训验证", "", f"更新：{now()}", "", "seed=10，共同训练成员4115；所有组共享初始权重，固定300 epoch，末轮为主结果。",
             "恢复配方组是组合干预；其余四组为窗口×速度权重的2×2对照。原版HP/OT固定的组件结果与当前HP/OT固定的整流水线结果分别报告。", "",
             "| 实验 | 状态 | epoch/300 | 原版HP/OT固定 Obj | 速度偏置 m/s | 当前 offline Obj/HOI | 当前 online Obj/HOI |",
             "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        directory = OUT / arm
        progress = json.loads((directory / "progress.json").read_text()) if (directory / "progress.json").exists() else {}
        component = json.loads((directory / "component_final.json").read_text()).get("metrics", {}) if (directory / "component_final.json").exists() else {}
        scores = []
        for mode in ("offline", "online"):
            path = directory / f"current_pipeline_{mode}_final.json"
            if path.exists():
                result = json.loads(path.read_text())["results"]["omomo"]
                scores.append(f'{result["obj_trans_err_fusion"]:.4f}/{result["hoi_err_fusion"]:.4f}')
            else:
                scores.append("—")
        obj = f'{component["frozen_original_hp_ot_obj_cm"]:.4f}' if component else "—"
        bias = f'{component["bias_norm_mps"]:.6f}' if component else "—"
        lines.append(f'| {arm} | {state["arms"][arm]["status"]} | {progress.get("epochs_completed", 0)}/300 | {obj} | {bias} | {scores[0]} | {scores[1]} |')
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def run_queue():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare()
    protocol = json.loads((OUT / "protocol.json").read_text())
    state = json.loads((OUT / "status.json").read_text())
    state.update(status="running", pid=os.getpid(), updated_at=now())
    write_json(OUT / "status.json", state)
    for arm in ARMS:
        steps = [("training", [str(PYTHON), __file__, "train", "--arm", arm], OUT / arm / "final.pt"),
                 ("component_eval", [str(PYTHON), __file__, "component_eval", "--arm", arm], OUT / arm / "component_final.json")]
        for mode in ("offline", "online"):
            steps.append((f"current_{mode}_eval", full_eval_command(arm, mode), OUT / arm / f"current_pipeline_{mode}_final.json"))
        try:
            verify_receipt(protocol)
            for phase, command, expected in steps:
                if expected.exists():
                    continue
                gpu = check_gpu()
                state.update(active_arm=arm, phase=phase, updated_at=now())
                state["arms"][arm].update(status=phase, gpu=gpu)
                write_json(OUT / "status.json", state)
                update_summary(state)
                log = OUT / arm / ("train.log" if phase == "training" else f"{phase}.log")
                with log.open("a") as handle:
                    handle.write(f"\n{now()} COMMAND {json.dumps(command)}\n")
                    handle.flush()
                    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                               env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
                    state.update(child_pid=process.pid)
                    write_json(OUT / "status.json", state)
                    code = process.wait()
                if code != 0 or not expected.exists():
                    raise RuntimeError(f"{arm}/{phase} exit={code}; see {log}")
            state["arms"][arm].update(status="complete", completed_at=now())
        except Exception as exc:
            state["arms"][arm].update(status="failed", error=str(exc), failed_at=now())
            print(traceback.format_exc(), flush=True)
        state.update(updated_at=now())
        write_json(OUT / "status.json", state)
        update_summary(state)
    state.update(status="complete" if all(a["status"] == "complete" for a in state["arms"].values()) else "completed_with_failures",
                 updated_at=now(), active_arm=None, child_pid=None)
    write_json(OUT / "status.json", state)
    update_summary(state)


def run_factorial_training():
    """Use spare GPU capacity while the smaller legacy batches train."""
    import fcntl
    lock = (OUT / "factorial_train.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    auxiliary = {"status": "running", "pid": os.getpid(), "started_at": now(),
                 "scope": "Training only; the primary queue owns every evaluation"}
    for arm in ARMS[1:]:
        if (OUT / arm / "final.pt").exists():
            continue
        auxiliary.update(active_arm=arm, updated_at=now())
        write_json(OUT / "factorial_status.json", auxiliary)
        while True:
            try:
                check_gpu()
                break
            except RuntimeError as exc:
                auxiliary.update(status="waiting_for_gpu", resource=str(exc), updated_at=now())
                write_json(OUT / "factorial_status.json", auxiliary)
                time.sleep(30)
        auxiliary.update(status="training", updated_at=now())
        write_json(OUT / "factorial_status.json", auxiliary)
        with (OUT / arm / "train.log").open("a") as handle:
            command = [str(PYTHON), __file__, "train", "--arm", arm]
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
            auxiliary.update(child_pid=process.pid)
            write_json(OUT / "factorial_status.json", auxiliary)
            code = process.wait()
        if code:
            auxiliary.update(status="failed", error=f"{arm} exited {code}", updated_at=now())
            write_json(OUT / "factorial_status.json", auxiliary)
            raise RuntimeError(auxiliary["error"])
    auxiliary.update(status="training_complete", active_arm=None, child_pid=None, updated_at=now())
    write_json(OUT / "factorial_status.json", auxiliary)
    # Reconcile any resource-contention failures only after the primary queue exits.
    main_lock = (OUT / "queue.lock").open("a")
    fcntl.flock(main_lock, fcntl.LOCK_EX)
    main_lock.close()
    run_queue()
    auxiliary.update(status="complete", updated_at=now())
    write_json(OUT / "factorial_status.json", auxiliary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "component_eval", "run", "factorial_train"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("Use WSL native SAGE Python")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(hard, 65536)), hard))
    check_gpu(minimum=0 if args.mode == "factorial_train" else 8000)
    os.chdir(ROOT)
    if args.mode in ("train", "component_eval") and not args.arm:
        parser.error("--arm required")
    {"prepare": prepare, "train": lambda: train_arm(args.arm), "component_eval": lambda: evaluate_component(args.arm),
     "run": run_queue, "factorial_train": run_factorial_training}[args.mode]()


if __name__ == "__main__":
    main()
