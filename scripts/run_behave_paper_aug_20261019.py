#!/usr/bin/env python3
"""Train augmented IMUHOI++ on original BEHAVE memberships and paper stage budgets."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_egoimu_vc_recipe as common

OUT = ROOT / "outputs/IMUHOI++/paper_recipe_aug_seed10_behave_20261019"
EXP, PYTHON = OUT, common.PYTHON
EGO = ROOT.parent / "EgoIMU"
STAGES = [("shared", "hp"), ("proposed", "vc"), ("proposed", "ot"), ("proposed", "joint")]
write, sha, now = common.write_json, common.sha, common.now


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def original_config():
    import yaml
    class SavedConfigLoader(yaml.SafeLoader):
        pass
    def easydict(loader, node):
        return loader.construct_mapping(node, deep=True)["dictitems"]
    SavedConfigLoader.add_constructor("tag:yaml.org,2002:python/object/new:easydict.EasyDict", easydict)
    return yaml.load((EGO / "outputs/IMUHOI/behave/training_config.yaml").read_text(), Loader=SavedConfigLoader)


def verify(protocol, full=False):
    for path, digest in protocol["frozen_sha256"].items():
        if sha(path) != digest:
            raise RuntimeError(f"Frozen file changed: {path}")
    for split, rows in protocol["data"].items():
        for row in rows:
            path = EXP / "data" / split / row["name"]
            st = path.stat()
            if [st.st_size, st.st_mtime_ns] != row["stat"] or (full and sha(path) != row["sha256"]):
                raise RuntimeError(f"Data changed: {path}")


def prepare():
    import yaml
    if (OUT / "protocol.json").exists():
        verify(read(OUT / "protocol.json")); return
    audit = read(OUT / "data_audit.json")
    assert audit.get('removed_derivative_caches'), 'Finalize derivative caches before freezing'
    saved = original_config()
    stages = {s["name"]: s for s in saved["staged_training"]["stages"]}
    names = {"hp":"human_pose", "vc":"velocity_contact", "ot":"object_trans", "joint":"joint_training"}
    reference = ROOT / "outputs/IMUHOI++/original_recipe_k1_seed10_omomo_20261018"
    configs, budget = {}, {}
    for method, phase in STAGES:
        source = reference / method / phase / "recipe.yaml"
        cfg = yaml.safe_load(source.read_text())
        stage = stages[names[phase]]
        cfg.update(seed=10, train_datasets=["behave"], save_dir=str(OUT/method/phase),
                   epochs=stage["epochs"], batch_size=stage["batch_size"],
                   lr=stage.get("lr",saved["lr"]), weight_decay=stage.get("weight_decay",saved["weight_decay"]),
                   milestones=stage["milestones"], gamma=stage["gamma"], num_workers=12,
                   body_model_path="datasets/smpl_models/smplh/neutral/model.npz",
                   pretrained_ckpt=None, resume_dir=None, hp_ckpt=None, vc_ckpt=None,
                   windows_per_sequence=1, eval_hoi_protocol="egoimu")
        entry = {"body_model_type":"smplh", "train_path":str(OUT/"data/train"),
                 "test_path":str(OUT/"data/test"), "debug_path":str(OUT/"data/train"),
                 "full_sequence_test_path":str(OUT/"data/eval"), "object_geo_root":"datasets/BEHAVE/objects"}
        cfg["datasets"]["behave"] = entry
        cfg["datasets"]["behave_full"] = copy.deepcopy(entry)
        assert cfg["v5_init_noise_prob"]==.5 and cfg["v5_obj_vel_init_noise_std"]==cfg["v5_hand_vel_init_noise_std"]==.2
        assert cfg["loss_weights"]["obj_vel"]==cfg["loss_weights"]["hand_vel"]==10
        assert all(cfg['loss_weights'][k]==v for k,v in saved['loss_weights'].items())
        path=OUT/method/phase/"recipe.yaml"; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        configs[f"{method}_{phase}"]=str(path)
        steps=len(audit["members"]["train"])//cfg["batch_size"]
        budget[phase]={"batch_size":cfg["batch_size"],"epochs":300,"steps_per_epoch":steps,
                       "updates":steps*300,"windows_used_per_epoch":steps*cfg["batch_size"],
                       "lr":cfg["lr"],"milestones":cfg["milestones"],"gamma":cfg["gamma"]}
    inventory=read(OUT/"data/inventory.json")
    frozen=[Path(p) for p in configs.values()]+[Path(__file__), ROOT/"scripts/prepare_behave_paper_data_20261019.py", ROOT/"scripts/finalize_behave_paper_data_20261019.py",
        ROOT/"scripts/eval_behave_paper_checked_20261019.py", EGO/"outputs/IMUHOI/behave/training_config.yaml",
        OUT/"data_audit.json", OUT/"canonical_recovery.json", OUT/"evaluation_members.json", ROOT/"scripts/prepare_behave_eval_members_20261019.py"]
    for folder in ("model/rnn","train/rnn","dataset","utils","configs"):
        frozen.extend((ROOT/folder).rglob("*.py"))
    frozen.extend([ROOT/"eval_IMUHOI.py", ROOT/"eval_batch_online.py", OUT/"paper_reference.txt"])
    protocol={"created_at":now(),"seed":10,"configs":configs,"budget":budget,"members":audit["members"],
        "data":inventory,"frozen_sha256":{str(p):sha(p) for p in frozen},"original_config":saved,
        "checkpoint_policy":"Window-validation best primary; final300 supplementary. Downstream stages load preceding best, no test-result checkpoint search.",
        "sampler":"K=1 without replacement; one random W120 crop per sequence per epoch, drop_last=True",
        "method":"Native vc-v5ar + ot-v2armbic; initializer augmentation p0.5 sigma0.2m/s; velocity120/contact180; joint VC+OT with HP frozen",
        "noise":"Original IMU Gaussian sigma0.1 during training/window validation; no added full-evaluation noise",
        "evaluation":"340 original motion-eligible BEHAVE test crops; all original 387 saved; offline primary, online W60/lookahead10/chunk16; EgoIMU wrist HOI",
        "paper_reference_pdf":str(ROOT.parent.parent/"paper_work/IMUHOI-CVPR/40509.pdf"),"paper_targets":{"table":2,"dataset":"BEHAVE","obj_trans_err_fusion":20.90,"hoi_err_fusion":25.74,"mpjpe":2.77,"root_trans_err":11.14,"jitter":8.19},
        "remaining_differences":["Current neutral zero-shape canonical SMPL-H preprocessing, motion/contact labels and HP physics retained; not an exact original-data reproduction.",
            "Original memberships, segment frame boundaries and original eligibility restored; four missing sources regenerated with current preprocessor.",
            "Initialization uses per-stage seed10, not bitwise original RNG replay. Original test split also supplies window validation, matching saved recipe.",
            "Common paper stage budgets and global LR schedules match the prior OMOMO comparison; method-specific augmentation, contact, palm, bimanual and human refinement losses retained."]}
    verify(protocol,full=True)
    write(OUT/"protocol.json",protocol)
    write(OUT/"status.json",{"status":"prepared","updated_at":now(),"stages":{}})
    write(OUT/"paper_targets.json",protocol["paper_targets"])
    print(json.dumps({"counts":audit["counts"],"budget":budget}),flush=True)

def make_loaders(cfg, protocol):
    import torch
    from torch.utils.data import DataLoader
    from dataset.dataset_IMUHOI import IMUDataset
    datasets = []
    for split in ("train", "test"):
        ds = IMUDataset(data_dir=[str(EXP / "data" / split)], window_size=120,
            sequence_paths=[str(EXP / "data" / split / name) for name in protocol["members"][split]],
            debug=False, random_window=True, simulate_imu_noise=False, min_obj_contact_frames=0,
            require_canonical_skeleton=True, resolve_bimanual_contact_conflicts=False,
            obj_points_sample_count=int(cfg.mesh_downsample_points))
        assert len(ds) == len(protocol["members"][split])
        # Recompute derivatives after cropping, matching original crop-before-features order.
        for sequence in ds.loaded_data.values():
            for key in ("human_imu", "obj_imu", "ori_root_reduced", "root_vel", "obj_vel",
                        "sensor_vel_root", "sensor_vel_glb", "obj_rot_6d"):
                sequence.pop(key, None)
        datasets.append(ds)
    return tuple(DataLoader(ds, batch_size=cfg.batch_size, shuffle=i == 0,
                           num_workers=cfg.num_workers, drop_last=i == 0,
                           pin_memory=False, persistent_workers=False,
                           generator=torch.Generator().manual_seed(10)) for i, ds in enumerate(datasets))


def load_weights(model, path, device):
    import torch
    from train.rnn.train_utils import load_model_state_from_checkpoint
    payload = torch.load(path, map_location=device, weights_only=False)
    load_model_state_from_checkpoint(model, payload, strict=True, name=type(model).__name__)


def init_models(cfg, method, phase, directory, dependencies=True):
    import torch
    from model import HumanPoseModule
    from model.rnn.velocity_contact import build_velocity_contact_module
    from model.rnn.object_trans_IMUHOI import IMUHOIObjectTransModule
    from model.rnn.object_trans_v2 import V2ObjectTransModule
    from utils.utils import setup_seed
    device = torch.device(cfg.device)
    setup_seed(10)
    hp = HumanPoseModule(cfg, device, no_trans=False).to(device)
    if phase == "hp":
        return hp, None, None
    if dependencies:
        load_weights(hp, OUT / "shared/hp/best.pt", device)
    setup_seed(10)
    vc = build_velocity_contact_module(cfg).to(device)
    if phase == "vc":
        # Both architectures contain identical velocity heads; share seed10 initializer, never trained weights.
        from model.rnn.velocity_contact_IMUHOI import IMUHOIVelocityContactModule
        setup_seed(10)
        base = IMUHOIVelocityContactModule(cfg).to(device)
        for name in ("hand_vel_net", "obj_vel_net"):
            getattr(vc, name).load_state_dict(getattr(base, name).state_dict(), strict=True)
        del base
        return hp, vc, None
    if dependencies:
        load_weights(vc, OUT / method / "vc/best.pt", device)
    setup_seed(10)
    # Exact native class identity matters to v2 closure/filter checks; do not subclass the model.
    ot = (IMUHOIObjectTransModule(cfg) if method == "baseline" else V2ObjectTransModule(cfg)).to(device)
    if phase == "joint" and dependencies:
        load_weights(ot, OUT / method / "ot/best.pt", device)
    return hp, vc, ot


def worker(method, phase, smoke=False):
    import torch
    import yaml
    from easydict import EasyDict
    from utils.utils import setup_seed
    from train.rnn.train_human_pose import HumanPoseTrainer
    from train.rnn.train_velocity_contact import VelocityContactTrainer
    from train.rnn.train_object_trans import Stage3JointTrainer
    from train.rnn.train_utils import build_model_input_dict, save_config_snapshot
    from train.rnn.loss import HumanPoseLoss, IMUHOIVelocityContactLoss, VelocityContactLoss
    protocol = read(OUT / "protocol.json")
    verify(protocol)
    cfg = EasyDict(yaml.safe_load(Path(protocol["configs"][f"{method}_{phase}"]).read_text()))
    directory = Path(cfg.save_dir)
    if smoke:
        directory = OUT / "preflight" / method / phase
        directory.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(directory)
    lock = (directory / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    last = directory / ("last_joint.pt" if phase == "joint" else "last.pt")
    cfg.resume_dir = str(directory) if last.exists() else None
    setup_seed(10)
    train_loader, test_loader = make_loaders(cfg, protocol)
    hp, vc, ot = init_models(cfg, method, phase, directory, dependencies=not smoke)
    expected = protocol["budget"][phase]
    assert len(train_loader) == expected["steps_per_epoch"]
    assert not train_loader.sampler.replacement and cfg.windows_per_sequence == 1
    # Audit the sampler without consuming the generator used by training.
    generator_state = train_loader.generator.get_state()
    sampled_indices = list(iter(train_loader.sampler))
    train_loader.generator.set_state(generator_state)
    assert len(sampled_indices) == len(set(sampled_indices)) == len(protocol["members"]["train"])
    write(directory / "loader_receipt.json", {"train_sequences": len(train_loader.dataset),
        "validation_sequences": len(test_loader.dataset), "steps_per_epoch": len(train_loader),
        "batch_size": cfg.batch_size, "windows_per_sequence": 1, "replacement": False,
        "drop_last": train_loader.drop_last, "workers": cfg.num_workers, "unique_sequence_indices_per_epoch": len(set(sampled_indices)),
        "window_train_validation": [120, 120], "crop_before_feature_recomputation": True})
    progress = {"method": method, "phase": phase, "pid": os.getpid(), "status": "training",
                "epoch_budget": cfg.epochs, "started_at": now()}

    def record(trainer, epoch, loss, components, validation=False):
        if loss is not None and not math.isfinite(loss):
            raise RuntimeError(f"Nonfinite {method}/{phase} loss")
        progress.update(updated_at=now(), optimizer_steps=trainer.n_iter,
            learning_rates=[g["lr"] for g in trainer.optimizer.param_groups],
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        if validation:
            progress.update(validation_epoch=epoch, validation_loss=loss, validation_components=components)
        else:
            progress.update(epoch=epoch, epochs_completed=epoch+1, train_loss=loss)
        write(directory / "progress.json", progress)

    class Reporting:
        def train_epoch(self, epoch):
            self.train_loader.generator.manual_seed(10 + epoch)
            expected_lr = cfg.lr * cfg.gamma ** sum(epoch >= m for m in cfg.milestones)
            assert all(math.isclose(g["lr"], expected_lr, rel_tol=1e-6) for g in self.optimizer.param_groups)
            value = super().train_epoch(epoch)
            assert self.n_iter == (epoch+1)*expected["steps_per_epoch"]
            record(self, epoch, *value)
            return value
        def evaluate(self, epoch):
            self.test_loader.generator.manual_seed(10 + epoch)
            value = super().evaluate(epoch)
            record(self, epoch, *value, validation=True)
            return value

    class HPTrainer(Reporting, HumanPoseTrainer):
        pass

    class VCTrainer(Reporting, VelocityContactTrainer):
        def build_optimizer_and_scheduler(self):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, foreach=False)
            return optimizer, torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
        def evaluate_all_losses(self, epoch):
            self.test_loader.generator.manual_seed(10 + epoch)
            value = super().evaluate_all_losses(epoch)
            record(self, epoch, value[0], value[2], validation=True)
            return value

    class OTTrainer(Reporting, Stage3JointTrainer):
        def evaluate(self, epoch):
            # Original joint selection sums active-module TEST terms, not the current HOI-only selector.
            self.test_loader.generator.manual_seed(10 + epoch)
            self.hp_model.eval(); self.vc_model.eval(); self.ot_model.eval()
            total, components = 0., {}
            with torch.no_grad():
                for batch in self.test_loader:
                    data = build_model_input_dict(batch, cfg, self.device, add_noise=True)
                    _, vc_out, hp_out, ot_out = self.forward_all(data, batch=batch, epoch=epoch, training=False)
                    _, _, weighted = self.ot_loss_fn.compute_loss(ot_out, batch, self.device)
                    keys = ("lhand_obj_direction", "rhand_obj_direction", "lhand_lb", "rhand_lb", "hoi_error_l", "hoi_error_r")
                    selected = {k: weighted[k] for k in keys}
                    if self.hp_trainable:
                        _, _, values = self.hp_loss_fn(hp_out, batch, self.device)
                        selected.update({"hp_"+k: values[k] for k in ("pose_reduced", "root_trans")})
                    if self.vc_trainable:
                        vc_loss, _ = self.vc_loss_fn.compute_test_loss(vc_out, batch, self.device)
                        selected["vc_test"] = vc_loss
                    total += sum(selected.values()).item()
                    for k, v in selected.items():
                        components[k] = components.get(k, 0.) + v.item()
            value = total / len(self.test_loader), {k: v/len(self.test_loader) for k, v in components.items()}
            record(self, epoch, *value, validation=True)
            return value

    if phase == "hp":
        trainer = HPTrainer(cfg, hp, HumanPoseLoss(weights=cfg.loss_weights, no_trans=False), train_loader, test_loader)
    elif phase == "vc":
        # Disable the current unconditional VC norm clipping locally for this recipe.
        import train.rnn.train_velocity_contact as vc_entry
        vc_entry.clip_grad_norm_ = lambda *args, **kwargs: None
        loss = IMUHOIVelocityContactLoss(weights=cfg.imuhoi_loss_weights) if method == "baseline" else VelocityContactLoss(
            weights=cfg.loss_weights, anticipation_horizon=cfg.v4_anticipation_horizon)
        trainer = VCTrainer(cfg, vc, loss, train_loader, test_loader, hp_model=hp)
    else:
        # Original model.train() keeps dropout active even in frozen HP/VC.
        # Restore layer stochasticity without enabling unrelated v5 train-only augmentations.
        def restore_dropout(module):
            original_forward = module.forward
            def forward(*args, **kwargs):
                for layer in module.modules():
                    if isinstance(layer, (torch.nn.Dropout, torch.nn.RNNBase)):
                        layer.train(ot.training)
                return original_forward(*args, **kwargs)
            module.forward = forward
        restore_dropout(hp)
        restore_dropout(vc)
        if method == "baseline":
            original = hp.forward
            def wrist_forward(*args, **kwargs):
                value = dict(original(*args, **kwargs))
                value["pred_palm_glb_pos"] = value["pred_hand_glb_pos"]
                hp._expected_wrists = value["pred_hand_glb_pos"]
                return value
            hp.forward = wrist_forward
            forward_ot = ot.forward
            def checked_forward(*args, **kwargs):
                difference = (args[0] - hp._expected_wrists).abs().max().item()
                if difference != 0:
                    raise RuntimeError("Baseline OT training must use wrists")
                if not (directory / "anchor.json").exists():
                    write(directory / "anchor.json", {"anchor": "wrists20_21", "max_abs_difference": difference})
                return forward_ot(*args, **kwargs)
            ot.forward = checked_forward
        trainer = OTTrainer(cfg, vc, hp, ot, train_loader, test_loader, joint_train=phase == "joint")
        trainer.joint_checkpoint_selection_keys = ("lhand_obj_direction", "rhand_obj_direction", "lhand_lb", "rhand_lb", "hoi_error_l", "hoi_error_r")
    # Match the original AdamW implementation path for every stage.
    for group in trainer.optimizer.param_groups:
        group["foreach"] = False
    save_config_snapshot(cfg)
    write(directory / "effective_recipe.json", {"optimizer": type(trainer.optimizer).__name__,
        "optimizer_groups": [{k: v for k, v in g.items() if k != "params"} for g in trainer.optimizer.param_groups],
        "scheduler": type(trainer.scheduler).__name__, "milestones": dict(trainer.scheduler.milestones),
        "amp_enabled": trainer.amp_enabled, "gradient_clipping": False,
        "frozen_hp_vc_dropout_in_ot_training": phase in ("ot", "joint"),
        "epochs": cfg.epochs, "budget": expected, "checkpoint_selection": protocol["checkpoint_policy"],
        "loss_weights": cfg.loss_weights, "imuhoi_loss_weights": cfg.imuhoi_loss_weights,
        "trainable_modules": {"hp": phase == "hp" or getattr(trainer, "hp_trainable", False),
                              "vc": phase == "vc" or getattr(trainer, "vc_trainable", False),
                              "ot": phase in ("ot", "joint")}})
    if smoke:
        class OneBatch:
            def __init__(self, loader):
                self.generator = loader.generator
                self.batch = next(iter(loader))
            def __len__(self):
                return 1
            def __iter__(self):
                yield self.batch
        trainer.train_loader = OneBatch(train_loader)
        trainer.test_loader = OneBatch(test_loader)
        loss, _ = super(Reporting, trainer).train_epoch(0)
        assert math.isfinite(loss)
        assert all(torch.isfinite(p).all() for g in trainer.optimizer.param_groups for p in g["params"])
        validation = trainer.evaluate_all_losses(0) if phase == "vc" else trainer.evaluate(0)
        # Also exercise the proposed contact phase, which uses different heads and losses.
        contact_loss = None
        if method == "proposed" and phase == "vc":
            contact_loss, _ = super(Reporting, trainer).train_epoch(120)
            assert math.isfinite(contact_loss)
        result = {"status": "passed", "method": method, "phase": phase, "batch_size": cfg.batch_size,
                  "train_loss": loss, "validation_loss": validation[0], "contact_phase_smoke_loss": contact_loss,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
                  "scope": "Disposable random models; one actual full-size forward/backward batch and validation, no weights reused"}
        write(directory / "smoke.json", result)
        print(json.dumps(result), flush=True)
        return
    trainer.train()
    final = directory / ("final_joint.pt" if phase == "joint" else "final.pt")
    payload = torch.load(final, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 299 and payload["n_iter"] == expected["updates"]
    verify(protocol)
    progress.update(status="complete", completed_at=now())
    write(directory / "progress.json", progress)
    write(directory / "completed.json", {"epoch": 299, "updates": payload["n_iter"], "final_sha256": sha(final), "completed_at": now()})


def eval_command(method, mode, tag):
    protocol = read(OUT / "protocol.json")
    directory = OUT / method / "joint"
    hp = directory / f"{tag}_human_pose.pt" if method == "baseline" else OUT / "shared/hp/best.pt"
    vc = OUT / "baseline/vc/best.pt" if method == "baseline" else directory / f"{tag}_velocity_contact.pt"
    output = OUT / method / f"eval_{mode}_{tag}.json"
    command = [str(PYTHON), str(ROOT / "scripts/eval_behave_paper_checked_20261019.py"), mode,
        "--config", protocol["configs"][f"{method}_joint"], "--dataset", "behave", "--hp_ckpt", str(hp),
        "--vc_ckpt", str(vc), "--ot_ckpt", str(directory / f"{tag}_object_trans.pt"),
        "--velocity_contact_arch", "imuhoi" if method == "baseline" else "v5",
        "--object_trans_arch", "imuhoi" if method == "baseline" else "v2", "--seed", "10",
        "--num_workers", "0", "--output_json", str(output)]
    command += ["--inference_mode", "offline"] if mode == "offline" else [
        "--online_batch_size", "24", "--online_window", "60", "--online_lookahead", "10", "--online_chunk", "16"]
    return command, output


def report(state):
    protocol=read(OUT/"protocol.json")
    counts=read(OUT/"data_audit.json")["counts"]
    lines=["# BEHAVE IMUHOI++：开启增广，论文配方比较","",f'状态：{state["status"]}；更新：{now()}',"",
           "seed10，K=1，W120；HP/VC/OT/joint从头训练，各300轮。VC保留120轮速度+180轮接触课程，初值增广p0.5、sigma0.2m/s。",
           f"原片段划分及边界恢复：{counts}。当前canonical SMPL-H预处理保留。","",
           "| 阶段 | 完成轮数/300 | 更新数 | 最近验证loss |","|---|---:|---:|---:|"]
    for method,phase in STAGES:
        path=OUT/method/phase/"progress.json"; p=read(path) if path.exists() else {}
        lines.append(f'| {method}/{phase} | {p.get("epochs_completed",0)} | {p.get("optimizer_steps",0)} | {p.get("validation_loss","—")} |')
    lines += ["","论文Table2位置指标单位cm；best为窗口验证选择，final为固定300轮。",
              "| 条件 | Obj | HOI | MPJPE | Root | Jitter |","|---|---:|---:|---:|---:|---:|",
              "| 论文 offline | 20.90 | 25.74 | 2.77 | 11.14 | 8.19 |"]
    available={}
    for tag in ("best","final"):
        for mode in ("offline","online"):
            path=OUT/"proposed"/f"eval_{mode}_{tag}.json"
            if path.exists():
                value=read(path); row=value["results"]["behave"]
                coverage=read(path.with_suffix(".coverage.json"))
                assert coverage["status"]=="passed"
                available[f"{mode}_{tag}"]=row
                lines.append('| '+f'{mode}/{tag}'+' | '+' | '.join(f'{row[k]:.4f}' for k in ("obj_trans_err_fusion","hoi_err_fusion","mpjpe","root_trans_err","jitter"))+' |')
    lines += ["","主评估严格采用原版min_obj_contact_frames=10留下的340个片段，387个原test文件全部保留供审计；各片段完整推理。online为补充，不当作论文offline结果。",
              "保留canonical预处理、当前HP物理实现和++专有损失，因此称论文配方条件下的++比较，不称原版逐位复刻。",
              "Jitter按现有与EgoIMU一致的代码数值报告；论文文字单位与代码缩放需以实现口径区分。"]
    (OUT/"summary.md").write_text('\n'.join(lines)+'\n')
    if available:
        from scripts.probe_vc_joint_20261019 import clean
        write(OUT/"comparison.json",{"updated_at":now(),"paper":protocol["paper_targets"],"results":clean(available),
            "offline_minus_paper":{name:{k:row[k]-protocol["paper_targets"][k] for k in ("obj_trans_err_fusion","hoi_err_fusion","mpjpe","root_trans_err","jitter")} for name,row in available.items() if name.startswith("offline")}})

def run():
    lock = (OUT / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(OUT / "protocol.json")
    assert read(OUT / "preflight.json")["status"] == "passed"
    state = read(OUT / "status.json")
    state.update(status="running", pid=os.getpid(), error=None)
    jobs = []
    for method, phase in STAGES:
        jobs.append((f"{method}_{phase}", [str(PYTHON), __file__, "worker", "--method", method, "--phase", phase],
                     OUT / method / phase / "completed.json"))
        if phase == "joint":
            for tag in ("best", "final"):
                for mode in ("offline", "online"):
                    command, expected = eval_command(method, mode, tag)
                    jobs.append((f"{method}_{mode}_{tag}", command, expected))
    try:
        for name, command, expected in jobs:
            if expected.exists():
                continue
            verify(protocol)
            while True:
                try:
                    gpu = common.check_gpu(6000)
                    break
                except RuntimeError as exc:
                    state.update(status="waiting_for_gpu", active_stage=name, error=str(exc), updated_at=now())
                    write(OUT / "status.json", state); report(state)
                    time.sleep(30)
            state.update(status="running", active_stage=name, gpu=gpu, updated_at=now())
            state["stages"][name] = {"status": "running", "started_at": now(), "command": command}
            with (OUT / f"{name}.log").open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONHASHSEED": "10"})
                state["child_pid"] = process.pid
                write(OUT / "status.json", state)
                while process.poll() is None:
                    report(state); time.sleep(20)
            if process.returncode or not expected.exists():
                raise RuntimeError(f"{name} failed ({process.returncode}); see {name}.log")
            state["stages"][name].update(status="complete", finished_at=now())
            write(OUT / "status.json", state)
        state.update(status="complete", active_stage=None, child_pid=None, updated_at=now())
    except Exception as exc:
        state.update(status="failed", error=str(exc), updated_at=now())
        state["stages"][state["active_stage"]].update(status="failed", error=str(exc))
        raise
    finally:
        write(OUT / "status.json", state); report(state)


def preflight():
    protocol = read(OUT / "protocol.json")
    verify(protocol)
    for method, phase in STAGES:
        common.check_gpu(6000)
        path = OUT / "preflight" / method / phase
        path.mkdir(parents=True, exist_ok=True)
        with (path / "smoke.log").open("w") as log:
            subprocess.run([str(PYTHON), __file__, "smoke", "--method", method, "--phase", phase],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert read(path / "smoke.json")["status"] == "passed"
        print(f"PREFLIGHT PASSED {method}/{phase}", flush=True)
    write(OUT / "preflight.json", {"status": "passed", "finished_at": now(), "stages": [f"{m}/{p}" for m,p in STAGES],
                                   "all_actual_batch_sizes_tested": True, "all_samplers_unique_indices": len(protocol["members"]["train"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "smoke", "preflight", "run"))
    parser.add_argument("--method", choices=("shared", "baseline", "proposed"))
    parser.add_argument("--phase", choices=("hp", "vc", "ot", "joint"))
    args = parser.parse_args()
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(hard, 65536)), hard))
    common.check_gpu(0 if args.mode == "prepare" else 6000)
    os.chdir(ROOT)
    if args.mode in ("worker", "smoke"):
        worker(args.method, args.phase, smoke=args.mode == "smoke")
    else:
        {"prepare": prepare, "preflight": preflight, "run": run}[args.mode]()


if __name__ == "__main__":
    main()




