#!/usr/bin/env python3
"""Call EgoIMU's native entry point, adding strict load checks and JSON receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EGO = ROOT.parent / "EgoIMU"
OUTPUT = ROOT / "outputs/analysis/egoimu_native_eval_20261017"
PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("debug", "test"), required=True)
    parser.add_argument("--weights", choices=("modules_best", "last_1199"), default="modules_best")
    parser.add_argument("--profiler", choices=("native", "disabled"), default="disabled")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"Use {PYTHON}")
    sys.dont_write_bytecode = True
    fd_before = resource.getrlimit(resource.RLIMIT_NOFILE)
    fd_target = 65536 if fd_before[1] == resource.RLIM_INFINITY else min(65536, fd_before[1])
    if fd_before[0] < fd_target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (fd_target, fd_before[1]))
    os.chdir(EGO)
    sys.path.insert(0, str(EGO))
    import numpy as np
    import torch
    import eval_IMUHOI as native

    if Path(native.__file__).resolve() != EGO / "eval_IMUHOI.py":
        raise RuntimeError("Imported the wrong evaluator")
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)

    folder = OUTPUT / f"{args.weights}_{args.split}_{args.profiler}"
    folder.mkdir(parents=True, exist_ok=True)
    sources = [EGO / name for name in ("eval_IMUHOI.py", "models/IMUHOI_stage_net.py",
               "models/do_train_IMUHOI.py", "dataloader/dataloader_IMUHOI.py", "configs/IMUHOI_train.yaml")]
    source_hashes = {str(p.relative_to(EGO)): sha256(p) for p in sources}
    metadata = {"status": "running", "seed": 10, "split": args.split, "weights": args.weights,
                "inference_mode": "offline_native_full_forward", "profiler": args.profiler,
                "native_evaluator": str(native.__file__), "python": sys.executable,
                "fd_limit_before": fd_before, "fd_limit_after": resource.getrlimit(resource.RLIMIT_NOFILE),
                "source_sha256": source_hashes, "add_eval_noise": False,
                "data_directory": str(EGO / "process/processed_split_data_OMOMO" / args.split)}
    (folder / "status.json").write_text(json.dumps(metadata, indent=2))
    original_load = native.load_model
    original_evaluate = native.evaluate_model
    original_print = print

    def checked_print(*values, **kwargs):
        message = " ".join(str(v) for v in values)
        if any(fragment in message for fragment in (
            "Model inference failed", "Failed to build model input", "SMPL forward failed", "skipping sample")):
            raise RuntimeError("Native evaluator skipped data: " + message)
        original_print(*values, **kwargs)

    native.print = checked_print

    def load_model(cfg, device):
        model = original_load(cfg, device)
        paths = {}
        for name in ("velocity_contact", "human_pose", "object_trans"):
            path = EGO / "outputs/IMUHOI/omomo/modules" / f"{name}_best.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            expected = payload["module_state_dict"]
            actual = getattr(model, f"{name}_module").state_dict()
            if set(actual) != set(expected) or any(not torch.equal(actual[k].cpu(), expected[k]) for k in expected):
                raise RuntimeError(f"Native loader failed to load exact {name} weights")
            paths[name] = {"path": str(path), "sha256": sha256(path), "epoch": payload["epoch"]}
        joint_path = EGO / "outputs/IMUHOI/omomo/stage_best_joint_training.pt"
        joint = torch.load(joint_path, map_location="cpu", weights_only=False)["model_state_dict"]
        actual = model.state_dict()
        metadata["default_modules_equal_joint_best"] = set(actual) == set(joint) and all(
            torch.equal(actual[k].cpu(), joint[k]) for k in joint
        )
        if args.weights == "last_1199":
            path = EGO / "outputs/IMUHOI/omomo/final_stage_joint_training_epoch_1199.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            paths = {"full_model": {"path": str(path), "sha256": sha256(path), "epoch": payload["epoch"]}}
        metadata["loaded_checkpoints"] = paths
        metadata["all_weights_strictly_verified"] = True
        model.eval()
        print("NATIVE WEIGHTS VERIFIED " + json.dumps(paths), flush=True)
        return model

    def evaluate(model, smpl_model, loader, cfg, device, **kwargs):
        n = len(loader.dataset)
        expected_count = 13 if args.split == "debug" else 1177
        if n != expected_count:
            raise RuntimeError(f"Unexpected native dataset filtering: {n} != {expected_count}")
        metadata["sequence_count"] = n
        metadata["sequences"] = [dict(x) for x in loader.dataset.sequence_info]
        calls = {"model": 0, "eval_body_model": 0}
        def model_hook(_module, _inputs, _outputs):
            calls["model"] += 1
        def smpl_hook(_module, _inputs, _outputs):
            calls["eval_body_model"] += 1
        handles = [model.register_forward_hook(model_hook), smpl_model.register_forward_hook(smpl_hook)]
        if args.profiler == "disabled":
            kwargs["stage_profiler"]["enable_flops"] = False
            kwargs["stage_profiler"]["use_cuda_profiler"] = False
        started = time.monotonic()
        try:
            metrics, profiling = original_evaluate(model, smpl_model, loader, cfg, device, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        if calls != {"model": n, "eval_body_model": 2 * n}:
            raise RuntimeError(f"Incomplete native evaluation: {calls}")
        for key in ("mpjpe", "root_trans_err", "obj_trans_err_fusion", "hoi_err_fusion"):
            if not math.isfinite(metrics[key]):
                raise RuntimeError(f"Invalid metric: {key}")
        if source_hashes != {str(p.relative_to(EGO)): sha256(p) for p in sources}:
            raise RuntimeError("EgoIMU source/config changed during evaluation")
        metadata.update(status="complete", elapsed_seconds=time.monotonic() - started,
                        evaluated_calls=calls, results=metrics, source_unchanged=True)
        (folder / "results.json").write_text(json.dumps(metadata, indent=2))
        (folder / "status.json").write_text(json.dumps(metadata, indent=2))
        print("NATIVE RESULT " + json.dumps(metrics), flush=True)
        return metrics, profiling

    native.load_model = load_model
    native.evaluate_model = evaluate
    command = [str(EGO / "eval_IMUHOI.py"), "--config", "configs/IMUHOI_train.yaml",
               "--dataset", "processed_split_data_OMOMO", "--test_data_dir", metadata["data_directory"],
               "--num_workers", "0", "--compare_3"]
    metadata["native_argv"] = command
    (folder / "command.json").write_text(json.dumps(command, indent=2))
    sys.argv = command
    try:
        native.main()
        if metadata["status"] != "complete":
            raise RuntimeError("Native main returned without evaluating")
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        (folder / "status.json").write_text(json.dumps(metadata, indent=2))
        raise


if __name__ == "__main__":
    main()
