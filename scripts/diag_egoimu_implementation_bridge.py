#!/usr/bin/env python3
"""Isolate input, forward implementation, and learned-weight differences."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
EGO = ROOT.parent / "EgoIMU"
EXPERIMENT = ROOT / "outputs/IMUHOI++/egoimu_split_seed10_omomo_20261016"
OUT = ROOT / "outputs/analysis/egoimu_implementation_bridge_20261017"


def cpu_copy(values):
    return {key: value.detach().cpu() for key, value in values.items() if hasattr(value, "detach")}


def export_native():
    os.chdir(EGO)
    sys.path.insert(0, str(EGO))
    import torch
    import eval_IMUHOI as native
    from torch.utils.data import DataLoader
    cfg = native.load_config("configs/IMUHOI_train.yaml")
    device = torch.device("cuda:0")
    model = native.load_model(cfg, device)
    for name in ("human_pose", "velocity_contact", "object_trans"):
        path = EGO / f"outputs/IMUHOI/omomo/modules/{name}_best.pt"
        getattr(model, name + "_module").load_state_dict(
            torch.load(path, map_location=device, weights_only=False)["module_state_dict"], strict=True)
    model.eval()
    smpl = native.load_smpl_model(cfg.body_model_path, device)
    dataset = native.IMUDataset(str(EGO / "process/processed_split_data_OMOMO/test"), full_sequence=True)
    if len(dataset) != 1177:
        raise RuntimeError("Native test incomplete")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    pending = {}
    entries = []
    (OUT / "native_cache").mkdir(parents=True, exist_ok=True)
    pred_keys = {"v_pred", "p_pred", "contact_pred", "root_vel_local_pred", "root_vel_pred", "root_trans_pred",
                 "pred_hand_glb_pos", "pred_obj_vel", "pred_hand_glb_vel", "pred_hand_contact_prob",
                 "pred_obj_trans", "gating_weights", "pred_lhand_obj_trans", "pred_rhand_obj_trans"}
    target_keys = {"obj_trans", "obj_vel", "trans", "position_global", "lhand_contact", "rhand_contact", "obj_contact"}

    class CaptureLoader:
        def __len__(self):
            return len(loader)
        def __iter__(self):
            for i, batch in enumerate(loader):
                pending.clear()
                pending.update(index=i, target={k: v for k, v in batch.items() if k in target_keys}, smpl_calls=0)
                yield batch

    def capture_model(_model, inputs, output):
        pending["data"] = inputs[0]
        pending["pred"] = {k: v for k, v in output.items() if k in pred_keys}

    def capture_body(_model, _inputs, output):
        pending["smpl_calls"] += 1
        if pending["smpl_calls"] == 1:
            pending["gt_joints"] = output.Jtr[:, :24].detach().cpu()
        elif pending["smpl_calls"] == 2:
            i = pending["index"]
            name = Path(dataset.sequence_info[i]["file_path"]).name
            payload = {"data": cpu_copy(pending["data"]), "pred": cpu_copy(pending["pred"]),
                       "target": cpu_copy(pending["target"]), "gt_joints": pending["gt_joints"],
                       "pred_joints": output.Jtr[:, :24].detach().cpu()}
            torch.save(payload, OUT / "native_cache" / name)
            entries.append({"file": name, "frames": int(output.Jtr.shape[0])})
        else:
            raise RuntimeError("Unexpected evaluator BodyModel call")

    handles = [model.register_forward_hook(capture_model), smpl.register_forward_hook(capture_body)]
    try:
        metrics, _ = native.evaluate_model(model, smpl, CaptureLoader(), cfg, device,
                                           compare_three=False, stage_profiler=native._init_stage_profiler(False))
    finally:
        for handle in handles:
            handle.remove()
    reference = json.loads((ROOT / "outputs/analysis/egoimu_native_eval_20261017/modules_best_test_disabled/results.json").read_text())
    for key in ("mpjpe", "root_trans_err", "obj_trans_err_fusion", "hoi_err_fusion"):
        if abs(metrics[key] - reference["results"][key]) > 1e-5:
            raise RuntimeError(f"Native export failed metric recheck: {key}")
    if len(entries) != 1177:
        raise RuntimeError(f"Incomplete cache: {len(entries)}")
    (OUT / "native_manifest.json").write_text(json.dumps({"metrics": metrics, "entries": entries}, indent=2))
    print("NATIVE EXPORT VERIFIED " + json.dumps(metrics), flush=True)


def compare_current():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import torch
    import yaml
    from easydict import EasyDict
    from torch.utils.data._utils.collate import default_collate
    from model.rnn.velocity_contact_IMUHOI import IMUHOIVelocityContactModule
    from model.rnn.object_trans_IMUHOI import IMUHOIObjectTransModule
    from dataset.dataset_IMUHOI import IMUDataset
    from utils.utils import build_model_input_dict
    from utils.hoi_metrics import compute_hoi_error_egoimu
    cfg = EasyDict(yaml.safe_load((EXPERIMENT / "recipe.yaml").read_text()))
    manifest = json.loads((EXPERIMENT / "baseline/experiment_manifest.json").read_text())
    pipeline = manifest["pipelines"]["vc_imuhoi_ot_3branch"]
    device = torch.device("cuda:0")

    def state(path):
        payload = torch.load(path, map_location=device, weights_only=False)
        for key in ("module_state_dict", "model_state_dict", "state_dict"):
            if key in payload:
                return payload[key]
        raise ValueError(f"Unknown checkpoint {path}")

    vc_old, vc_new = (IMUHOIVelocityContactModule(cfg).to(device).eval() for _ in range(2))
    ot_old, ot_new = (IMUHOIObjectTransModule(cfg).to(device).eval() for _ in range(2))
    vc_old.load_state_dict(state(EGO / "outputs/IMUHOI/omomo/modules/velocity_contact_best.pt"), strict=True)
    vc_new.load_state_dict(state(manifest["stages"]["velocity_contact"]["checkpoint"]), strict=True)
    ot_old.load_state_dict(state(EGO / "outputs/IMUHOI/omomo/modules/object_trans_best.pt"), strict=True)
    stage = Path(pipeline["stage_dir"])
    ot_new.load_state_dict(state(stage / "final_object_trans.pt"), strict=True)
    data_new = IMUDataset(str(EXPERIMENT / "data/test"), full_sequence=True, simulate_imu_noise=False,
                         require_canonical_skeleton=True)
    mapping = {Path(x["file_path"]).name: i for i, x in enumerate(data_new.sequence_info)}
    exported = json.loads((OUT / "native_manifest.json").read_text())
    maxima = {}
    rows = []

    def maxdiff(name, a, b):
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch {name}: {a.shape}, {b.shape}")
        value = (a - b).abs().max().item()
        maxima[name] = max(maxima.get(name, 0), value)
        return value

    def score(pos, target, cache):
        return {"obj_cm": (pos - target).norm(dim=-1).mean().item() * 100,
                "hoi_cm": float(compute_hoi_error_egoimu(pos, target, cache["pred_joints"].to(device),
                    cache["gt_joints"].to(device), cache["target"]["lhand_contact"][0].to(device),
                    cache["target"]["rhand_contact"][0].to(device)))}

    with torch.inference_mode():
        for idx, entry in enumerate(exported["entries"]):
            cache = torch.load(OUT / "native_cache" / entry["file"], map_location="cpu", weights_only=False)
            data = {k: v.to(device) for k, v in cache["data"].items()}
            old_pred = {k: v.to(device) for k, v in cache["pred"].items()}
            a, b = vc_old(data), vc_new(data)
            for key in ("pred_obj_vel", "pred_hand_glb_vel", "pred_hand_contact_prob"):
                maxdiff("vc_original_weights_" + key, a[key], old_pred[key])
            batch_new = default_collate([data_new[mapping[entry["file"]]]])
            new_data = build_model_input_dict(batch_new, cfg, device, add_noise=False)
            obj_imu_diff = maxdiff("input_object_imu", data["obj_imu"], new_data["obj_imu"])
            maxdiff("input_object_initial_velocity", data["obj_vel_init"], new_data["obj_vel_init"])
            maxdiff("target_object_velocity", cache["target"]["obj_vel"].to(device), batch_new["obj_vel"].to(device))
            maxdiff("input_human_imu", data["human_imu"], new_data["human_imu"])
            a_new_input, b_new_input = vc_old(new_data), vc_new(new_data)
            maxdiff("vc_old_weights_object_velocity_data_change", a["pred_obj_vel"], a_new_input["pred_obj_vel"])
            maxdiff("vc_new_weights_object_velocity_data_change", b["pred_obj_vel"], b_new_input["pred_obj_vel"])
            target = cache["target"]["obj_trans"][0].to(device)
            row = {"file": entry["file"], "frames": entry["frames"], "object_imu_max_difference": obj_imu_diff,
                   "scores": {}, "velocity_rmse_mps": {}}
            for name, output in (("old_weights_old_data", a), ("new_weights_old_data", b),
                                 ("old_weights_new_data", a_new_input), ("new_weights_new_data", b_new_input)):
                vel = output["pred_obj_vel"][0]
                disp = vel.cumsum(0) / 30
                disp = torch.cat((disp.new_zeros(1, 3), disp[:-1]), dim=0)
                pos = target[0] + disp
                row["scores"]["integral_" + name] = score(pos, target, cache)
                row["velocity_rmse_mps"][name] = (vel - cache["target"]["obj_vel"][0].to(device)).square().mean().sqrt().item()
            kwargs = {"obj_trans_init": data["obj_trans_init"].repeat(2, 1),
                      "obj_imu": data["obj_imu"].repeat(2, 1, 1), "human_imu": data["human_imu"].repeat(2, 1, 1, 1),
                      "contact_init": data["contact_init"].repeat(2, 1), "has_object_mask": data["has_object"].repeat(2),
                      "obj_vel_input": torch.cat((a["pred_obj_vel"], b["pred_obj_vel"]), 0)}
            for name, ot in (("old_ot", ot_old), ("new_ot", ot_new)):
                result = ot(old_pred["pred_hand_glb_pos"].repeat(2, 1, 1, 1),
                            torch.cat((a["pred_hand_contact_prob"], b["pred_hand_contact_prob"]), 0), **kwargs)
                for i, label in enumerate(("old_vc", "new_vc")):
                    row["scores"][name + "_" + label] = score(result["pred_obj_trans"][i], target, cache)
                    row[name + "_" + label + "_imu_weight"] = result["gating_weights"][i, :, 2].mean().item()
                if name == "old_ot":
                    for key in ("pred_obj_trans", "gating_weights", "pred_lhand_obj_trans", "pred_rhand_obj_trans"):
                        maxdiff("ot_original_weights_" + key, result[key][:1], old_pred[key])
            rows.append(row)
            if (idx + 1) % 100 == 0:
                print(f"BRIDGE {idx+1}/{len(exported['entries'])}", flush=True)
    summary = {"sequence_count": len(rows), "max_absolute_differences": maxima,
               "scope": "All OT crosses hold original HP, original data and original metric GT fixed; diagnostics, not retrained systems.",
               "mean_scores": {}, "mean_velocity_rmse_mps": {}, "mean_imu_weights": {}}
    for name in rows[0]["scores"]:
        summary["mean_scores"][name] = {key: sum(r["scores"][name][key] for r in rows) / len(rows)
                                         for key in ("obj_cm", "hoi_cm")}
    for name in rows[0]["velocity_rmse_mps"]:
        summary["mean_velocity_rmse_mps"][name] = sum(r["velocity_rmse_mps"][name] for r in rows) / len(rows)
    for name in ("old_ot_old_vc", "old_ot_new_vc", "new_ot_old_vc", "new_ot_new_vc"):
        summary["mean_imu_weights"][name] = sum(r[name + "_imu_weight"] for r in rows) / len(rows)
    if len(rows) != 1177:
        raise RuntimeError("Incomplete bridge")
    (OUT / "bridge_results.json").write_text(json.dumps(summary, indent=2))
    (OUT / "bridge_sequences.json").write_text(json.dumps(rows))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export_native", "compare_current"))
    args = parser.parse_args()
    if Path(sys.executable).resolve() != Path("/home/l/anaconda3/envs/SAGE/bin/python").resolve():
        raise RuntimeError("Use WSL native SAGE Python")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(hard, 65536)), hard))
    OUT.mkdir(parents=True, exist_ok=True)
    export_native() if args.mode == "export_native" else compare_current()


if __name__ == "__main__":
    main()
