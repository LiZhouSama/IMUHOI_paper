#!/usr/bin/env python3
"""Decompose velocity drift and separately swap VC velocity/contact outputs."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/analysis/egoimu_implementation_bridge_20261017"
EXP = ROOT / "outputs/IMUHOI++/egoimu_split_seed10_omomo_20261016"
EGO = ROOT.parent / "EgoIMU"


def main():
    import torch
    import torch.nn.functional as F
    import yaml
    from easydict import EasyDict
    from model.rnn.velocity_contact_IMUHOI import IMUHOIVelocityContactModule
    from model.rnn.object_trans_IMUHOI import IMUHOIObjectTransModule
    from model.rnn.human_pose import HumanPoseModule
    from dataset.dataset_IMUHOI import _build_object_imu
    from utils.rotation_conversions import rotation_6d_to_matrix
    cfg = EasyDict(yaml.safe_load((EXP / "recipe.yaml").read_text()))
    manifest = json.loads((EXP / "baseline/experiment_manifest.json").read_text())
    pipeline = manifest["pipelines"]["vc_imuhoi_ot_3branch"]
    device = torch.device("cuda:0")
    def load_state(path):
        payload = torch.load(path, map_location=device, weights_only=False)
        return payload.get("module_state_dict", payload.get("model_state_dict"))
    vc = IMUHOIVelocityContactModule(cfg).to(device).eval()
    vc.load_state_dict(load_state(manifest["stages"]["velocity_contact"]["checkpoint"]), strict=True)
    ots = {"old_ot": IMUHOIObjectTransModule(cfg).to(device).eval(),
           "new_ot": IMUHOIObjectTransModule(cfg).to(device).eval()}
    ots["old_ot"].load_state_dict(load_state(EGO / "outputs/IMUHOI/omomo/modules/object_trans_best.pt"), strict=True)
    ots["new_ot"].load_state_dict(load_state(Path(pipeline["stage_dir"]) / "final_object_trans.pt"), strict=True)
    # Call current root-fusion/integration methods on identical saved head outputs.
    hp = HumanPoseModule.__new__(HumanPoseModule)
    torch.nn.Module.__init__(hp)
    hp.fps, hp.prob_threshold = 30., (.5, .9)
    hp.gravity_velocity = torch.tensor([0., -.018, 0.], device=device)
    hp.prevent_floor_penetration, hp.floor_height, hp._foot_joint_indices = True, .03, (7, 8)
    entries = json.loads((OUT / "native_manifest.json").read_text())["entries"]
    rows, profiles = [], {name: [] for name in ("old", "new")}
    labels = ("old_contact_old_velocity", "new_contact_old_velocity", "old_contact_new_velocity", "new_contact_new_velocity")

    def integrate(vel, init):
        disp = vel.cumsum(0) / 30
        return init + torch.cat((disp.new_zeros(1, 3), disp[:-1]), 0)

    with torch.inference_mode():
        for i, entry in enumerate(entries):
            cache = torch.load(OUT / "native_cache" / entry["file"], weights_only=False)
            data = {k: v.to(device) for k, v in cache["data"].items()}
            old = {k: v.to(device) for k, v in cache["pred"].items()}
            new = vc(data)
            target = cache["target"]["obj_trans"][0].to(device)
            gtvel = cache["target"]["obj_vel"][0].to(device)
            row = {"file": entry["file"], "frames": entry["frames"], "velocity": {}, "ot_obj_cm": {}}
            for name, pred in (("old", old), ("new", new)):
                vel = pred["pred_obj_vel"][0]
                error = vel - gtvel
                bias = error.mean(0)
                smooth_error = F.avg_pool1d(error.T[None], 31, stride=1, padding=15, count_include_pad=False)[0].T
                centered = error - bias
                lag1 = (centered[1:] * centered[:-1]).sum() / (centered[1:].square().sum() * centered[:-1].square().sum()).sqrt().clamp_min(1e-12)
                pos = integrate(vel, target[0])
                debiased = integrate(vel - bias, target[0])
                row["velocity"][name] = {
                    "rmse_mps": error.square().mean().sqrt().item(), "bias_norm_mps": bias.norm().item(),
                    "one_second_low_frequency_error_mps": smooth_error.norm(dim=-1).mean().item(),
                    "centered_error_lag1_correlation": lag1.item(),
                    "obj_cm": (pos - target).norm(dim=-1).mean().item() * 100,
                    "oracle_debiased_obj_cm": (debiased - target).norm(dim=-1).mean().item() * 100,
                    "end_obj_cm": (pos[-1] - target[-1]).norm().item() * 100,
                }
                err_profile = (pos - target).norm(dim=-1) * 100
                profiles[name].append([x.mean().item() for x in torch.tensor_split(err_profile, 10)])

            kwargs = {"obj_trans_init": data["obj_trans_init"].repeat(4, 1), "obj_imu": data["obj_imu"].repeat(4, 1, 1),
                      "human_imu": data["human_imu"].repeat(4, 1, 1, 1), "contact_init": data["contact_init"].repeat(4, 1),
                      "has_object_mask": data["has_object"].repeat(4),
                      "obj_vel_input": torch.cat((old["pred_obj_vel"], old["pred_obj_vel"], new["pred_obj_vel"], new["pred_obj_vel"]), 0)}
            contacts = torch.cat((old["pred_hand_contact_prob"], new["pred_hand_contact_prob"], old["pred_hand_contact_prob"], new["pred_hand_contact_prob"]), 0)
            for name, ot in ots.items():
                result = ot(old["pred_hand_glb_pos"].repeat(4, 1, 1, 1), contacts, **kwargs)
                row["ot_obj_cm"][name] = {label: (result["pred_obj_trans"][j] - target).norm(dim=-1).mean().item() * 100
                                           for j, label in enumerate(labels)}

            # Explicit old formula; compare raw velocity fusion before floor correction.
            joints = cache["pred_joints"].to(device)[None] - old["root_trans_pred"][:, :, None]
            rotation = rotation_6d_to_matrix(data["human_imu"][:, :, 0, -6:])
            world = (rotation @ old["root_vel_local_pred"][..., None]).squeeze(-1)
            footdelta = torch.zeros_like(world)
            selected = old["contact_pred"].argmax(-1)
            for index, joint in enumerate((7, 8)):
                delta = torch.zeros_like(world)
                delta[:, 1:] = joints[:, :-1, joint] - joints[:, 1:, joint]
                footdelta = torch.where((selected == index)[..., None], delta, footdelta)
            weight = hp._prob_to_weight(old["contact_pred"].sigmoid().max(-1).values)[..., None]
            legacy_velocity = (1 - weight) * world * 3 + weight * (footdelta + hp.gravity_velocity)
            current_velocity = hp._fuse_velocities_batched(old["root_vel_local_pred"], old["contact_pred"], joints, rotation)
            row["same_heads_root_fusion_velocity_delta_mps"] = (current_velocity - legacy_velocity).norm(dim=-1).mean().item()
            adjustment = F.relu(-joints[:, :, 7:9, 1].amin(-1) - legacy_velocity[:, :, 1].cumsum(1)).cummax(1).values
            corrected = legacy_velocity.clone()
            corrected[:, :, 1] += adjustment - F.pad(adjustment[:, :-1], (1, 0))
            legacy_trans = corrected.cumsum(1) / 30 + data["trans_init"][:, None]
            row["legacy_root_formula_reconstruction_max_abs_m"] = (legacy_trans - old["root_trans_pred"]).abs().max().item()
            current_trans = hp._integrate_root_velocity(current_velocity) + data["trans_init"][:, None]
            current_trans, _ = hp._apply_ground_correction(current_trans, joints)
            trans_gt = cache["target"]["trans"].to(device)
            row["same_heads_current_root_formula_error_cm"] = (current_trans - trans_gt).norm(dim=-1).mean().item() * 100
            row["same_heads_legacy_root_formula_error_cm"] = (legacy_trans - trans_gt).norm(dim=-1).mean().item() * 100

            row["window_feature_probes"] = {}
            for window in (60, 120):
                if entry["frames"] <= window + 1:
                    continue
                start = (entry["frames"] - window) // 2
                end = start + window
                recomputed = _build_object_imu(data["obj_imu"][0, start:end, 3:9], target[start:end], 30)
                difference = (recomputed[:, :3] - data["obj_imu"][0, start:end, :3]).abs()
                init_delta = (target[start + 1] - target[start]) * 30 - gtvel[start]
                row["window_feature_probes"][str(window)] = {
                    "acc_mean_abs_difference": difference.mean().item(),
                    "acc_edge_mean_abs_difference": torch.cat((difference[:4], difference[-4:])).mean().item(),
                    "acc_interior_max_abs_difference": difference[4:-4].max().item(),
                    "initial_velocity_delta_norm_mps": init_delta.norm().item()}
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"DRIFT {i+1}/{len(entries)}", flush=True)
    summary = {"n": len(rows), "velocity": {}, "ot_contact_velocity_swaps_obj_cm": {},
               "root_formula_probe_is_original_heads_not_retrained_current_HP": True,
               "oracle_bias_removal_uses_test_GT_and_is_not_a_method": True}
    for name in ("old", "new"):
        summary["velocity"][name] = {k: sum(r["velocity"][name][k] for r in rows) / len(rows) for k in rows[0]["velocity"][name]}
    for name in ots:
        summary["ot_contact_velocity_swaps_obj_cm"][name] = {label: sum(r["ot_obj_cm"][name][label] for r in rows) / len(rows) for label in labels}
    for key in ("same_heads_root_fusion_velocity_delta_mps", "same_heads_current_root_formula_error_cm", "same_heads_legacy_root_formula_error_cm"):
        summary[key] = sum(r[key] for r in rows) / len(rows)
    summary["legacy_root_formula_reconstruction_max_abs_m"] = max(r["legacy_root_formula_reconstruction_max_abs_m"] for r in rows)
    summary["window_feature_probes"] = {}
    for window in ("60", "120"):
        values = [r["window_feature_probes"][window] for r in rows if window in r["window_feature_probes"]]
        summary["window_feature_probes"][window] = {"n": len(values), **{key: sum(v[key] for v in values) / len(values) for key in values[0]}}
    with (ROOT / "outputs/analysis/omomo_preprocessing_audit_20261016/per_sequence.csv").open() as stream:
        inventory = list(csv.DictReader(stream))
    old_train = [r for r in inventory if r["old_split"] == "train" and int(r["old_frames"]) >= 121
                 and float(r["old_object_move_fraction"]) * int(r["old_frames"]) >= 9.99999]
    summary["saved_split_actual_training_eligibility"] = {
        "old_120_frame_min_motion_10": len(old_train), "current_60_frame": 4701,
        "old_steps_vc": len(old_train) // 100, "old_steps_hp_ot": len(old_train) // 60,
        "old_steps_joint": len(old_train) // 50, "current_steps_all": 23}
    summary["obj_error_deciles_cm"] = {name: [sum(v[j] for v in profiles[name]) / len(rows) for j in range(10)] for name in profiles}
    (OUT / "velocity_drift_results.json").write_text(json.dumps(summary, indent=2))
    (OUT / "velocity_drift_sequences.json").write_text(json.dumps(rows))
    print(json.dumps(summary, indent=2), flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, label in (("old", "EgoIMU weights"), ("new", "Current VC weights")):
        axes[0].plot(range(1, 11), summary["obj_error_deciles_cm"][name], marker="o", label=label)
    axes[0].set(xlabel="Sequence time decile", ylabel="Object position error (cm)")
    axes[0].legend(frameon=False)
    axes[1].bar([0, 1], [summary["velocity"][name]["rmse_mps"] for name in ("old", "new")])
    axes[1].set(xticks=[0, 1], xticklabels=["EgoIMU", "Current VC"], ylabel="Velocity RMSE (m/s)")
    fig.tight_layout()
    fig.savefig(OUT / "velocity_vs_drift.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    if Path(sys.executable).resolve() != Path("/home/l/anaconda3/envs/SAGE/bin/python").resolve():
        raise RuntimeError("Use WSL SAGE")
    main()
