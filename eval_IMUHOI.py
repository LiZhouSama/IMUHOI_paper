"""
IMUHOI评估脚本
统一版本：通过 --no_trans 参数控制?否使用noTrans模式
"""
import os
import sys
import argparse
import copy
import json
import random
import time
import types
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from tqdm import tqdm

# 添加项目根目录到??
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytorch3d.transforms as t3d
except Exception:
    from utils import rotation_conversions as _rot

    pytorch3d_mod = types.ModuleType("pytorch3d")
    t3d = types.ModuleType("pytorch3d.transforms")
    t3d.rotation_6d_to_matrix = _rot.rotation_6d_to_matrix
    t3d.matrix_to_rotation_6d = _rot.matrix_to_rotation_6d
    t3d.matrix_to_axis_angle = _rot.matrix_to_axis_angle
    pytorch3d_mod.transforms = t3d
    sys.modules["pytorch3d"] = pytorch3d_mod
    sys.modules["pytorch3d.transforms"] = t3d

from dataset.dataset_IMUHOI import IMUDataset
from model import IMUHOIModel, load_model
from model.rnn.velocity_contact import (
    VELOCITY_CONTACT_ARCH_CHOICES,
    normalize_velocity_contact_arch,
)
from model.rnn.object_trans_IMUHOI import is_imuhoi_object_trans_arch
from utils.dataset_config import (
    get_dataset_configs,
    get_full_sequence_test_path,
    resolve_dataset_path,
)
from utils.utils import (
    load_config,
    load_smpl_model,
    build_model_input_dict,
)
from utils.human_pose import (
    append_virtual_palm_joints,
    build_smpl24_joint_positions,
    select_hand_anchor_positions,
)
from utils.hoi_metrics import compute_hoi_error_at_gt_contact
from configs import (
    FRAME_RATE,
    _IGNORED_INDICES,
    _REDUCED_POSE_NAMES,
    _SENSOR_ROT_INDICES,
)


OBJECT_TRANS_ARCH_CHOICES = (
    "current",
    "legacy",
    "legacy_mesh",
    "mergeHeads_obs",
    "imuhoi",
    "imuhoi_4heads",
)


def _normalize_object_trans_arch(value):
    normalized = str(value).strip().lower()
    if normalized == "mergeheads_obs":
        return "mergeHeads_obs"
    return normalized

PROJECT_ROOT = Path(__file__).resolve().parent


def get_default_dataset_config(no_trans: bool = False, config=None):
    """Load dataset evaluation settings from the YAML ``datasets`` section."""
    del no_trans  # Kept for compatibility with legacy diagnostic callers.
    if config is None:
        config = load_config(str(PROJECT_ROOT / "configs" / "IMUHOI_train_rnn.yaml"))
    return get_dataset_configs(config)


def _build_dataset_runs(args: argparse.Namespace, default_config: dict):
    runs = []
    if args.dataset:
        if args.dataset not in default_config:
            valid = ", ".join(sorted(default_config))
            raise ValueError(f"Unknown dataset '{args.dataset}'. Valid datasets: {valid}")
        runs.append((args.dataset, copy.deepcopy(default_config[args.dataset])))
        return runs
    for name, cfg in default_config.items():
        runs.append((name, copy.deepcopy(cfg)))
    return runs


def _set_eval_seed(seed: Optional[int]) -> Optional[int]:
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _apply_module_overrides(config: edict, modules_override: Dict[str, Path]) -> None:
    if not modules_override:
        return
    # New config style: top-level pretrained_modules (used by DiT pipeline / vis script).
    current_top = {}
    if hasattr(config, "pretrained_modules") and config.pretrained_modules:
        current_top = dict(config.pretrained_modules)
    for module_name, module_path in modules_override.items():
        current_top[module_name] = str(module_path)
    config.pretrained_modules = current_top

    # Backward-compatible style: staged_training.modular_training.pretrained_modules.
    staged_cfg = getattr(config, "staged_training", None)
    if not staged_cfg:
        config.staged_training = {"modular_training": {"enabled": True, "pretrained_modules": {}}}
        staged_cfg = config.staged_training

    if isinstance(staged_cfg, dict):
        modular_cfg = staged_cfg.get("modular_training", {})
    else:
        modular_cfg = getattr(staged_cfg, "modular_training", {})

    if not modular_cfg:
        modular_cfg = {"enabled": True, "pretrained_modules": {}}

    pretrained_modules = dict(modular_cfg.get("pretrained_modules", {}))
    print("Using dataset-specific pretrained modules:")
    for module_name, module_path in modules_override.items():
        pretrained_modules[module_name] = str(module_path)
        print(f"  - {module_name}: {module_path}")
    modular_cfg["pretrained_modules"] = pretrained_modules
    modular_cfg["enabled"] = True

    if isinstance(staged_cfg, dict):
        staged_cfg["modular_training"] = modular_cfg
    else:
        staged_cfg.modular_training = modular_cfg
    config.staged_training = staged_cfg


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _get_human_pose_module(model):
    core = _unwrap_model(model)
    module = getattr(core, "human_pose_module", None)
    if module is not None:
        return module
    if core.__class__.__name__.lower() == "humanposemodule":
        return core
    wrapped = getattr(core, "module", None)
    if wrapped is not None:
        return getattr(wrapped, "human_pose_module", None)
    return None


def _infer_model_kind(model):
    core = _unwrap_model(model)
    name = core.__class__.__name__.lower()
    if "imuhoimixmodule" in name:
        return "mix"
    if "imuhoimodel" in name:
        return "pipeline"
    if "interactionmodule" in name:
        return "interaction"
    if "humanposemodule" in name:
        return "human_pose"
    if hasattr(core, "human_pose_module") and hasattr(core, "interaction_module"):
        return "pipeline"
    if hasattr(core, "mesh_encoder") and hasattr(core, "object_codebook"):
        return "interaction"
    return "unknown"


def _filter_pipeline_module_paths(module_paths):
    if not isinstance(module_paths, dict):
        return None
    allowed = {"human_pose", "interaction", "velocity_contact", "object_trans"}
    out = {}
    for key, value in module_paths.items():
        if key in allowed and value:
            out[key] = value
    return out if out else None


def _filter_human_pose_module_paths(module_paths):
    if not isinstance(module_paths, dict):
        return None
    human_pose_path = module_paths.get("human_pose")
    return {"human_pose": human_pose_path} if human_pose_path else None


def _apply_cli_module_overrides(module_paths, args):
    out = dict(module_paths) if isinstance(module_paths, dict) else {}
    if getattr(args, "hp_ckpt", None):
        out["human_pose"] = args.hp_ckpt
    if getattr(args, "interaction_ckpt", None):
        out["interaction"] = args.interaction_ckpt
    if getattr(args, "vc_ckpt", None):
        out["velocity_contact"] = args.vc_ckpt
    if getattr(args, "ot_ckpt", None):
        out["object_trans"] = args.ot_ckpt
    return out if out else None


def _normalize_module_paths(module_paths):
    if not isinstance(module_paths, dict):
        return None
    out = {key: value for key, value in module_paths.items() if value}
    return out if out else None


def _object_contact_capabilities(module_paths):
    modules = _normalize_module_paths(module_paths) or {}
    has_interaction = bool(modules.get("interaction"))
    has_velocity_contact = has_interaction or bool(modules.get("velocity_contact"))
    has_object_trans = has_interaction or bool(modules.get("object_trans"))
    return has_velocity_contact, has_object_trans


def _global_to_local_rotmat(global_rotmats, parents):
    if global_rotmats.dim() != 4 or global_rotmats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [T, J, 3, 3], got {tuple(global_rotmats.shape)}")

    seq_len, num_joints = global_rotmats.shape[:2]
    local_rotmats = torch.zeros_like(global_rotmats)
    local_rotmats[:, 0] = global_rotmats[:, 0]

    for i in range(1, num_joints):
        parent_idx = int(parents[i]) if i < len(parents) else 0
        parent_idx = max(0, min(parent_idx, num_joints - 1))
        parent_rot = global_rotmats[:, parent_idx]
        local_rotmats[:, i] = torch.matmul(parent_rot.transpose(-1, -2), global_rotmats[:, i])

    valid_ignored = [idx for idx in _IGNORED_INDICES if idx < num_joints]
    if valid_ignored:
        eye = torch.eye(3, device=global_rotmats.device, dtype=global_rotmats.dtype).view(1, 1, 3, 3)
        local_rotmats[:, valid_ignored] = eye.expand(seq_len, len(valid_ignored), 3, 3)

    return local_rotmats


def _to_smpl24_joint_positions(joints: torch.Tensor) -> torch.Tensor:
    """Return body-22 plus the two virtual palm anchors for evaluation."""
    if not isinstance(joints, torch.Tensor):
        raise TypeError("joints must be a torch.Tensor")
    joint_count = int(joints.shape[-2])
    if joint_count in {52, 55}:
        return build_smpl24_joint_positions(joints)
    if joint_count == 22:
        return append_virtual_palm_joints(joints)
    if joint_count >= 24:
        return joints[..., :24, :]
    raise ValueError(f"Expected 22, 24, 52, or 55 joints, got {joint_count}")


def _build_published_object_trans(
    raw_obj_trans: Optional[torch.Tensor],
    stage1_hand_pos: Optional[torch.Tensor],
    refined_hand_pos: Optional[torch.Tensor],
    left_fk_pos: Optional[torch.Tensor],
    right_fk_pos: Optional[torch.Tensor],
    imu_pos: Optional[torch.Tensor],
    static_pos: Optional[torch.Tensor],
    gating_weights: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Re-run fusion with refined hands while keeping the trained OT state raw."""
    required = (
        raw_obj_trans,
        stage1_hand_pos,
        refined_hand_pos,
        left_fk_pos,
        right_fk_pos,
        imu_pos,
        static_pos,
        gating_weights,
    )
    if any(not isinstance(value, torch.Tensor) for value in required):
        return raw_obj_trans
    if stage1_hand_pos.dim() != 3 or stage1_hand_pos.shape[-2:] != (2, 3):
        return raw_obj_trans
    if refined_hand_pos.shape != stage1_hand_pos.shape:
        return raw_obj_trans
    if left_fk_pos.shape != raw_obj_trans.shape or right_fk_pos.shape != raw_obj_trans.shape:
        return raw_obj_trans
    if imu_pos.shape != raw_obj_trans.shape or static_pos.shape != raw_obj_trans.shape:
        return raw_obj_trans
    if gating_weights.dim() != 2 or gating_weights.shape[0] != raw_obj_trans.shape[0]:
        return raw_obj_trans
    if gating_weights.shape[-1] not in {3, 4}:
        return raw_obj_trans
    if torch.allclose(stage1_hand_pos, refined_hand_pos):
        return raw_obj_trans

    corrected_left_fk = refined_hand_pos[:, 0] + (left_fk_pos - stage1_hand_pos[:, 0])
    corrected_right_fk = refined_hand_pos[:, 1] + (right_fk_pos - stage1_hand_pos[:, 1])
    previous = static_pos[0]
    published = []
    for frame_idx in range(raw_obj_trans.shape[0]):
        imu_delta = imu_pos[frame_idx] - static_pos[frame_idx]
        imu_candidate = previous + imu_delta
        if gating_weights.shape[-1] == 4:
            current = (
                gating_weights[frame_idx, 0] * corrected_left_fk[frame_idx]
                + gating_weights[frame_idx, 1] * corrected_right_fk[frame_idx]
                + gating_weights[frame_idx, 2] * imu_candidate
                + gating_weights[frame_idx, 3] * previous
            )
        else:
            current = (
                gating_weights[frame_idx, 0] * corrected_left_fk[frame_idx]
                + gating_weights[frame_idx, 1] * corrected_right_fk[frame_idx]
                + gating_weights[frame_idx, 2] * imu_candidate
            )
        published.append(current)
        previous = current
    return torch.stack(published, dim=0)


def _build_published_object_candidates(
    raw_obj_trans: Optional[torch.Tensor],
    stage1_hand_pos: Optional[torch.Tensor],
    refined_hand_pos: Optional[torch.Tensor],
    left_fk_pos: Optional[torch.Tensor],
    right_fk_pos: Optional[torch.Tensor],
    imu_pos: Optional[torch.Tensor],
    static_pos: Optional[torch.Tensor],
    gating_weights: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return the four candidate trajectories used by published fusion."""
    required = (
        raw_obj_trans,
        stage1_hand_pos,
        refined_hand_pos,
        left_fk_pos,
        right_fk_pos,
        imu_pos,
        static_pos,
        gating_weights,
    )
    if any(not isinstance(value, torch.Tensor) for value in required):
        return {}
    if stage1_hand_pos.dim() != 3 or stage1_hand_pos.shape[-2:] != (2, 3):
        return {}
    if refined_hand_pos.shape != stage1_hand_pos.shape:
        return {}
    if left_fk_pos.shape != raw_obj_trans.shape or right_fk_pos.shape != raw_obj_trans.shape:
        return {}
    if imu_pos.shape != raw_obj_trans.shape or static_pos.shape != raw_obj_trans.shape:
        return {}
    if gating_weights.dim() != 2 or gating_weights.shape[0] != raw_obj_trans.shape[0]:
        return {}
    if gating_weights.shape[-1] not in {3, 4}:
        return {}

    corrected_left = refined_hand_pos[:, 0] + (left_fk_pos - stage1_hand_pos[:, 0])
    corrected_right = refined_hand_pos[:, 1] + (right_fk_pos - stage1_hand_pos[:, 1])
    previous = static_pos[0]
    replayed_imu = []
    replayed_static = []
    for frame_idx in range(raw_obj_trans.shape[0]):
        imu_candidate = previous + (imu_pos[frame_idx] - static_pos[frame_idx])
        replayed_imu.append(imu_candidate)
        replayed_static.append(previous)
        if gating_weights.shape[-1] == 4:
            previous = (
                gating_weights[frame_idx, 0] * corrected_left[frame_idx]
                + gating_weights[frame_idx, 1] * corrected_right[frame_idx]
                + gating_weights[frame_idx, 2] * imu_candidate
                + gating_weights[frame_idx, 3] * previous
            )
        else:
            previous = (
                gating_weights[frame_idx, 0] * corrected_left[frame_idx]
                + gating_weights[frame_idx, 1] * corrected_right[frame_idx]
                + gating_weights[frame_idx, 2] * imu_candidate
            )
    candidates = {
        "left_fk": corrected_left,
        "right_fk": corrected_right,
        "imu": torch.stack(replayed_imu, dim=0),
    }
    if gating_weights.shape[-1] == 4:
        candidates["static"] = torch.stack(replayed_static, dim=0)
    return candidates


def evaluate_model(
    model: IMUHOIModel,
    smpl_model,
    data_loader: DataLoader,
    config: edict,
    device: torch.device,
    no_trans: bool = False,
    evaluate_objects: bool = True,
    evaluate_contacts: bool = True,
    run_object_branch: Optional[bool] = None,
    compare_three: bool = True,
    interaction_human_source: str = "pred",
    ot_refine: bool = False,
    hand_contact_threshold: float = 0.5,
    object_contact_threshold: float = 0.5,
    inference_mode: str = "online",
    online_window: Optional[int] = None,
    online_lookahead: int = 5,
    sequence_records: Optional[list] = None,
):
    """评估模型"""
    metrics = {
        "mpjpe": [],
        "mpjre_angle": [],
        "jitter": [],
        "obj_jitter": [],
    }

    # 非noTrans模式添加root_trans_err
    if not no_trans:
        metrics["root_trans_err"] = []
        metrics["stage1_root_trans_err"] = []
    else:
        metrics["hand_vel_err_lhand"] = []
        metrics["hand_vel_err_rhand"] = []
        metrics["hand_vel_err_avg"] = []

    # 物体相关指标
    metrics.update({
        "obj_trans_err_fusion_raw": [],
        "obj_trans_err_fusion_refine": [],
        "obj_trans_err_fusion": [],
        "obj_trans_err_fk": [],
        "obj_trans_err_imu": [],
        "hoi_err_fusion": [],
        "hoi_err_fk": [],
        "hoi_err_imu": [],
        "contact_f1_lhand": [],
        "contact_f1_rhand": [],
        "contact_precision_lhand": [],
        "contact_precision_rhand": [],
        "contact_recall_lhand": [],
        "contact_recall_rhand": [],
        "contact_f05_lhand": [],
        "contact_f05_rhand": [],
        "contact_hard_neg_fp_rate_any": [],
        "contact_hard_neg_fp_rate_lhand": [],
        "contact_hard_neg_fp_rate_rhand": [],
        "contact_f1_obj": [],
    })

    num_eval_joints = 22

    progress_bar = tqdm(
        data_loader,
        total=len(data_loader),
        desc="Evaluating",
        unit="batch",
        dynamic_ncols=True,
    )

    with torch.inference_mode():
        for batch_idx, batch in enumerate(progress_bar):
            # 移动到???
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                else:
                    batch_device[key] = value

            try:
                data_dict = build_model_input_dict(batch_device, config, device, add_noise=False)
            except Exception as exc:
                print(f"Failed to build model input (batch {batch_idx}): {exc}")
                continue

            model_kind = _infer_model_kind(model)
            source = str(interaction_human_source).lower()
            if source not in {"pred", "gt"}:
                source = "pred"
            interaction_use_human_pred = source != "gt"
            want_object_predictions = bool(evaluate_objects if run_object_branch is None else run_object_branch)
            compute_fk = bool(compare_three and want_object_predictions)

            def _run_model_inference(use_object_data, compute_fk_flag):
                def _call_forward():
                    return model(
                        data_dict,
                        use_object_data=use_object_data,
                        compute_fk=compute_fk_flag,
                        refine_human=ot_refine,
                    )

                def _call_inference(*args, **kwargs):
                    if "inference_mode" not in kwargs:
                        kwargs["inference_mode"] = inference_mode
                    if "online_window" not in kwargs:
                        kwargs["online_window"] = online_window
                    if "online_lookahead" not in kwargs:
                        kwargs["online_lookahead"] = online_lookahead
                    try:
                        return model.inference(*args, **kwargs)
                    except TypeError:
                        kwargs.pop("inference_mode", None)
                        kwargs.pop("online_window", None)
                        kwargs.pop("online_lookahead", None)
                        return model.inference(*args, **kwargs)

                if model_kind == "mix":
                    return _call_inference(data_dict, gt_targets=batch_device)
                if model_kind == "pipeline":
                    if not hasattr(model, "inference"):
                        return _call_forward()
                    return _call_inference(
                        data_dict,
                        gt_targets=batch_device,
                        use_object_data=use_object_data,
                        compute_fk=compute_fk_flag,
                        interaction_use_human_pred=interaction_use_human_pred,
                        refine_human=ot_refine,
                    )
                if model_kind == "interaction":
                    if interaction_use_human_pred:
                        print("Warning: interaction-only model has no human branch; fallback to GT human context.")
                    return _call_inference(data_dict, hp_out=None, gt_targets=batch_device)
                if model_kind == "human_pose":
                    if hasattr(model, "inference"):
                        return _call_inference(data_dict, gt_targets=batch_device)
                    return model(data_dict)
                # Unknown model type: try broad signature, then simplified.
                if not hasattr(model, "inference"):
                    return _call_forward()
                try:
                    return _call_inference(
                        data_dict,
                        gt_targets=batch_device,
                        use_object_data=use_object_data,
                        compute_fk=compute_fk_flag,
                        interaction_use_human_pred=interaction_use_human_pred,
                        refine_human=ot_refine,
                    )
                except TypeError:
                    return _call_inference(data_dict, gt_targets=batch_device)

            try:
                pred_dict = _run_model_inference(
                    use_object_data=want_object_predictions,
                    compute_fk_flag=compute_fk,
                )
            except Exception as exc:
                if want_object_predictions and model_kind == "pipeline":
                    try:
                        print(
                            f"Model inference with object branch failed on batch {batch_idx}, "
                            f"fallback to human-only: {exc}"
                        )
                        pred_dict = _run_model_inference(use_object_data=False, compute_fk_flag=False)
                    except Exception as inner_exc:
                        print(f"Model inference failed on batch {batch_idx}: {inner_exc}")
                        continue
                else:
                    print(f"Model inference failed on batch {batch_idx}: {exc}")
                    continue

            human_imu = data_dict["human_imu"]
            batch_size, seq_len = human_imu.shape[:2]

            for sample_idx in range(batch_size):
                human_imu_seq = human_imu[sample_idx]
                pred_length_source = pred_dict.get("p_pred")
                if not isinstance(pred_length_source, torch.Tensor):
                    pred_length_source = pred_dict.get("root_trans_pred")
                T = (
                    min(human_imu_seq.shape[0], int(pred_length_source.shape[1]))
                    if isinstance(pred_length_source, torch.Tensor) and pred_length_source.dim() >= 2
                    else human_imu_seq.shape[0]
                )
                if T <= 0:
                    continue
                human_imu_seq = human_imu_seq[:T]

                pose_batch = batch_device.get("pose")
                trans_batch = batch_device.get("trans")
                if pose_batch is None or trans_batch is None:
                    continue

                gt_pose_seq = pose_batch[sample_idx, :T]
                gt_trans_seq = trans_batch[sample_idx, :T]

                if gt_pose_seq.shape[-1] < 6:
                    continue

                gt_root_axis = gt_pose_seq[:, :3]
                gt_body_axis = gt_pose_seq[:, 3:3 + 63]
                if gt_body_axis.shape[-1] < 63:
                    pad = torch.zeros(T, 63 - gt_body_axis.shape[-1], device=device, dtype=gt_body_axis.dtype)
                    gt_body_axis = torch.cat([gt_body_axis, pad], dim=-1)
                else:
                    gt_body_axis = gt_body_axis[:, :63]
                gt_body_axis = gt_body_axis.view(T, 21, 3)

                try:
                    gt_smpl_out = smpl_model(
                        root_orient=gt_root_axis,
                        pose_body=gt_body_axis.reshape(T, -1),
                        trans=gt_trans_seq,
                    )
                except Exception as exc:
                    print(f"SMPL forward failed for GT: {exc}")
                    continue

                gt_joints_all = gt_smpl_out.Jtr
                gt_joints_smpl24 = _to_smpl24_joint_positions(gt_joints_all)

                # ?定使用的trans
                if no_trans:
                    pred_trans = gt_trans_seq
                else:
                    pred_root_trans_all = pred_dict.get("root_trans_pred")
                    pred_trans = pred_root_trans_all[sample_idx] if pred_root_trans_all is not None else gt_trans_seq

                human_module = _get_human_pose_module(model)
                pred_joints_all = None
                pred_body_axis = None

                pred_full_pose_rotmat_all = pred_dict.get("pred_full_pose_rotmat")
                pred_full_pose_rotmat_seq = (
                    pred_full_pose_rotmat_all[sample_idx]
                    if isinstance(pred_full_pose_rotmat_all, torch.Tensor)
                    else None
                )

                if pred_full_pose_rotmat_seq is not None and pred_trans is not None:
                    try:
                        full_glb = pred_full_pose_rotmat_seq
                        if full_glb.dim() != 4 or full_glb.shape[0] != T:
                            raise ValueError(
                                f"Invalid pred_full_pose_rotmat shape: {tuple(full_glb.shape)}, expected [{T}, J, 3, 3]"
                            )

                        if human_module is not None and hasattr(human_module, "_global2local"):
                            parents = human_module.smpl_parents.tolist()
                            local_rot = human_module._global2local(full_glb, parents)
                        else:
                            default_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
                            local_rot = _global_to_local_rotmat(full_glb, default_parents)

                        pose_axis_full = t3d.matrix_to_axis_angle(local_rot.reshape(-1, 3, 3)).reshape(
                            T, full_glb.shape[1], 3
                        )
                        pred_root_axis = pose_axis_full[:, 0, :]
                        pred_body_axis = pose_axis_full[:, 1:22, :]
                        pred_smpl_out = smpl_model(
                            root_orient=pred_root_axis,
                            pose_body=pred_body_axis.reshape(T, -1),
                            trans=pred_trans,
                        )
                        pred_joints_all = pred_smpl_out.Jtr
                    except Exception as exc:
                        print(f"Predicted SMPL reconstruction from pred_full_pose_rotmat failed: {exc}")

                p_pred_all = pred_dict.get("p_pred")
                p_pred_seq = p_pred_all[sample_idx] if isinstance(p_pred_all, torch.Tensor) else None
                if (
                    pred_joints_all is None
                    and p_pred_seq is not None
                    and pred_trans is not None
                    and human_module is not None
                    and hasattr(human_module, "_reduced_glb_6d_to_full_glb_mat")
                    and hasattr(human_module, "_global2local")
                    and hasattr(human_module, "smpl_parents")
                ):
                    try:
                        reduced_pose = p_pred_seq.reshape(T, len(_REDUCED_POSE_NAMES), 6)
                        orientation_6d = human_imu_seq[:, :, -6:]
                        orientation_mat = t3d.rotation_6d_to_matrix(orientation_6d.reshape(-1, 6)).reshape(
                            T, human_imu_seq.shape[1], 3, 3
                        )
                        orientation_subset = orientation_mat[:, :len(_SENSOR_ROT_INDICES), :, :]
                        full_glb = human_module._reduced_glb_6d_to_full_glb_mat(
                            reduced_pose,
                            orientation_subset.reshape(T, len(_SENSOR_ROT_INDICES), 3, 3),
                        )
                        parents = human_module.smpl_parents.tolist()
                        local_rot = human_module._global2local(full_glb, parents)
                        pose_axis_full = t3d.matrix_to_axis_angle(local_rot.reshape(-1, 3, 3)).reshape(
                            T, full_glb.shape[1], 3
                        )
                        pred_root_axis = pose_axis_full[:, 0, :]
                        pred_body_axis = pose_axis_full[:, 1:22, :]
                        pred_smpl_out = smpl_model(
                            root_orient=pred_root_axis,
                            pose_body=pred_body_axis.reshape(T, -1),
                            trans=pred_trans,
                        )
                        pred_joints_all = pred_smpl_out.Jtr
                    except Exception as exc:
                        print(f"Predicted SMPL reconstruction from p_pred failed: {exc}")

                if pred_joints_all is None:
                    pred_joints_global_all = pred_dict.get("pred_joints_global")
                    if isinstance(pred_joints_global_all, torch.Tensor):
                        try:
                            pred_joints_all = pred_joints_global_all[sample_idx]
                        except Exception:
                            pred_joints_all = None

                if pred_joints_all is None:
                    continue

                # --- MPJPE ---
                pred_joints_eval = pred_joints_all[:, :num_eval_joints, :]
                gt_joints_eval = gt_joints_all[:, :num_eval_joints, :]
                pred_joints_rel = pred_joints_eval - pred_joints_eval[:, 0:1, :]
                gt_joints_rel = gt_joints_eval - gt_joints_eval[:, 0:1, :]
                joint_distances = torch.linalg.norm(pred_joints_rel - gt_joints_rel, dim=-1)
                metrics["mpjpe"].append(joint_distances.mean().item() * 100.0)

                # --- MPJRE ---
                if pred_body_axis is not None:
                    rot_error = torch.mean(torch.absolute(gt_body_axis - pred_body_axis)) * 57.2958
                    metrics["mpjre_angle"].append(rot_error.item())
                else:
                    metrics["mpjre_angle"].append(float("nan"))

                # --- Root Trans Error (非noTrans模式) ---
                if not no_trans and "root_trans_pred" in pred_dict:
                    pred_root_trans_seq = pred_dict["root_trans_pred"][sample_idx]
                    root_err = torch.linalg.norm(pred_root_trans_seq - gt_trans_seq, dim=-1).mean().item() * 100.0
                    metrics["root_trans_err"].append(root_err)
                    stage1_root_trans = pred_dict.get("stage1_root_trans_pred")
                    if isinstance(stage1_root_trans, torch.Tensor):
                        stage1_root_trans_seq = stage1_root_trans[sample_idx]
                        stage1_root_err = torch.linalg.norm(
                            stage1_root_trans_seq - gt_trans_seq,
                            dim=-1,
                        ).mean().item() * 100.0
                        metrics["stage1_root_trans_err"].append(stage1_root_err)

                # --- Hand Velocity Error (noTrans模式) ---
                if no_trans:
                    pred_hand_vel = pred_dict.get("pred_hand_glb_vel")
                    if pred_hand_vel is not None and T >= 2:
                        pred_hand_vel_seq = pred_hand_vel[sample_idx]
                        gt_hand_pos = select_hand_anchor_positions(gt_joints_smpl24)
                        gt_lhand_pos = gt_hand_pos[:, 0, :]
                        gt_rhand_pos = gt_hand_pos[:, 1, :]

                        dt = 1.0 / float(FRAME_RATE)
                        gt_lhand_vel = (gt_lhand_pos[1:] - gt_lhand_pos[:-1]) / dt
                        gt_rhand_vel = (gt_rhand_pos[1:] - gt_rhand_pos[:-1]) / dt

                        pred_lhand_vel = pred_hand_vel_seq[:-1, 0, :]
                        pred_rhand_vel = pred_hand_vel_seq[:-1, 1, :]

                        lhand_vel_err = torch.linalg.norm(pred_lhand_vel - gt_lhand_vel, dim=-1).mean().item()
                        rhand_vel_err = torch.linalg.norm(pred_rhand_vel - gt_rhand_vel, dim=-1).mean().item()

                        metrics["hand_vel_err_lhand"].append(lhand_vel_err)
                        metrics["hand_vel_err_rhand"].append(rhand_vel_err)
                        metrics["hand_vel_err_avg"].append((lhand_vel_err + rhand_vel_err) / 2.0)

                # --- Jitter ---
                if T >= 3:
                    accel = pred_joints_eval[2:] - 2 * pred_joints_eval[1:-1] + pred_joints_eval[:-2]
                    jitter = torch.linalg.norm(accel, dim=-1).mean().item() * 1000.0
                    metrics["jitter"].append(jitter)

                # --- Object Metrics ---
                has_object_mask = data_dict.get("has_object")
                has_object_sample = bool(has_object_mask[sample_idx].item()) if isinstance(has_object_mask, torch.Tensor) else True

                gt_obj_trans = batch_device.get("obj_trans")
                gt_obj_trans_seq = gt_obj_trans[sample_idx, :T] if gt_obj_trans is not None else None

                gt_lhand_contact = batch_device.get("lhand_contact")
                gt_rhand_contact = batch_device.get("rhand_contact")
                gt_obj_contact = batch_device.get("obj_contact")
                gt_lhand_contact_seq = gt_lhand_contact[sample_idx, :T].bool() if gt_lhand_contact is not None else None
                gt_rhand_contact_seq = gt_rhand_contact[sample_idx, :T].bool() if gt_rhand_contact is not None else None
                gt_obj_contact_seq = gt_obj_contact[sample_idx, :T].bool() if gt_obj_contact is not None else None

                pred_obj_trans_fusion = pred_dict.get("pred_obj_trans")
                if pred_obj_trans_fusion is None:
                    pred_obj_trans_fusion = pred_dict.get("p_obj_pred")
                pred_obj_trans_fusion = pred_obj_trans_fusion[sample_idx] if pred_obj_trans_fusion is not None else None

                pred_joints_smpl24 = _to_smpl24_joint_positions(pred_joints_all)
                pred_hand_pos_refined = select_hand_anchor_positions(pred_joints_smpl24)
                stage1_hand_all = pred_dict.get("stage1_pred_palm_glb_pos")
                if isinstance(stage1_hand_all, torch.Tensor):
                    stage1_hand_pos = stage1_hand_all[sample_idx, :T]
                else:
                    stage1_hand_pos = pred_hand_pos_refined

                def _sample_prediction_tensor(key):
                    value = pred_dict.get(key)
                    if not isinstance(value, torch.Tensor) or value.dim() < 2:
                        return None
                    return value[sample_idx, :T]

                pred_obj_trans_fusion_raw = pred_obj_trans_fusion
                pred_obj_trans_fusion_refine = _build_published_object_trans(
                    pred_obj_trans_fusion_raw,
                    stage1_hand_pos,
                    pred_hand_pos_refined,
                    _sample_prediction_tensor("pred_lhand_obj_trans"),
                    _sample_prediction_tensor("pred_rhand_obj_trans"),
                    _sample_prediction_tensor("pred_imu_obj_trans"),
                    _sample_prediction_tensor("pred_static_obj_trans"),
                    _sample_prediction_tensor("gating_weights"),
                )

                pred_obj_trans_fk = pred_dict.get("pred_obj_trans_fk")
                pred_obj_trans_fk = pred_obj_trans_fk[sample_idx] if pred_obj_trans_fk is not None else None

                pred_obj_vel = pred_dict.get("pred_obj_vel")
                pred_obj_vel = pred_obj_vel[sample_idx] if pred_obj_vel is not None else None

                pred_obj_trans_imu = None
                if evaluate_objects and has_object_sample and pred_obj_vel is not None:
                    dt = 1.0 / float(FRAME_RATE)
                    vel_scaled = pred_obj_vel * dt
                    disp = torch.cumsum(vel_scaled, dim=0)
                    if disp.shape[0] > 0:
                        zero_row = torch.zeros(1, 3, device=device, dtype=disp.dtype)
                        disp = torch.cat([zero_row, disp[:-1]], dim=0)
                    obj_trans_init = data_dict.get("obj_trans_init")
                    if isinstance(obj_trans_init, torch.Tensor):
                        init_pos = obj_trans_init[sample_idx]
                    else:
                        init_pos = gt_obj_trans_seq[0] if gt_obj_trans_seq is not None else torch.zeros(3, device=device)
                    pred_obj_trans_imu = init_pos.unsqueeze(0) + disp

                def _append_obj_error(name: str, pred_trans):
                    if evaluate_objects and has_object_sample and gt_obj_trans_seq is not None and pred_trans is not None:
                        err = torch.linalg.norm(pred_trans - gt_obj_trans_seq, dim=-1).mean().item() * 100.0
                        metrics[name].append(err)
                        return err
                    else:
                        metrics[name].append(float("nan"))
                        return float("nan")

                obj_err_fusion_raw = _append_obj_error(
                    "obj_trans_err_fusion_raw", pred_obj_trans_fusion_raw
                )
                obj_err_fusion_published = _append_obj_error(
                    "obj_trans_err_fusion_refine", pred_obj_trans_fusion_refine
                )
                _append_obj_error("obj_trans_err_fusion", pred_obj_trans_fusion_refine)
                _append_obj_error("obj_trans_err_fk", pred_obj_trans_fk)
                _append_obj_error("obj_trans_err_imu", pred_obj_trans_imu)

                # --- Object Jitter (Fusion trajectory) ---
                if evaluate_objects and has_object_sample and pred_obj_trans_fusion is not None and T >= 3:
                    obj_accel = (
                        pred_obj_trans_fusion[2:]
                        - 2 * pred_obj_trans_fusion[1:-1]
                        + pred_obj_trans_fusion[:-2]
                    )
                    obj_jitter = torch.linalg.norm(obj_accel, dim=-1).mean().item() * 1000.0
                    metrics["obj_jitter"].append(obj_jitter)
                else:
                    metrics["obj_jitter"].append(float("nan"))

                gt_hand_pos = select_hand_anchor_positions(gt_joints_smpl24)

                def compute_hoi_error(pred_obj_trans_seq_, pred_hand_pos):
                    if not (evaluate_objects and has_object_sample):
                        return float("nan")
                    return compute_hoi_error_at_gt_contact(
                        pred_obj_trans_seq_,
                        gt_obj_trans_seq,
                        pred_hand_pos,
                        gt_hand_pos,
                        gt_lhand_contact_seq,
                        gt_rhand_contact_seq,
                    )

                hoi_err_fusion = compute_hoi_error(
                    pred_obj_trans_fusion_refine, pred_hand_pos_refined
                )
                metrics["hoi_err_fusion"].append(hoi_err_fusion)
                metrics["hoi_err_fk"].append(
                    compute_hoi_error(pred_obj_trans_fk, pred_hand_pos_refined)
                )
                metrics["hoi_err_imu"].append(
                    compute_hoi_error(pred_obj_trans_imu, pred_hand_pos_refined)
                )

                if sequence_records is not None:
                    def _metadata_value(key: str, default=""):
                        value = batch.get(key, default)
                        if isinstance(value, (list, tuple)):
                            return value[sample_idx] if sample_idx < len(value) else default
                        if isinstance(value, torch.Tensor) and value.numel() > sample_idx:
                            return value.reshape(-1)[sample_idx].item()
                        return value

                    def _distance_cm(prediction, target, mask=None):
                        if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
                            return None
                        if prediction.shape != target.shape:
                            return None
                        values = torch.linalg.norm(prediction - target, dim=-1) * 100.0
                        if isinstance(mask, torch.Tensor):
                            mask = mask.bool()
                            if mask.shape != values.shape or not mask.any():
                                return None
                            values = values[mask]
                        return values

                    def _mean_value(values):
                        if not isinstance(values, torch.Tensor) or values.numel() == 0:
                            return float("nan")
                        finite = values[torch.isfinite(values)]
                        return finite.mean().item() if finite.numel() else float("nan")

                    def _series_summary(values):
                        if not isinstance(values, torch.Tensor) or values.numel() == 0:
                            return {}
                        values = values.detach().float().reshape(-1)
                        values = values[torch.isfinite(values)]
                        if values.numel() == 0:
                            return {}
                        count = int(values.numel())
                        edge = max(1, count // 10)
                        frame_axis = torch.arange(count, device=values.device, dtype=values.dtype)
                        centered_axis = frame_axis - frame_axis.mean()
                        denominator = torch.sum(centered_axis.square())
                        slope = (
                            torch.sum(centered_axis * (values - values.mean())) / denominator * 100.0
                            if denominator > 0
                            else values.new_zeros(())
                        )
                        profile = []
                        for bin_idx in range(10):
                            start = count * bin_idx // 10
                            end = count * (bin_idx + 1) // 10
                            profile.append(values[start:end].mean().item() if end > start else float("nan"))
                        return {
                            "mean": values.mean().item(),
                            "median": values.median().item(),
                            "p90": torch.quantile(values, 0.90).item(),
                            "p95": torch.quantile(values, 0.95).item(),
                            "max": values.max().item(),
                            "max_frame": int(torch.argmax(values).item()),
                            "first_10pct_mean": values[:edge].mean().item(),
                            "last_10pct_mean": values[-edge:].mean().item(),
                            "slope_per_100_frames": slope.item(),
                            "decile_means": profile,
                        }

                    published_frame_error = _distance_cm(
                        pred_obj_trans_fusion_refine, gt_obj_trans_seq
                    )
                    raw_frame_error = _distance_cm(
                        pred_obj_trans_fusion_raw, gt_obj_trans_seq
                    )
                    any_hand_contact = None
                    if (
                        isinstance(gt_lhand_contact_seq, torch.Tensor)
                        and isinstance(gt_rhand_contact_seq, torch.Tensor)
                    ):
                        any_hand_contact = gt_lhand_contact_seq | gt_rhand_contact_seq

                    candidate_predictions = _build_published_object_candidates(
                        pred_obj_trans_fusion_raw,
                        stage1_hand_pos,
                        pred_hand_pos_refined,
                        _sample_prediction_tensor("pred_lhand_obj_trans"),
                        _sample_prediction_tensor("pred_rhand_obj_trans"),
                        _sample_prediction_tensor("pred_imu_obj_trans"),
                        _sample_prediction_tensor("pred_static_obj_trans"),
                        _sample_prediction_tensor("gating_weights"),
                    )
                    candidate_error_series = {
                        name: _distance_cm(prediction, gt_obj_trans_seq)
                        for name, prediction in candidate_predictions.items()
                    }
                    valid_candidates = [
                        (name, values)
                        for name, values in candidate_error_series.items()
                        if isinstance(values, torch.Tensor) and values.shape == published_frame_error.shape
                    ] if isinstance(published_frame_error, torch.Tensor) else []

                    candidate_diagnostics = {
                        name: _mean_value(values)
                        for name, values in candidate_error_series.items()
                    }
                    gating_diagnostics = {}
                    normalized_weights = None
                    oracle_candidate_mean = float("nan")
                    if valid_candidates:
                        candidate_names = [name for name, _ in valid_candidates]
                        candidate_matrix = torch.stack([values for _, values in valid_candidates], dim=-1)
                        oracle_errors, oracle_indices = candidate_matrix.min(dim=-1)
                        oracle_candidate_mean = oracle_errors.mean().item()
                        oracle_fractions = []
                        for candidate_idx in range(len(candidate_names)):
                            oracle_fractions.append(
                                (oracle_indices == candidate_idx).float().mean().item()
                            )
                        candidate_diagnostics["oracle_per_frame"] = oracle_candidate_mean
                        candidate_diagnostics["oracle_fraction"] = dict(
                            zip(candidate_names, oracle_fractions)
                        )

                        gating_weights = _sample_prediction_tensor("gating_weights")
                        if (
                            isinstance(gating_weights, torch.Tensor)
                            and gating_weights.dim() == 2
                            and gating_weights.shape == candidate_matrix.shape
                        ):
                            normalized_weights = gating_weights.clamp_min(0.0)
                            normalized_weights = normalized_weights / normalized_weights.sum(
                                dim=-1, keepdim=True
                            ).clamp_min(1e-8)
                            selected_indices = normalized_weights.argmax(dim=-1)
                            selected_errors = candidate_matrix.gather(
                                -1, selected_indices.unsqueeze(-1)
                            ).squeeze(-1)
                            gate_fractions = []
                            for candidate_idx in range(len(candidate_names)):
                                gate_fractions.append(
                                    (selected_indices == candidate_idx).float().mean().item()
                                )
                            gating_diagnostics = {
                                "candidate_names": candidate_names,
                                "mean_weights": dict(zip(
                                    candidate_names,
                                    normalized_weights.mean(dim=0).tolist(),
                                )),
                                "argmax_fraction": dict(zip(candidate_names, gate_fractions)),
                                "argmax_matches_oracle_fraction": (
                                    selected_indices == oracle_indices
                                ).float().mean().item(),
                                "argmax_candidate_error_cm": selected_errors.mean().item(),
                                "entropy": (
                                    -(normalized_weights * normalized_weights.clamp_min(1e-8).log())
                                    .sum(dim=-1)
                                    .mean()
                                    .item()
                                ),
                            }
                            if (
                                isinstance(gt_lhand_contact_seq, torch.Tensor)
                                and isinstance(gt_rhand_contact_seq, torch.Tensor)
                            ):
                                contact_states = {
                                    "left_only": gt_lhand_contact_seq & ~gt_rhand_contact_seq,
                                    "right_only": gt_rhand_contact_seq & ~gt_lhand_contact_seq,
                                    "bimanual": gt_lhand_contact_seq & gt_rhand_contact_seq,
                                    "no_hand": ~gt_lhand_contact_seq & ~gt_rhand_contact_seq,
                                }
                                gating_diagnostics["mean_weights_by_gt_contact"] = {
                                    state_name: dict(zip(
                                        candidate_names,
                                        normalized_weights[state_mask].mean(dim=0).tolist(),
                                    ))
                                    for state_name, state_mask in contact_states.items()
                                    if state_mask.any()
                                }

                    refined_hand_error = _distance_cm(pred_hand_pos_refined, gt_hand_pos)
                    stage1_hand_error = _distance_cm(stage1_hand_pos, gt_hand_pos)
                    root_error = _distance_cm(pred_trans, gt_trans_seq)
                    stage1_root_error = None
                    stage1_root_trans = pred_dict.get("stage1_root_trans_pred")
                    if isinstance(stage1_root_trans, torch.Tensor):
                        stage1_root_error = _distance_cm(
                            stage1_root_trans[sample_idx, :T], gt_trans_seq
                        )

                    hoi_by_hand = {}
                    hoi_error_observations = []
                    for hand_name, hand_idx, contact_mask in (
                        ("left", 0, gt_lhand_contact_seq),
                        ("right", 1, gt_rhand_contact_seq),
                    ):
                        if (
                            isinstance(contact_mask, torch.Tensor)
                            and contact_mask.any()
                            and isinstance(pred_obj_trans_fusion_refine, torch.Tensor)
                        ):
                            relative_error = torch.linalg.norm(
                                (pred_obj_trans_fusion_refine - pred_hand_pos_refined[:, hand_idx])
                                - (gt_obj_trans_seq - gt_hand_pos[:, hand_idx]),
                                dim=-1,
                            ) * 100.0
                            contact_error = relative_error[contact_mask]
                            hoi_by_hand[hand_name] = contact_error.mean().item()
                            hoi_error_observations.append(contact_error)
                        else:
                            hoi_by_hand[hand_name] = float("nan")

                    gt_obj_delta = gt_obj_trans_seq[1:] - gt_obj_trans_seq[:-1]
                    gt_obj_step = torch.linalg.norm(gt_obj_delta, dim=-1)
                    contact_counts = {
                        "left": int(gt_lhand_contact_seq.sum().item())
                        if isinstance(gt_lhand_contact_seq, torch.Tensor) else 0,
                        "right": int(gt_rhand_contact_seq.sum().item())
                        if isinstance(gt_rhand_contact_seq, torch.Tensor) else 0,
                        "any_hand": int(any_hand_contact.sum().item())
                        if isinstance(any_hand_contact, torch.Tensor) else 0,
                    }
                    contact_diagnostics = {}
                    pred_contact_all = pred_dict.get("pred_hand_contact_prob")
                    if pred_contact_all is None:
                        pred_contact_all = pred_dict.get("contact_prob_pred")
                    if isinstance(pred_contact_all, torch.Tensor):
                        pred_contact_seq = pred_contact_all[sample_idx, :T]
                        for hand_name, hand_idx, target in (
                            ("left", 0, gt_lhand_contact_seq),
                            ("right", 1, gt_rhand_contact_seq),
                        ):
                            if (
                                pred_contact_seq.shape[-1] > hand_idx
                                and isinstance(target, torch.Tensor)
                            ):
                                prediction = pred_contact_seq[:, hand_idx] > hand_contact_threshold
                                target = target.bool()
                                true_positive = int((prediction & target).sum().item())
                                false_positive = int((prediction & ~target).sum().item())
                                false_negative = int((~prediction & target).sum().item())
                                precision_denominator = true_positive + false_positive
                                recall_denominator = true_positive + false_negative
                                precision = (
                                    true_positive / precision_denominator
                                    if precision_denominator else float("nan")
                                )
                                recall = (
                                    true_positive / recall_denominator
                                    if recall_denominator else float("nan")
                                )
                                f1_denominator = 2 * true_positive + false_positive + false_negative
                                contact_diagnostics[hand_name] = {
                                    "precision": precision,
                                    "recall": recall,
                                    "f1": (
                                        2 * true_positive / f1_denominator
                                        if f1_denominator else float("nan")
                                    ),
                                    "false_positive_frames": false_positive,
                                    "false_negative_frames": false_negative,
                                }

                    sequence_records.append({
                        "seq_name": str(_metadata_value("seq_name")),
                        "seq_path": str(_metadata_value("seq_path")),
                        "obj_name": str(_metadata_value("obj_name", "unknown_object")),
                        "num_frames": int(T),
                        "duration_s": float(T) / float(FRAME_RATE),
                        "obj_trans_err_fusion_raw_cm": obj_err_fusion_raw,
                        "obj_trans_err_fusion_published_cm": obj_err_fusion_published,
                        "hoi_err_fusion_cm": hoi_err_fusion,
                        "published_minus_raw_cm": obj_err_fusion_published - obj_err_fusion_raw,
                        "obj_error_profile_cm": _series_summary(published_frame_error),
                        "raw_obj_error_profile_cm": _series_summary(raw_frame_error),
                        "obj_error_on_contact_cm": _mean_value(
                            published_frame_error[any_hand_contact]
                            if isinstance(published_frame_error, torch.Tensor)
                            and isinstance(any_hand_contact, torch.Tensor)
                            and any_hand_contact.any() else None
                        ),
                        "obj_error_off_contact_cm": _mean_value(
                            published_frame_error[~any_hand_contact]
                            if isinstance(published_frame_error, torch.Tensor)
                            and isinstance(any_hand_contact, torch.Tensor)
                            and (~any_hand_contact).any() else None
                        ),
                        "hoi_error_by_hand_cm": hoi_by_hand,
                        "hoi_error_profile_cm": _series_summary(
                            torch.cat(hoi_error_observations)
                            if hoi_error_observations else None
                        ),
                        "candidate_error_cm": candidate_diagnostics,
                        "fusion_minus_oracle_candidate_cm": (
                            obj_err_fusion_published - oracle_candidate_mean
                        ),
                        "gating": gating_diagnostics,
                        "human_error_cm": {
                            "mpjpe": joint_distances.mean().item() * 100.0,
                            "root_trans": _mean_value(root_error),
                            "stage1_root_trans": _mean_value(stage1_root_error),
                            "refined_hand": _mean_value(refined_hand_error),
                            "stage1_hand": _mean_value(stage1_hand_error),
                        },
                        "contact_frames": contact_counts,
                        "contact_fraction": {
                            name: count / float(T) for name, count in contact_counts.items()
                        },
                        "contact_prediction": contact_diagnostics,
                        "gt_object_motion": {
                            "path_length_m": gt_obj_step.sum().item(),
                            "displacement_m": torch.linalg.norm(
                                gt_obj_trans_seq[-1] - gt_obj_trans_seq[0]
                            ).item(),
                            "mean_speed_mps": gt_obj_step.mean().item() * float(FRAME_RATE)
                            if gt_obj_step.numel() else 0.0,
                            "max_speed_mps": gt_obj_step.max().item() * float(FRAME_RATE)
                            if gt_obj_step.numel() else 0.0,
                        },
                    })

                # --- Contact F1 ---
                pred_hand_contact_prob = pred_dict.get("pred_hand_contact_prob")
                if pred_hand_contact_prob is None:
                    pred_hand_contact_prob = pred_dict.get("contact_prob_pred")
                if evaluate_contacts and pred_hand_contact_prob is not None:
                    pred_hand_contact_prob = pred_hand_contact_prob[sample_idx]
                    if pred_hand_contact_prob.shape[-1] < 2:
                        continue
                    pred_lhand = (pred_hand_contact_prob[:, 0] > hand_contact_threshold).long().cpu().numpy()
                    pred_rhand = (pred_hand_contact_prob[:, 1] > hand_contact_threshold).long().cpu().numpy()
                    pred_objc = None
                    if pred_hand_contact_prob.shape[-1] >= 3:
                        pred_objc = (pred_hand_contact_prob[:, 2] > object_contact_threshold).long().cpu().numpy()
                    else:
                        # Two-stage DiT interaction predicts left/right hand contacts only.
                        # Keep object-contact metric by falling back to union of two hands.
                        pred_objc = np.logical_or(pred_lhand > 0, pred_rhand > 0).astype(int)

                    def _binary_metric(metric_fn, y_true, y_pred, **kwargs):
                        if np.unique(y_true).size <= 1:
                            return float("nan")
                        return metric_fn(y_true, y_pred, zero_division=0, **kwargs)

                    if gt_lhand_contact_seq is not None:
                        gt_lhand = gt_lhand_contact_seq.cpu().numpy().astype(int)
                        metrics["contact_f1_lhand"].append(_binary_metric(f1_score, gt_lhand, pred_lhand))
                        metrics["contact_precision_lhand"].append(
                            _binary_metric(precision_score, gt_lhand, pred_lhand)
                        )
                        metrics["contact_recall_lhand"].append(_binary_metric(recall_score, gt_lhand, pred_lhand))
                        metrics["contact_f05_lhand"].append(
                            _binary_metric(fbeta_score, gt_lhand, pred_lhand, beta=0.5)
                        )

                    if gt_rhand_contact_seq is not None:
                        gt_rhand = gt_rhand_contact_seq.cpu().numpy().astype(int)
                        metrics["contact_f1_rhand"].append(_binary_metric(f1_score, gt_rhand, pred_rhand))
                        metrics["contact_precision_rhand"].append(
                            _binary_metric(precision_score, gt_rhand, pred_rhand)
                        )
                        metrics["contact_recall_rhand"].append(_binary_metric(recall_score, gt_rhand, pred_rhand))
                        metrics["contact_f05_rhand"].append(
                            _binary_metric(fbeta_score, gt_rhand, pred_rhand, beta=0.5)
                        )

                    if gt_obj_contact_seq is not None:
                        gt_objc = gt_obj_contact_seq.cpu().numpy().astype(int)
                        metrics["contact_f1_obj"].append(
                            _binary_metric(f1_score, gt_objc, pred_objc)
                        )

                    if (
                        gt_lhand_contact_seq is not None
                        and gt_rhand_contact_seq is not None
                        and gt_obj_contact_seq is not None
                    ):
                        gt_lhand_bool = gt_lhand_contact_seq.cpu().numpy().astype(bool)
                        gt_rhand_bool = gt_rhand_contact_seq.cpu().numpy().astype(bool)
                        gt_obj_bool = gt_obj_contact_seq.cpu().numpy().astype(bool)
                        hard_neg_mask = gt_obj_bool & (~gt_lhand_bool) & (~gt_rhand_bool)
                        if hard_neg_mask.any():
                            pred_lhand_bool = pred_lhand.astype(bool)
                            pred_rhand_bool = pred_rhand.astype(bool)
                            pred_any_hand = pred_lhand_bool | pred_rhand_bool
                            metrics["contact_hard_neg_fp_rate_any"].append(
                                float(pred_any_hand[hard_neg_mask].mean())
                            )
                            metrics["contact_hard_neg_fp_rate_lhand"].append(
                                float(pred_lhand_bool[hard_neg_mask].mean())
                            )
                            metrics["contact_hard_neg_fp_rate_rhand"].append(
                                float(pred_rhand_bool[hard_neg_mask].mean())
                            )

    # 计算平均?
    avg_metrics = {}
    for key, values in metrics.items():
        valid = [v for v in values if not np.isnan(v)]
        avg_metrics[key] = float(np.mean(valid)) if valid else float("nan")

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate IMUHOI Model")
    parser.add_argument("--config", type=str, default="configs/IMUHOI_train_rnn.yaml", help="配置文件??")
    parser.add_argument("--dataset", type=str, default="omomo", help="数据集名称（默?：omomo?")
    parser.add_argument("--smpl_model_path", type=str, default="datasets/smpl_models/smplh/male/model.npz", help="SMPL模型??")
    parser.add_argument("--num_workers", type=int, default=12, help="DataLoader workers")
    parser.add_argument("--no_trans", action="store_true", help="使用noTrans模式")
    parser.add_argument("--no_objects", action="store_true", help="跳过物体相关指标")
    parser.add_argument("--compare_3", action="store_true", help="比较FK/IMU方法")
    parser.add_argument("--model_arch", type=str, choices=["rnn", "dit", "mamba", "mamba_simple"], default=None, help="选择模型架构")
    parser.add_argument("--hp_ckpt", type=str, default=None, help="覆盖 HumanPose checkpoint")
    parser.add_argument("--interaction_ckpt", type=str, default=None, help="覆盖 Interaction/Object checkpoint")
    parser.add_argument("--vc_ckpt", type=str, default=None, help="覆盖 VelocityContact checkpoint")
    parser.add_argument(
        "--object_trans_arch",
        "--object-trans-arch",
        type=_normalize_object_trans_arch,
        choices=OBJECT_TRANS_ARCH_CHOICES,
        default=None,
        help=(
            "ObjectTrans架构；imuhoi为EgoIMU原?三?融合?"
            "imuhoi_4heads为?加静?头的四?融合?"
        ),
    )
    parser.add_argument(
        "--velocity_contact_arch", "--velocity-contact-arch",
        dest="velocity_contact_arch",
        type=str.lower,
        choices=VELOCITY_CONTACT_ARCH_CHOICES,
        default=None,
        help=(
            "VelocityContact架构；v2为velocity→obj_move→direct hand-contact级联；"
            "imuhoi为EgoIMU VC"
        ),
    )
    parser.add_argument("--ot_ckpt", type=str, default=None, help="覆盖 ObjectTrans checkpoint")
    parser.add_argument("--hand_contact_threshold", type=float, default=None, help="左右手接触评估阈?")
    parser.add_argument("--object_contact_threshold", type=float, default=None, help="物体接触评估阈?")
    parser.add_argument("--inference_mode", "--inference-mode", choices=["online", "offline"], default="online", help="RNN inference mode")
    parser.add_argument("--online_window", "--online-window", type=int, default=None, help="RNN online sliding window size")
    parser.add_argument(
        "--online_lookahead",
        "--online-lookahead",
        type=int,
        default=5,
        help="Future frames appended after the legacy RNN online update frame (strict tail)",
    )
    parser.add_argument(
        "--object_trans_state_feedback",
        "--object-trans-state-feedback",
        dest="object_trans_state_feedback",
        choices=["none", "fused"],
        default=None,
        help="覆盖 OT 融合状态反馈开关；none=shared-head无反馈，fused=显式OT state feedback",
    )
    parser.add_argument("--output_json", type=str, default=None, help="?选：保存评估结果JSON")
    parser.add_argument("--seed", type=int, default=None, help="?选：固定评估随机种子，用于可复现实验")
    parser.add_argument(
        "--simulate_imu_noise",
        "--simulate-imu-noise",
        action="store_true",
        help="在完整评估序列上?用确定性的 Dataset IMU ?声?",
    )
    parser.add_argument(
        "--simulate_imu_noise_seed",
        "--simulate-imu-noise-seed",
        type=int,
        default=42,
        help="Dataset IMU ?声基础种子；与序列?径混合以消除访问顺序影响?",
    )
    parser.add_argument(
        "--interaction_human_source",
        type=str,
        default="pred",
        choices=["pred", "gt"],
        help="Interaction分支人体来源: pred=使用HumanPose预测, gt=使用GT人体状?",
    )
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument(
        "--enable_ot_refine",
        dest="ot_refine",
        action="store_true",
        default=None,
        help="允?ObjectTrans refine覆盖人体pose/root trans",
    )
    refine_group.add_argument(
        "--disable_ot_refine",
        dest="ot_refine",
        action="store_false",
        help="禁用ObjectTrans人体refine，保留HumanPose输出",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {'noTrans' if args.no_trans else 'Normal'}")
    print(f"Interaction human source: {args.interaction_human_source}")
    print(
        f"Inference mode: {args.inference_mode} | online_window: "
        f"{args.online_window or 'config/default'} | online_lookahead: {args.online_lookahead}"
    )
    print(f"OT state feedback: {args.object_trans_state_feedback or 'config'}")
    print(f"VelocityContact architecture: {args.velocity_contact_arch or 'config/default'}")
    print(f"ObjectTrans architecture: {args.object_trans_arch or 'config/default'}")
    seed = _set_eval_seed(args.seed)
    print(f"Eval seed: {seed if seed is not None else 'unset'}")
    print(
        "simulate_imu_noise: "
        f"{'enabled' if args.simulate_imu_noise else 'disabled'}"
        + (f" | base_seed={args.simulate_imu_noise_seed}" if args.simulate_imu_noise else "")
    )

    try:
        config_for_dataset_lookup = load_config(args.config)
        default_config = get_default_dataset_config(config=config_for_dataset_lookup)
        dataset_runs = _build_dataset_runs(args, default_config)
    except Exception as exc:
        print(f"[Eval] Argument error: {exc}")
        return

    all_results = {}
    effective_ot_state_feedback = None
    for dataset_name, dataset_cfg in dataset_runs:
        print(f"\n=== Evaluating dataset: {dataset_name} ===")
        config = load_config(args.config)
        if args.model_arch is not None:
            config.model_arch = args.model_arch
        if args.object_trans_state_feedback is not None:
            config.object_trans_state_feedback = args.object_trans_state_feedback == "fused"
        elif not hasattr(config, "object_trans_state_feedback"):
            config.object_trans_state_feedback = False
        effective_ot_state_feedback = bool(getattr(config, "object_trans_state_feedback", False))
        if args.velocity_contact_arch is not None:
            config.velocity_contact_arch = args.velocity_contact_arch
        config.velocity_contact_arch = normalize_velocity_contact_arch(
            getattr(config, "velocity_contact_arch", "current")
        )
        if args.object_trans_arch is not None:
            config.object_trans_arch = args.object_trans_arch
        config.object_trans_arch = _normalize_object_trans_arch(
            getattr(config, "object_trans_arch", "current")
        )
        ot_refine = False if is_imuhoi_object_trans_arch(
            getattr(config, "object_trans_arch", "current")
        ) else (
            bool(args.ot_refine)
            if args.ot_refine is not None
            else bool(getattr(config, "enable_ot_refine", False))
        )
        config.enable_ot_refine = ot_refine

        if args.num_workers is not None:
            config.num_workers = args.num_workers
        if not hasattr(config, "debug"):
            config.debug = False
        hand_contact_threshold = float(
            args.hand_contact_threshold
            if args.hand_contact_threshold is not None
            else getattr(config, "hand_contact_threshold", 0.5)
        )
        object_contact_threshold = float(
            args.object_contact_threshold
            if args.object_contact_threshold is not None
            else getattr(config, "object_contact_threshold", 0.5)
        )

        modules_override = dataset_cfg.get("modules")
        module_paths = None
        if hasattr(config, "pretrained_modules") and config.pretrained_modules:
            module_paths = dict(config.pretrained_modules)
        if modules_override:
            _apply_module_overrides(config, modules_override)
            module_paths = dict(config.pretrained_modules) if getattr(config, "pretrained_modules", None) else module_paths
        module_paths = _apply_cli_module_overrides(module_paths, args)
        module_paths = _filter_pipeline_module_paths(module_paths)
        model_arch = str(getattr(config, "model_arch", "rnn")).lower()
        has_velocity_contact_ckpt, has_object_trans_ckpt = _object_contact_capabilities(module_paths)

        if args.smpl_model_path:
            config.body_model_path = args.smpl_model_path

        smpl_model_path = config.get("body_model_path", "datasets/smpl_models/smplh/neutral/model.npz")
        try:
            smpl_model = load_smpl_model(smpl_model_path, device)
        except FileNotFoundError as exc:
            print(f"[Eval] Skipping '{dataset_name}': {exc}")
            continue

        model = load_model(
            config,
            device,
            no_trans=args.no_trans,
            module_paths=module_paths,
            object_trans_arch=args.object_trans_arch,
            velocity_contact_arch=args.velocity_contact_arch,
        )
        model_kind = _infer_model_kind(model)
        evaluate_contacts = (not args.no_objects) and has_velocity_contact_ckpt
        evaluate_objects = (not args.no_objects) and has_velocity_contact_ckpt
        run_object_branch = evaluate_objects and has_object_trans_ckpt
        if model_kind == "human_pose" and evaluate_objects:
            print("[Eval] 当前加载的是 human-only 模型；?需评估物体，?提? pretrained_modules.interaction ? --interaction_ckpt?")
            evaluate_objects = False
            evaluate_contacts = False
            run_object_branch = False
        elif model_kind == "pipeline" and not args.no_objects:
            if not has_velocity_contact_ckpt:
                print(
                    "[Eval] ?加载 velocity_contact/interaction checkpoint；跳? contact/object 指标?"
                    "避免使用随机初?化? VC 分支?"
                )
            elif not has_object_trans_ckpt:
                print(
                    "[Eval] ?加载 object_trans/interaction checkpoint；不运?? object_trans 分支?"
                    "仅评? VC 产生? contact ? obj_vel/IMU object baseline?"
                )

        data_dir_default = get_full_sequence_test_path(dataset_cfg)
        if data_dir_default is None:
            print(f"[Eval] Skipping '{dataset_name}' (no dataset directory configured).")
            continue

        try:
            data_path = resolve_dataset_path(data_dir_default, PROJECT_ROOT)
        except ValueError as exc:
            print(f"[Eval] Skipping '{dataset_name}': {exc}")
            continue

        if not data_path.exists():
            print(f"[Eval] Skipping '{dataset_name}' (data not found at {data_path}).")
            continue

        print(f"Loading test dataset from: {data_path}")

        test_window = config.test.get("window", config.train.get("window", 60))
        test_dataset = IMUDataset(
            data_dir=str(data_path),
            window_size=test_window,
            debug=config.get("debug", False),
            simulate_imu_noise=bool(args.simulate_imu_noise),
            simulate_imu_noise_seed=args.simulate_imu_noise_seed,
            min_obj_contact_frames=0,
            full_sequence=True,
            resolve_bimanual_contact_conflicts=config.get("resolve_bimanual_contact_conflicts", False),
        )

        if len(test_dataset) == 0:
            print(f"[Eval] Skipping '{dataset_name}' (dataset is empty).")
            continue

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.get("num_workers", 0),
            pin_memory=True,
            drop_last=False,
            worker_init_fn=_seed_worker if seed is not None else None,
            generator=(torch.Generator().manual_seed(seed) if seed is not None else None),
        )

        print(f"Dataset size: {len(test_dataset)} | Loader batches: {len(test_loader)}")
        print(f"OT human refine: {'enabled' if ot_refine else 'disabled'}")

        eval_start_time = time.time()
        results = evaluate_model(
            model,
            smpl_model,
            test_loader,
            config,
            device,
            no_trans=args.no_trans,
            evaluate_objects=evaluate_objects,
            evaluate_contacts=evaluate_contacts,
            run_object_branch=run_object_branch,
            compare_three=args.compare_3,
            interaction_human_source=args.interaction_human_source,
            ot_refine=ot_refine,
            hand_contact_threshold=hand_contact_threshold,
            object_contact_threshold=object_contact_threshold,
            inference_mode=args.inference_mode,
            online_window=args.online_window,
            online_lookahead=args.online_lookahead,
        )

        eval_duration = time.time() - eval_start_time
        all_results[dataset_name] = results

        def _fmt(key: str) -> str:
            val = results.get(key, float("nan"))
            return f"{val:.4f}" if not np.isnan(val) else "NaN"

        print("\n--- Evaluation Results ---")
        print(f"MPJPE (cm):                     {_fmt('mpjpe')}")
        print(f"MPJRE (deg):                    {_fmt('mpjre_angle')}")
        if not args.no_trans:
            print(f"Root Trans Error (cm):          {_fmt('root_trans_err')}")
            if not np.isnan(results.get("stage1_root_trans_err", float("nan"))):
                print(f"Stage1 Root Trans Error (cm):   {_fmt('stage1_root_trans_err')}")
        else:
            print(f"Hand Vel Error L (m/s):         {_fmt('hand_vel_err_lhand')}")
            print(f"Hand Vel Error R (m/s):         {_fmt('hand_vel_err_rhand')}")
            print(f"Hand Vel Error Avg (m/s):       {_fmt('hand_vel_err_avg')}")
        print(f"Jitter (mm/frame^2):            {_fmt('jitter')}")

        if evaluate_objects:
            print("\n--- Object Translation Errors ---")
            print(f"Fusion raw (cm):                {_fmt('obj_trans_err_fusion_raw')}")
            print(f"Fusion refine/published (cm):   {_fmt('obj_trans_err_fusion_refine')}")
            print(f"FK (cm):                        {_fmt('obj_trans_err_fk')}")
            print(f"IMU (cm):                       {_fmt('obj_trans_err_imu')}")

            print("\n--- HOI Errors ---")
            print(f"Fusion published (cm):          {_fmt('hoi_err_fusion')}")
            print(f"FK (cm):                        {_fmt('hoi_err_fk')}")
            print(f"IMU (cm):                       {_fmt('hoi_err_imu')}")

        if evaluate_contacts:
            print("\n--- Contact Prediction F1 ---")
            print(f"Hand Threshold:                 {hand_contact_threshold:.4f}")
            print(f"Object Threshold:               {object_contact_threshold:.4f}")
            print(f"Left Hand:                      {_fmt('contact_f1_lhand')}")
            print(f"Right Hand:                     {_fmt('contact_f1_rhand')}")
            print(f"Object:                         {_fmt('contact_f1_obj')}")
            print("\n--- Contact Precision-Oriented Metrics ---")
            print(f"Left Precision:                 {_fmt('contact_precision_lhand')}")
            print(f"Right Precision:                {_fmt('contact_precision_rhand')}")
            print(f"Left Recall:                    {_fmt('contact_recall_lhand')}")
            print(f"Right Recall:                   {_fmt('contact_recall_rhand')}")
            print(f"Left F0.5:                      {_fmt('contact_f05_lhand')}")
            print(f"Right F0.5:                     {_fmt('contact_f05_rhand')}")
            print(f"HardNeg FP Any Hand:            {_fmt('contact_hard_neg_fp_rate_any')}")
            print(f"HardNeg FP Left:                {_fmt('contact_hard_neg_fp_rate_lhand')}")
            print(f"HardNeg FP Right:               {_fmt('contact_hard_neg_fp_rate_rhand')}")
        elif not args.no_objects:
            print("\n--- Contact Prediction ---")
            print("Skipped: no velocity_contact/interaction checkpoint was loaded.")

        print(f"\n评估耗时: {eval_duration:.2f}?")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": args.config,
            "inference_mode": args.inference_mode,
            "online_window": args.online_window,
            "online_lookahead": args.online_lookahead,
            "object_trans_state_feedback": effective_ot_state_feedback,
            "object_trans_state_feedback_override": args.object_trans_state_feedback,
            "velocity_contact_arch": args.velocity_contact_arch or getattr(load_config(args.config), "velocity_contact_arch", "current"),
            "object_trans_arch": args.object_trans_arch or getattr(load_config(args.config), "object_trans_arch", "current"),
            "seed": seed,
            "results": all_results,
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Eval] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
