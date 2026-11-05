"""Frozen external human-pose estimator behind the HumanPoseModule interface.

HP plug-in experiment (2026-11-05): VC and OT are retrained on top of a comparison method's
human pose (DIP, TIP, TransPose or GlobalPose from Comparisons/) to test IMUHOI++'s
interaction stages as a plug-in for other IMU pose estimators.  The method itself runs once
per sequence in its own online protocol and is cached (``utils/external_hp_cache.py``);
``build_model_input_dict`` attaches the cached frames of every window as ``ext_hp_*``.

This module turns those frames into exactly the outputs HumanPoseModule publishes, with the
same skeleton machinery (zero-betas SMPL-24 rest joints of each sample's body-model family):
global/local rotations, root-relative FK joints, wrist/palm anchors, the reduced root-relative
pose ``p_pred`` and a root trajectory.  Its own choices, all documented where they are made:

* Root: the method's per-frame root displacement integrated from the window's initial
  translation, as HumanPose integrates its velocities (GT at training-window starts, the
  published trajectory online).  Online this reproduces the method's own trajectory, which
  also starts from the ground-truth first frame.  No ground correction is added.
* Latency: a method that looks ``L`` frames ahead (DIP and TransPose use 5 future frames)
  has, at a window's last frame, emitted poses only up to ``L`` frames earlier.  The last
  ``L`` frames of every window therefore hold the last available pose and extrapolate the
  root at its last velocity, in training and online alike, so the online lookahead never
  sees beyond its own window.
* No output smoothing (``human_pose_smoothing_half_width`` belongs to our HP).

Configure it with ``external_hp_cache_dir`` (and optionally ``external_hp_method`` as a
guard).  The HP "checkpoint" of such a run is the cache's manifest.json.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from configs import FRAME_RATE, _REDUCED_INDICES, _REDUCED_POSE_NAMES
from utils.external_hp_cache import (
    MANIFEST_NAME,
    external_hp_cache_dir,
    read_manifest,
    resolve_cache_dir,
)
from utils.human_pose import (
    WRIST_JOINT_INDICES,
    load_body_models_from_cfg,
    select_hand_anchor_positions,
    select_wrist_positions,
)

from .human_pose import HumanPoseModule
from .online import _history_tensor, normalize_inference_mode

SMPL24_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)


def is_external_hp(cfg) -> bool:
    return external_hp_cache_dir(cfg) is not None


def is_external_hp_manifest(path) -> bool:
    return path is not None and Path(str(path)).name == MANIFEST_NAME


def local_to_global_rotations(local: torch.Tensor) -> torch.Tensor:
    """``[..., 24, 3, 3]`` local SMPL rotations -> global ones (joint 0 is the root orientation)."""
    joints = [local[..., 0, :, :]]
    for index in range(1, len(SMPL24_PARENTS)):
        joints.append(torch.matmul(joints[SMPL24_PARENTS[index]], local[..., index, :, :]))
    return torch.stack(joints, dim=-3)


def global_to_local_rotations(global_rot: torch.Tensor) -> torch.Tensor:
    parents = torch.as_tensor(SMPL24_PARENTS[1:], device=global_rot.device)
    local = global_rot.clone()
    local[..., 1:, :, :] = torch.matmul(
        global_rot.index_select(-3, parents).transpose(-1, -2), global_rot[..., 1:, :, :]
    )
    return local


class ExternalHumanPoseModule(nn.Module):
    """Replays a cached external pose estimator with HumanPoseModule's outputs (no parameters)."""

    # Shared with HumanPoseModule so both HP sources build joints in the same skeleton.
    _smpl24_rest_joints = HumanPoseModule._smpl24_rest_joints
    _resolve_batch_body_model_types = HumanPoseModule._resolve_batch_body_model_types
    _compute_root_velocity_from_trans = HumanPoseModule._compute_root_velocity_from_trans

    def __init__(self, cfg, device, no_trans: bool = False):
        super().__init__()
        if no_trans:
            raise ValueError("an external HP estimates its own root translation; no_trans is not supported")
        cache_dir = external_hp_cache_dir(cfg)
        if cache_dir is None:
            raise ValueError("ExternalHumanPoseModule needs cfg.external_hp_cache_dir")
        manifest = read_manifest(cache_dir)
        method = str(manifest["method"]).strip().lower()
        expected = getattr(cfg, "external_hp_method", None) if not isinstance(cfg, dict) else cfg.get("external_hp_method")
        if expected and str(expected).strip().lower() != method:
            raise ValueError(f"external_hp_method={expected!r} but {cache_dir} caches {method!r}")
        self.cfg = cfg
        self.device = device
        self.no_trans = False
        self.cache_dir = cache_dir
        self.method = method
        self.latency = max(0, int(manifest.get("latency_frames", 0)))
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))
        self.smoothing_half_width = 0
        self.hand_joint_indices = tuple(WRIST_JOINT_INDICES)
        self.smpl_parents = torch.tensor(SMPL24_PARENTS, dtype=torch.long)
        models, default_type = load_body_models_from_cfg(cfg, num_betas=16)
        self.body_models = nn.ModuleDict(models)
        self.default_body_model_type = default_type
        self.body_model = self.body_models[default_type]
        self.requires_grad_(False)
        print(
            f"External HumanPose: {method} from {cache_dir} "
            f"(latency {self.latency} frames, dataset {manifest.get('dataset')})"
        )

    def check_source(self, path) -> None:
        """An external-HP run's HP 'checkpoint' must be this cache's manifest (or directory)."""
        if path is None:
            return
        if resolve_cache_dir(path) != self.cache_dir:
            raise ValueError(
                f"HP checkpoint {path} does not belong to external_hp_cache_dir {self.cache_dir}"
            )

    # ---- geometry ---------------------------------------------------------------------------
    def _fk(self, local: torch.Tensor, body_model_type) -> torch.Tensor:
        """Root-relative SMPL-24 joints ``[B, T, 24, 3]`` from local rotations ``[B, T, 24, 3, 3]``.

        Same rest joints and parents as HumanPoseModule._fk_from_smpl24_rest, fed with the
        rotation matrices directly instead of an axis-angle round trip.
        """
        from human_body_prior.body_model.lbs import batch_rigid_transform

        batch_size, seq_len = local.shape[:2]
        families = self._resolve_batch_body_model_types(body_model_type, batch_size)
        rest = torch.stack(
            [self._smpl24_rest_joints(family, local.device, local.dtype) for family in families]
        )
        rest = rest.unsqueeze(1).expand(batch_size, seq_len, 24, 3).reshape(-1, 24, 3)
        joints, _ = batch_rigid_transform(
            local.reshape(-1, 24, 3, 3), rest, self.smpl_parents, dtype=local.dtype
        )
        return joints.reshape(batch_size, seq_len, 24, 3)

    def _fk_chunked(self, local: torch.Tensor, body_model_type, chunk_rows: Optional[int]) -> torch.Tensor:
        batch_size, seq_len = local.shape[:2]
        step = seq_len if not chunk_rows else max(1, int(chunk_rows) // max(batch_size, 1))
        if step >= seq_len:
            return self._fk(local, body_model_type)
        return torch.cat(
            [self._fk(local[:, start:start + step], body_model_type) for start in range(0, seq_len, step)],
            dim=1,
        )

    def _hold_unavailable_tail(self, rot6d: torch.Tensor, disp: torch.Tensor):
        latency = self.latency
        seq_len = rot6d.shape[1]
        if latency <= 0 or seq_len <= latency:
            return rot6d, disp
        last = seq_len - 1 - latency
        rot6d = rot6d.clone()
        disp = disp.clone()
        rot6d[:, last + 1:] = rot6d[:, last:last + 1]
        disp[:, last + 1:] = disp[:, last:last + 1]
        return rot6d, disp

    def _outputs(self, global_rot, joints_local, root_trans, root_vel, dtype) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = global_rot.shape[:2]
        root_rot = global_rot[:, :, 0]
        reduced = torch.matmul(
            root_rot.unsqueeze(2).transpose(-1, -2), global_rot[:, :, list(_REDUCED_INDICES)]
        )
        p_pred = matrix_to_rotation_6d(reduced.reshape(-1, 3, 3)).reshape(
            batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6
        )
        joints_global = joints_local + root_trans.unsqueeze(2)
        root_vel_local = torch.matmul(root_rot.transpose(-1, -2), root_vel.unsqueeze(-1)).squeeze(-1)
        out = {
            "p_pred": p_pred,
            "pred_full_pose_rotmat": global_rot,
            "pred_full_pose_6d": matrix_to_rotation_6d(global_rot.reshape(-1, 3, 3)).reshape(
                batch_size, seq_len, 24, 6
            ),
            "pred_joints_local": joints_local,
            "pred_joints_global": joints_global,
            "pred_hand_glb_pos": select_wrist_positions(joints_global),
            "pred_palm_glb_pos": select_hand_anchor_positions(joints_global),
            "pred_foot_pos": joints_local[:, :, [7, 8], :],
            "root_trans_pred": root_trans,
            "root_trans_unadjusted_pred": root_trans,
            "root_vel_pred": root_vel,
            "root_vel_unadjusted_pred": root_vel,
            "root_vel_local_pred": root_vel_local,
            "ground_correction_pred": root_trans.new_zeros(batch_size, seq_len),
        }
        return {key: value.to(dtype) for key, value in out.items()}

    # ---- HumanPoseModule interface ------------------------------------------------------------
    def forward(self, data_dict: dict) -> Dict[str, torch.Tensor]:
        rot6d = data_dict.get("ext_hp_local_rot6d")
        disp = data_dict.get("ext_hp_root_disp")
        if not isinstance(rot6d, torch.Tensor) or not isinstance(disp, torch.Tensor):
            raise KeyError(
                "external HP inputs (ext_hp_local_rot6d / ext_hp_root_disp) are missing: the batch "
                "must go through build_model_input_dict with cfg.external_hp_cache_dir set"
            )
        trans_init = data_dict.get("trans_init")
        if not isinstance(trans_init, torch.Tensor):
            raise KeyError("external HP needs data_dict['trans_init'] to anchor the root trajectory")
        reference = data_dict.get("human_imu")
        dtype = reference.dtype if isinstance(reference, torch.Tensor) else rot6d.dtype
        batch_size, seq_len = rot6d.shape[:2]
        if trans_init.dim() == 3:
            trans_init = trans_init[:, 0]
        rot6d, disp = self._hold_unavailable_tail(rot6d, disp)
        with torch.autocast(device_type=rot6d.device.type, enabled=False):
            local = rotation_6d_to_matrix(rot6d.float().reshape(-1, 6)).reshape(batch_size, seq_len, 24, 3, 3)
            global_rot = local_to_global_rotations(local)
            joints_local = self._fk(local, data_dict.get("body_model_type"))
            disp = disp.float()
            start = trans_init.float().to(rot6d.device).reshape(batch_size, 1, 3)
            root_trans = torch.cat(
                (start, start + torch.cumsum(disp[:, 1:], dim=1)), dim=1
            )
            outputs = self._outputs(global_rot, joints_local, root_trans, disp * self.fps, dtype)
        return outputs

    def finalize_online_frame(self, frame: dict, history: dict) -> dict:
        """Advance the published root by this frame's displacement (no ground correction)."""
        previous = _history_tensor(history, "root_trans_pred", -1) if history else None
        if previous is None or "root_vel_pred" not in frame:
            return frame
        updated = dict(frame)
        root_trans = previous.unsqueeze(1) + frame["root_vel_pred"] / self.fps
        updated["root_trans_pred"] = root_trans
        updated["root_trans_unadjusted_pred"] = root_trans
        joints_local = frame.get("pred_joints_local")
        if isinstance(joints_local, torch.Tensor):
            joints_global = joints_local + root_trans.unsqueeze(2)
            updated["pred_joints_global"] = joints_global
            updated["pred_hand_glb_pos"] = select_wrist_positions(joints_global)
            updated["pred_palm_glb_pos"] = select_hand_anchor_positions(joints_global)
        return updated

    def inference(
        self,
        data_dict: dict,
        inference_mode: str = "offline",
        online_state: dict = None,
        return_online_state: bool = False,
        **_,
    ):
        if normalize_inference_mode(inference_mode) != "offline":
            raise NotImplementedError(
                "ExternalHumanPoseModule runs offline windows only; online inference goes through "
                "IMUHOIModel, which drives it window by window"
            )
        output = self.forward(data_dict)
        if return_online_state:
            return output, online_state or {}
        return output

    def derive_refined_human_outputs(self, results: dict, data_dict: dict, chunk_rows: Optional[int] = None) -> None:
        """Skeleton of the OT-refined pose for IMUHOIModel._derive_refined_human_outputs.

        The OT refine head corrects the ten reduced joints (root-relative, ``p_pred`` layout) and
        the root translation.  The refined skeleton keeps every other rotation of the external
        estimate (``stage1_pred_full_pose_rotmat``) and replaces only those ten.
        """
        refined_pose = results["p_pred"]
        refined_root = results["root_trans_pred"]
        base = results.get("stage1_pred_full_pose_rotmat", results.get("pred_full_pose_rotmat"))
        if not isinstance(base, torch.Tensor):
            raise KeyError("refined outputs of an external HP need the stage-1 full-pose rotations")
        batch_size, seq_len = refined_pose.shape[:2]
        dtype = refined_pose.dtype
        with torch.autocast(device_type=refined_pose.device.type, enabled=False):
            base = base[:, :seq_len].float()
            root_rot = base[:, :, 0]
            reduced = rotation_6d_to_matrix(refined_pose.float().reshape(-1, 6)).reshape(
                batch_size, seq_len, len(_REDUCED_POSE_NAMES), 3, 3
            )
            global_rot = base.clone()
            global_rot[:, :, list(_REDUCED_INDICES)] = torch.matmul(root_rot.unsqueeze(2), reduced)
            local = global_to_local_rotations(global_rot)
            joints_local = self._fk_chunked(local, data_dict.get("body_model_type"), chunk_rows)
            root_trans = refined_root.float()
            root_vel = self._compute_root_velocity_from_trans(root_trans)
            outputs = self._outputs(global_rot, joints_local, root_trans, root_vel, dtype)
        for key in (
            "root_vel_pred", "root_vel_local_pred", "pred_joints_local", "pred_joints_global",
            "pred_hand_glb_pos", "pred_palm_glb_pos", "pred_full_pose_rotmat", "pred_full_pose_6d",
        ):
            results[key] = outputs[key]


def maybe_build_external_human_pose(cfg, device, hp_ckpt=None, no_trans: bool = False):
    """The external HP of an external-HP run (validated against its HP 'checkpoint'), else None.

    Fails loudly when only one of the two says "external": the trainers would otherwise load
    a manifest as weights, or silently train on another HP.
    """
    external_cfg = is_external_hp(cfg)
    external_ckpt = is_external_hp_manifest(hp_ckpt)
    if not external_cfg and not external_ckpt:
        return None
    if not external_cfg:
        raise ValueError(
            f"HP checkpoint {hp_ckpt} is an external HP cache manifest, but external_hp_cache_dir is not set"
        )
    module = ExternalHumanPoseModule(cfg, device, no_trans=no_trans)
    if hp_ckpt is not None and str(hp_ckpt).strip():
        if not external_ckpt:
            raise ValueError(
                f"external_hp_cache_dir is set but the HP checkpoint is {hp_ckpt}; pass the cache's {MANIFEST_NAME}"
            )
        module.check_source(hp_ckpt)
    return module
