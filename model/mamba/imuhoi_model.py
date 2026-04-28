"""Unified IMUHOI model for the Mamba path."""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from utils.rotation_conversions import rotation_6d_to_matrix

from .human_pose import HumanPoseModule
from .interaction import InteractionModule


def _load_checkpoint(model, checkpoint_path, device, strict: bool = False, use_ema: bool = True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if use_ema and checkpoint.get("ema_state_dict") is not None:
            state_dict = checkpoint["ema_state_dict"]
        else:
            state_dict = checkpoint.get("module_state_dict", checkpoint.get("model_state_dict", checkpoint))
    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        model_state = model.state_dict()
        filtered = {}
        skipped = 0
        for key, value in state_dict.items():
            mapped_key = key
            if mapped_key not in model_state and mapped_key.startswith("module.") and mapped_key[7:] in model_state:
                mapped_key = mapped_key[7:]
            elif mapped_key not in model_state and f"module.{mapped_key}" in model_state:
                mapped_key = f"module.{mapped_key}"
            if mapped_key not in model_state:
                continue
            if model_state[mapped_key].shape != value.shape:
                skipped += 1
                continue
            filtered[mapped_key] = value
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"Skipped {skipped} checkpoint tensors with mismatched shapes")
        if missing:
            print(f"Missing checkpoint tensors: {len(missing)}")
        if unexpected:
            print(f"Unexpected checkpoint tensors: {len(unexpected)}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    return checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0


def _flatten_lstm_parameters(module):
    for child in module.children():
        if isinstance(child, torch.nn.LSTM):
            child.flatten_parameters()
        else:
            _flatten_lstm_parameters(child)


class IMUHOIModel(nn.Module):
    """Mamba IMUHOI stack: Stage-1 human pose followed by Stage-2 interaction."""

    @staticmethod
    def _fk_obj_trans_baseline_hard(
        pred_hand_contact_prob: torch.Tensor,
        pred_hand_positions: torch.Tensor,
        obj_rotm: torch.Tensor,
        obj_trans_init: torch.Tensor,
        contact_threshold: float = 0.5,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = pred_hand_contact_prob.shape
        device = pred_hand_contact_prob.device
        dtype = pred_hand_positions.dtype
        out = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
        if seq_len == 0:
            return out
        out[:, 0] = obj_trans_init

        l_contact = (pred_hand_contact_prob[..., 0] > contact_threshold).float()
        r_contact = (pred_hand_contact_prob[..., 1] > contact_threshold).float()
        l_contact[:, 0] = 0.0
        r_contact[:, 0] = 0.0
        l_start = torch.zeros_like(l_contact)
        r_start = torch.zeros_like(r_contact)
        if seq_len > 1:
            l_start[:, 1:] = ((l_contact[:, 1:] > 0) & (l_contact[:, :-1] == 0)).float()
            r_start[:, 1:] = ((r_contact[:, 1:] > 0) & (r_contact[:, :-1] == 0)).float()

        z_unit = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        default_len = torch.tensor(0.1, device=device, dtype=dtype)
        for b in range(batch_size):
            current = -1
            obj_pos = obj_trans_init[b].clone()
            dir_local = None
            dist = None
            for t in range(seq_len):
                new = -1
                if l_start[b, t] > 0:
                    new = 0
                elif r_start[b, t] > 0:
                    new = 1

                if current == 0:
                    has_contact = l_contact[b, t] > 0
                    has_other = r_contact[b, t] > 0
                elif current == 1:
                    has_contact = r_contact[b, t] > 0
                    has_other = l_contact[b, t] > 0
                else:
                    has_contact = False
                    has_other = False

                if new != -1 or ((not has_contact) and has_other):
                    current = new if new != -1 else 1 - current
                    hand0 = pred_hand_positions[b, t, current]
                    r0 = obj_rotm[b, t]
                    vec_world = obj_pos - hand0
                    dist0 = torch.norm(vec_world)
                    if dist0 > 1e-6:
                        unit_world = vec_world / dist0
                    else:
                        unit_world = z_unit
                        dist0 = default_len
                    dir_local = r0.transpose(0, 1) @ unit_world
                    dist = dist0
                elif (not has_contact) and current != -1:
                    current = -1
                    dir_local = None
                    dist = None

                if current == -1:
                    out[b, t] = obj_pos
                else:
                    obj_pos = pred_hand_positions[b, t, current] + (obj_rotm[b, t] @ dir_local) * dist
                    out[b, t] = obj_pos
        return out

    def __init__(self, cfg, device, no_trans: bool = False):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.no_trans = bool(no_trans)
        self.use_refined_outputs = self._cfg_bool("use_refined_outputs", True)

        self.human_pose_module = HumanPoseModule(cfg, device, no_trans=no_trans)
        self.interaction_module = InteractionModule(cfg)

    def _cfg_bool(self, name: str, default: bool) -> bool:
        mamba_cfg = getattr(self.cfg, "mamba", {})
        interaction_cfg = mamba_cfg.get("interaction", {}) if isinstance(mamba_cfg, dict) else getattr(mamba_cfg, "interaction", {})
        for container in (interaction_cfg, mamba_cfg, getattr(self.cfg, "mamba_interaction", {})):
            if isinstance(container, dict) and name in container:
                return bool(container[name])
            if hasattr(container, name):
                return bool(getattr(container, name))
        return bool(getattr(self.cfg, name, default))

    @staticmethod
    def _has_any_object(has_object) -> bool:
        if has_object is None:
            return True
        if isinstance(has_object, torch.Tensor):
            return bool(has_object.to(dtype=torch.bool).any().item())
        if isinstance(has_object, (list, tuple)):
            return bool(torch.as_tensor(has_object, dtype=torch.bool).any().item())
        return bool(has_object)

    def _build_hp_input(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hp_input = {
            "human_imu": data_dict["human_imu"],
            "v_init": data_dict["v_init"],
            "p_init": data_dict["p_init"],
        }
        if self.no_trans:
            hp_input["trans_gt"] = data_dict["trans_gt"]
        else:
            hp_input["trans_init"] = data_dict["trans_init"]
        return hp_input

    def _promote_refined_outputs(self, results: Dict, interaction_out: Dict) -> None:
        if not self.use_refined_outputs:
            return
        mapping = {
            "refined_pose": "p_pred",
            "refined_pose_part": "p_pred_part",
            "refined_root_vel": "root_vel_pred",
            "refined_trans": "root_trans_pred",
            "refined_full_pose_rotmat": "pred_full_pose_rotmat",
            "refined_full_pose_6d": "pred_full_pose_6d",
            "refined_joints_local": "pred_joints_local",
            "refined_joints_global": "pred_joints_global",
            "refined_hand_glb_pos": "pred_hand_glb_pos",
        }
        for src, dst in mapping.items():
            value = interaction_out.get(src)
            if isinstance(value, torch.Tensor):
                results[f"stage1_{dst}"] = results.get(dst)
                results[dst] = value
        if isinstance(interaction_out.get("refined_full_pose_rotmat"), torch.Tensor):
            results["R_pred_rotmat"] = interaction_out["refined_full_pose_rotmat"]
        if isinstance(interaction_out.get("refined_full_pose_6d"), torch.Tensor):
            results["R_pred_6d"] = interaction_out["refined_full_pose_6d"]

    def _merge_interaction_output(self, results: Dict, interaction_out: Dict) -> None:
        for key, value in interaction_out.items():
            if key in results:
                results[f"interaction_{key}"] = value
            else:
                results[key] = value
        self._promote_refined_outputs(results, interaction_out)

    def _maybe_add_fk_baseline(self, results: Dict, data_dict: Dict[str, torch.Tensor], hp_out: Dict, interaction_out: Dict) -> None:
        contact_prob = interaction_out.get("pred_hand_contact_prob")
        hand_pos = hp_out.get("pred_hand_glb_pos")
        obj_imu = data_dict.get("obj_imu")
        obj_trans_init = data_dict.get("obj_trans_init")
        if not (
            isinstance(contact_prob, torch.Tensor)
            and isinstance(hand_pos, torch.Tensor)
            and isinstance(obj_imu, torch.Tensor)
            and isinstance(obj_trans_init, torch.Tensor)
        ):
            return
        if contact_prob.dim() != 3 or hand_pos.dim() != 4 or obj_imu.shape[-1] < 9:
            return
        batch_size, seq_len = contact_prob.shape[:2]
        obj_rotm = rotation_6d_to_matrix(obj_imu[..., 3:9].reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
        results["pred_obj_trans_fk"] = self._fk_obj_trans_baseline_hard(contact_prob, hand_pos, obj_rotm, obj_trans_init)

    def load_pretrained_modules(self, module_paths: Dict[str, str], strict: bool = False):
        aliases = {
            "human_pose": "human_pose",
            "interaction": "interaction",
            "velocity_contact": "interaction",
            "object_trans": "interaction",
        }
        for name, path in module_paths.items():
            mapped = aliases.get(name)
            if mapped is None or not path:
                continue
            if not os.path.exists(path):
                print(f"Warning: checkpoint for {mapped} not found at {path}")
                continue
            module = getattr(self, f"{mapped}_module", None)
            if module is None:
                print(f"Warning: module '{mapped}_module' not found")
                continue
            _load_checkpoint(module, path, self.device, strict=strict)
            print(f"Loaded {mapped} from {path}")

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        use_object_data: bool = True,
        compute_fk: bool = False,
        gt_targets: Optional[Dict[str, torch.Tensor]] = None,
        detach_hp: bool = True,
        **_,
    ) -> Dict[str, torch.Tensor]:
        hp_input = self._build_hp_input(data_dict)
        hp_out = self.human_pose_module(hp_input)
        results: Dict[str, torch.Tensor] = {}
        results.update(hp_out)

        run_interaction = bool(use_object_data) and self._has_any_object(data_dict.get("has_object"))
        if run_interaction:
            interaction_hp_out = {}
            for key, value in hp_out.items():
                interaction_hp_out[key] = value.detach() if detach_hp and isinstance(value, torch.Tensor) else value
            interaction_out = self.interaction_module(data_dict, hp_out=interaction_hp_out, gt_targets=gt_targets)
            self._merge_interaction_output(results, interaction_out)
            if compute_fk:
                self._maybe_add_fk_baseline(results, data_dict, hp_out, interaction_out)

        results["has_object"] = data_dict.get("has_object")
        return results

    @torch.no_grad()
    def inference(
        self,
        data_dict: Dict[str, torch.Tensor],
        gt_targets: Optional[Dict[str, torch.Tensor]] = None,
        use_object_data: bool = True,
        compute_fk: bool = False,
        **_,
    ) -> Dict[str, torch.Tensor]:
        return self.forward(
            data_dict,
            gt_targets=gt_targets,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
            detach_hp=True,
        )


def load_model(config, device: torch.device, no_trans: bool = False, module_paths: Optional[Dict[str, str]] = None) -> IMUHOIModel:
    model = IMUHOIModel(config, device, no_trans=no_trans).to(device)
    pretrained_modules = module_paths or getattr(config, "pretrained_modules", {}) or {}
    if pretrained_modules:
        print("Loading pretrained modules:")
        model.load_pretrained_modules(pretrained_modules, strict=False)
    _flatten_lstm_parameters(model)
    model.eval()
    return model
