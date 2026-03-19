"""
Unified IMUHOI model (DiT path): HumanPose + Interaction.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix

from .human_pose import HumanPoseModule
from .interaction import InteractionModule


class IMUHOIModel(nn.Module):
    """DiT IMUHOI stack with merged interaction module."""

    @staticmethod
    def _fk_obj_trans_baseline_hard(
        pred_hand_contact_prob: torch.Tensor,  # [B, T, 3]
        pred_hand_positions: torch.Tensor,  # [B, T, 2, 3]
        obj_rotm: torch.Tensor,  # [B, T, 3, 3]
        obj_trans_init: torch.Tensor,  # [B, 3]
        contact_threshold: float = 0.5,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = pred_hand_contact_prob.shape
        device = pred_hand_contact_prob.device
        dtype = pred_hand_contact_prob.dtype

        computed_obj_trans = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
        if seq_len == 0:
            return computed_obj_trans
        computed_obj_trans[:, 0] = obj_trans_init

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
                    has_contact_another = r_contact[b, t] > 0
                elif current == 1:
                    has_contact = r_contact[b, t] > 0
                    has_contact_another = l_contact[b, t] > 0
                else:
                    has_contact = False
                    has_contact_another = False

                if new != -1:
                    current = new
                    hand0 = pred_hand_positions[b, t, current, :]
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
                elif (not has_contact) and has_contact_another:
                    current = 1 - current
                    hand0 = pred_hand_positions[b, t, current, :]
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
                    computed_obj_trans[b, t] = obj_pos
                else:
                    rt = obj_rotm[b, t]
                    direction_world = rt @ dir_local
                    obj_pos = pred_hand_positions[b, t, current, :] + direction_world * dist
                    computed_obj_trans[b, t] = obj_pos

        return computed_obj_trans

    def __init__(self, cfg, device, no_trans: bool = False):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.no_trans = no_trans
        inter_cfg = getattr(cfg, "interaction_training", {})

        def _inter_param(name: str, default):
            if isinstance(inter_cfg, dict) and name in inter_cfg:
                return inter_cfg[name]
            return getattr(cfg, name, default)

        self.hp_train_sample_steps = int(_inter_param("hp_train_sample_steps", 20))
        self.hp_train_sampler = str(_inter_param("hp_train_sampler", "ddim")).lower()
        self.hp_train_eta = float(_inter_param("hp_train_eta", 0.0))
        self._stage2_disabled_warned = False

        self.human_pose_module = HumanPoseModule(cfg, device, no_trans=no_trans)
        self.interaction_module = InteractionModule(cfg)

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

    def _run_hp(
        self,
        hp_input: Dict[str, torch.Tensor],
        gt_targets: Optional[Dict[str, torch.Tensor]],
        *,
        detach_hp: bool,
        sample_steps: Optional[int],
        sampler: Optional[str],
        eta: Optional[float],
    ) -> Dict[str, torch.Tensor]:
        del detach_hp  # Stage-2 path is disabled in this simplified Stage-1 pipeline.
        if gt_targets is None:
            raise ValueError("IMUHOIModel requires gt_targets for Stage-1 GT-warmup HumanPose inference.")

        if self.training:
            run_steps = self.hp_train_sample_steps if sample_steps is None else int(sample_steps)
            run_sampler = self.hp_train_sampler if sampler is None else sampler
            run_eta = self.hp_train_eta if eta is None else eta
        else:
            run_steps = sample_steps
            run_sampler = sampler
            run_eta = eta

        return self.human_pose_module.inference(
            hp_input,
            gt_targets=gt_targets,
            sample_steps=run_steps,
            sampler=run_sampler,
            eta=run_eta,
        )

    def _warn_stage2_disabled(self, use_object_data: bool, compute_fk: bool):
        if self._stage2_disabled_warned:
            return
        if use_object_data or compute_fk:
            print("Stage-2 interaction path is disabled in the simplified Stage-1 build.")
            self._stage2_disabled_warned = True

    def load_pretrained_modules(self, module_paths: Dict[str, str], strict: bool = True):
        from utils.utils import load_checkpoint

        alias = {
            "human_pose": "human_pose",
            "interaction": "interaction",
            # migration aliases from old two-stage modules
            "velocity_contact": "interaction",
            "object_trans": "interaction",
        }

        for name, path in module_paths.items():
            mapped = alias.get(name)
            if mapped is None:
                print(f"Warning: unknown pretrained module key '{name}', skipping")
                continue
            if not path:
                continue
            if not os.path.exists(path):
                print(f"Warning: checkpoint for {mapped} not found at {path}")
                continue

            module = getattr(self, f"{mapped}_module", None)
            if module is None:
                print(f"Warning: module '{mapped}_module' not found")
                continue

            try:
                load_checkpoint(module, path, self.device, strict=strict)
                print(f"Loaded {mapped} from {path}")
            except Exception as exc:
                print(f"Warning: failed to load '{name}' checkpoint into '{mapped}': {exc}")

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        use_object_data: bool = True,
        compute_fk: bool = False,
        gt_targets: Optional[Dict[str, torch.Tensor]] = None,
        force_inference: bool = False,
        detach_hp: bool = False,
        sample_steps: int | None = None,
        sampler: str | None = None,
        eta: float | None = None,
        **_,
    ) -> Dict[str, torch.Tensor]:
        if force_inference or (not self.training):
            return self.inference(
                data_dict,
                gt_targets=gt_targets,
                use_object_data=use_object_data,
                compute_fk=compute_fk,
                sample_steps=sample_steps,
                sampler=sampler,
                eta=eta,
            )

        self._warn_stage2_disabled(use_object_data=use_object_data, compute_fk=compute_fk)
        results: Dict[str, torch.Tensor] = {}

        hp_input = self._build_hp_input(data_dict)
        hp_out = self._run_hp(
            hp_input,
            gt_targets,
            detach_hp=detach_hp,
            sample_steps=sample_steps,
            sampler=sampler,
            eta=eta,
        )
        results.update(hp_out)

        results["has_object"] = data_dict.get("has_object")
        return results

    @torch.no_grad()
    def inference(
        self,
        data_dict: Dict[str, torch.Tensor],
        gt_targets: Optional[Dict[str, torch.Tensor]] = None,
        use_object_data: bool = True,
        compute_fk: bool = False,
        sample_steps: int | None = None,
        sampler: str | None = None,
        eta: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        if gt_targets is None:
            raise ValueError("IMUHOIModel.inference requires gt_targets for Stage-1 GT warmup.")

        self._warn_stage2_disabled(use_object_data=use_object_data, compute_fk=compute_fk)
        results: Dict[str, torch.Tensor] = {}

        hp_input = self._build_hp_input(data_dict)
        hp_out = self.human_pose_module.inference(
            hp_input,
            gt_targets=gt_targets,
            sample_steps=sample_steps,
            sampler=sampler,
            eta=eta,
        )
        results.update(hp_out)

        results["has_object"] = data_dict.get("has_object")
        return results


def load_model(
    config,
    device: torch.device,
    no_trans: bool = False,
    module_paths: Optional[Dict[str, str]] = None,
) -> IMUHOIModel:
    from utils.utils import flatten_lstm_parameters

    model = IMUHOIModel(config, device, no_trans=no_trans).to(device)

    pretrained_modules = {}
    if module_paths:
        pretrained_modules = module_paths
    else:
        staged_cfg = getattr(config, "staged_training", {})
        if staged_cfg:
            modular_cfg = staged_cfg.get("modular_training", {}) if isinstance(staged_cfg, dict) else getattr(
                staged_cfg, "modular_training", {}
            )
            if modular_cfg:
                pretrained_modules = modular_cfg.get("pretrained_modules", {}) if isinstance(modular_cfg, dict) else getattr(
                    modular_cfg, "pretrained_modules", {}
                )

    if pretrained_modules:
        print("Loading pretrained modules:")
        model.load_pretrained_modules(pretrained_modules, strict=False)

    flatten_lstm_parameters(model)
    model.eval()
    return model
