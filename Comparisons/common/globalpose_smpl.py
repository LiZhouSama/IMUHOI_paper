"""Minimal SMPL evaluator used by the GlobalPose comparison protocol."""
from __future__ import annotations

import pickle
from typing import Optional, Sequence

import numpy as np
import torch


class GlobalPoseSMPLModel:
    """Evaluate the mesh-mounted sensors used by released GlobalPose models."""

    DEFAULT_VERTEX_INDICES = (1961, 5424, 1176, 4662, 411, 3021)

    def __init__(self, model_path: str, vertex_indices=DEFAULT_VERTEX_INDICES):
        with open(model_path, "rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        self.model_path = str(model_path)
        self.vertex_indices = tuple(int(index) for index in vertex_indices)
        self.joints = torch.as_tensor(np.asarray(data["J"]).copy(), dtype=torch.float32)
        self.vertices = torch.as_tensor(
            np.asarray(data["v_template"]).copy(), dtype=torch.float32
        )
        self.weights = torch.as_tensor(
            np.asarray(data["weights"]).copy(), dtype=torch.float32
        )
        self.shapedirs = torch.as_tensor(
            np.asarray(data["shapedirs"]).copy(), dtype=torch.float32
        )
        regressor = data.get("J_regressor")
        self.joint_regressor = (
            torch.as_tensor(
                (
                    regressor.toarray()
                    if hasattr(regressor, "toarray")
                    else np.asarray(regressor)
                ).copy(),
                dtype=torch.float32,
            )
            if regressor is not None
            else None
        )
        parents = np.asarray(data["kintree_table"])[0].astype(np.int64)
        parents[0] = 0
        self.parents = tuple(int(parent) for parent in parents)
        if max(self.vertex_indices, default=-1) >= self.vertices.shape[0]:
            raise ValueError(
                f"GlobalPose vertex index exceeds model topology: {self.vertex_indices}"
            )

    def _shape_vertices(
        self, batch_size: int, betas: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if betas is None:
            return self.vertices.unsqueeze(0).expand(batch_size, -1, -1)
        betas = betas.reshape(batch_size, -1).to(
            device=self.vertices.device, dtype=self.vertices.dtype
        )
        count = min(int(betas.shape[-1]), int(self.shapedirs.shape[-1]))
        return self.vertices.unsqueeze(0) + torch.einsum(
            "nb,vcb->nvc", betas[:, :count], self.shapedirs[:, :, :count]
        )

    @torch.no_grad()
    def forward(
        self,
        local_rotations: torch.Tensor,
        trans: Optional[torch.Tensor] = None,
        betas: Optional[torch.Tensor] = None,
        ignored_indices: Optional[Sequence[int]] = None,
    ) -> dict[str, torch.Tensor]:
        """Run FK and return global rotations, joints, and selected vertices."""
        rotations = local_rotations.float()
        if rotations.shape[-3] < 24:
            eye = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
            pad = eye.view(*([1] * (rotations.ndim - 3)), 1, 3, 3).expand(
                *rotations.shape[:-3], 24 - rotations.shape[-3], 3, 3
            )
            rotations = torch.cat([rotations, pad], dim=-3)
        rotations = rotations.reshape(-1, 24, 3, 3)
        if ignored_indices:
            rotations = rotations.clone()
            eye = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
            valid = [
                int(index)
                for index in ignored_indices
                if 0 <= int(index) < 24
            ]
            if valid:
                rotations[:, valid] = eye

        batch_size = int(rotations.shape[0])
        device = rotations.device
        raw_root = self.joints[:1].to(device=device)
        joints_rest = (self.joints - self.joints[:1]).to(device=device)
        vertices = self._shape_vertices(batch_size, betas).to(device=device)
        if betas is not None:
            if self.joint_regressor is None:
                raise ValueError("SMPL model does not contain a joint regressor")
            joints_rest = torch.einsum(
                "jv,nvc->njc",
                self.joint_regressor.to(device=device),
                vertices,
            )
            joints_rest = joints_rest - joints_rest[:, :1]
            offsets = joints_rest.clone()
            for joint in range(1, 24):
                offsets[:, joint] = (
                    joints_rest[:, joint] - joints_rest[:, self.parents[joint]]
                )
        else:
            offsets = joints_rest.clone()
            for joint in range(1, 24):
                offsets[joint] = joints_rest[joint] - joints_rest[self.parents[joint]]

        if joints_rest.ndim == 3:
            vertex_reference = joints_rest[:, :1]
        else:
            vertex_reference = raw_root.view(1, 1, 3).expand(batch_size, -1, -1)
        vertices = vertices - vertex_reference

        global_rot = []
        global_pos = []
        for joint in range(24):
            if joint == 0:
                global_rot.append(rotations[:, 0])
                root_position = (
                    joints_rest[:, 0]
                    if joints_rest.ndim == 3
                    else joints_rest[0].expand(batch_size, 3)
                )
                global_pos.append(root_position)
                continue
            parent = self.parents[joint]
            parent_rot = global_rot[parent]
            global_rot.append(parent_rot.matmul(rotations[:, joint]))
            offset = (
                offsets[:, joint]
                if offsets.ndim == 3
                else offsets[joint].view(1, 3).expand(batch_size, -1)
            )
            global_pos.append(
                global_pos[parent]
                + parent_rot.matmul(offset.unsqueeze(-1)).squeeze(-1)
            )
        global_rot_tensor = torch.stack(global_rot, dim=1)
        global_pos_tensor = torch.stack(global_pos, dim=1)

        vertex_ids = torch.as_tensor(
            self.vertex_indices, device=device, dtype=torch.long
        )
        selected_vertices = vertices.index_select(1, vertex_ids)
        selected_weights = self.weights.to(device=device).index_select(0, vertex_ids)
        joint_reference = (
            joints_rest
            if joints_rest.ndim == 3
            else joints_rest.unsqueeze(0).expand(batch_size, -1, -1)
        )
        relative = selected_vertices[:, :, None, :] - joint_reference[:, None, :, :]
        rotated = torch.einsum("njab,nvjb->nvja", global_rot_tensor, relative)
        selected_world = (
            (rotated + global_pos_tensor[:, None, :, :])
            * selected_weights[None, :, :, None]
        ).sum(dim=2)
        if trans is not None:
            translation = trans.reshape(batch_size, 1, 3).to(
                device=device, dtype=selected_world.dtype
            )
            selected_world = selected_world + translation
            global_pos_tensor = global_pos_tensor + translation.to(
                dtype=global_pos_tensor.dtype
            )
        return {
            "global_rot": global_rot_tensor,
            "joints": global_pos_tensor,
            "vertices": selected_world,
        }
