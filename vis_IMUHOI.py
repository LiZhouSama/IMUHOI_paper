import torch
import os
import sys
import numpy as np
import random
import argparse
import yaml
import re
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.viewer import Viewer
from aitviewer.scene.camera import Camera
from moderngl_window.context.base import KeyModifiers
import pytorch3d.transforms as transforms
import trimesh
from easydict import EasyDict as edict

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_IMUHOI import IMUDataset
from process.preprocess import load_object_geometry
from model import IMUHOIModel, load_model
from utils.utils import (
    load_config,
    load_smpl_model,
    build_model_input_dict,
)
from configs import (
    FRAME_RATE,
    _SENSOR_NAMES,
    _SENSOR_VEL_NAMES,
    _REDUCED_POSE_NAMES,
    _REDUCED_INDICES,
    _IGNORED_INDICES,
    _SENSOR_ROT_INDICES,
    _SENSOR_POS_INDICES,
    _VEL_SELECTION_INDICES,
)

import imgui
try:
    from aitviewer.renderables.lines import Lines
except Exception as exc:
    raise ImportError(
        "aitviewer.renderables.lines.Lines is required for trajectory visualization. "
        "Please install an aitviewer version that provides this renderable."
    ) from exc

R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

OBJ_TRAJ_RADIUS = 0.006
HAND_TRAJ_RADIUS = 0.004
HAND_DASH_STRIDE = 2

OBJ_TRAJ_COLORS = {
    "gt": (0.1, 0.8, 0.3, 1),
    "pred": (0.9, 0.2, 0.2, 1),
    "pred_imu": (0.2, 0.2, 0.9, 1),
    "fk": (1.0, 1.0, 0.0, 1),
}

HAND_TRAJ_COLORS = {
    "gt_l_contact": (1.0, 0.0, 0.0, 1),
    "gt_l_non_contact": (1.0, 0.0, 0.0, 1),
    "gt_r_contact": (0.0, 0.0, 1.0, 1),
    "gt_r_non_contact": (0.0, 0.0, 1.0, 1),
    "pred_l_contact": (1.0, 0.5, 0.0, 1),
    "pred_l_non_contact": (1.0, 0.5, 0.0, 1),
    "pred_r_contact": (0.0, 0.6, 1.0, 1),
    "pred_r_non_contact": (0.0, 0.6, 1.0, 1),
}

GT_HUMAN_COLOR = (118/255, 147/255, 248/255, 0.9)
GT_OBJECT_COLOR = (186/255, 161/255, 246/255, 0.9)


def compute_virtual_bone_info(wrist_pos, obj_trans, obj_rot_mat):
    """Compute virtual bone length and direction."""
    v_HO_world = obj_trans - wrist_pos
    bone_length = torch.norm(v_HO_world, dim=1)
    v_HO_world_unit = v_HO_world / (bone_length.unsqueeze(-1) + 1e-8)
    obj_rot_inv = obj_rot_mat.transpose(-1, -2)
    obj_direction = torch.bmm(obj_rot_inv, v_HO_world_unit.unsqueeze(-1)).squeeze(-1)
    return bone_length, obj_direction


def _to_np_yup(points_torch, device):
    if points_torch is None or points_torch.dim() != 2 or points_torch.shape[0] < 2:
        return None
    points_yup = torch.matmul(points_torch, R_yup.T.to(device))
    return points_yup.detach().cpu().numpy()


def _segments_from_points(points_torch):
    if points_torch is None or points_torch.dim() != 2 or points_torch.shape[0] < 2:
        return None
    return torch.stack([points_torch[:-1], points_torch[1:]], dim=1)


def _segments_to_line_points(segments_torch):
    if segments_torch is None or segments_torch.numel() == 0:
        return None
    return segments_torch.reshape(-1, 3)


def _split_contact_segments(points_torch, contact_bool, dash_stride=2):
    if points_torch is None or contact_bool is None:
        return None, None
    if points_torch.dim() != 2 or points_torch.shape[0] < 2:
        return None, None

    contacts = contact_bool
    if not isinstance(contacts, torch.Tensor):
        contacts = torch.as_tensor(contacts, device=points_torch.device)
    else:
        contacts = contacts.to(points_torch.device)
    if contacts.dtype != torch.bool:
        contacts = contacts > 0.5

    valid_len = min(points_torch.shape[0], contacts.shape[0])
    if valid_len < 2:
        return None, None

    pts = points_torch[:valid_len]
    contacts = contacts[:valid_len]
    segments = _segments_from_points(pts)
    if segments is None:
        return None, None

    contact_mask = contacts[:-1] & contacts[1:]
    contact_segments = segments[contact_mask]
    non_contact_segments = segments[~contact_mask]
    if non_contact_segments.numel() > 0 and dash_stride > 1:
        non_contact_segments = non_contact_segments[::dash_stride]

    return _segments_to_line_points(contact_segments), _segments_to_line_points(non_contact_segments)


def _add_line_node_if_nonempty(viewer, points_torch, device, name, color, r_base, mode):
    if points_torch is None:
        return
    if points_torch.dim() != 2 or points_torch.shape[0] < 2:
        return

    draw_points = points_torch
    if mode == "lines" and draw_points.shape[0] % 2 != 0:
        draw_points = draw_points[:-1]
    if draw_points.shape[0] < 2:
        return

    points_np = _to_np_yup(draw_points, device)
    if points_np is None:
        return

    line_node = Lines(
        lines=points_np,
        mode=mode,
        r_base=r_base,
        color=color,
        cast_shadow=False,
        name=name,
        gui_affine=False,
        is_selectable=False,
    )
    viewer.scene.add(line_node)


def _normalize_overlay_frames(frame_ids, total_frames):
    if frame_ids is None:
        return None
    if total_frames <= 0:
        raise SystemExit(f"Invalid sequence length: {total_frames}")

    unique_ids = []
    seen = set()
    invalid = []
    for raw_idx in frame_ids:
        idx = int(raw_idx)
        if idx in seen:
            continue
        seen.add(idx)
        if idx < 0 or idx >= total_frames:
            invalid.append(idx)
        else:
            unique_ids.append(idx)

    if invalid:
        raise SystemExit(
            f"Invalid --overlay_frames: {invalid}. "
            f"Valid frame range is [0, {total_frames - 1}]."
        )
    if not unique_ids:
        raise SystemExit("No valid frame index found in --overlay_frames.")
    return unique_ids


def _build_alpha_gradient(n, alpha_min=0.30, alpha_max=0.95):
    if n <= 0:
        return []
    if n == 1:
        return [float(alpha_max)]
    return np.linspace(alpha_min, alpha_max, n, dtype=np.float32).tolist()


def _add_overlay_meshes(viewer, verts_seq, faces_np, frame_ids, base_name, base_rgb, alpha_values, device):
    if verts_seq is None or faces_np is None:
        return
    max_frames = verts_seq.shape[0]
    rgb = (float(base_rgb[0]), float(base_rgb[1]), float(base_rgb[2]))
    for idx, frame_id in enumerate(frame_ids):
        if frame_id < 0 or frame_id >= max_frames:
            continue
        frame_verts = verts_seq[frame_id:frame_id + 1]
        frame_verts_yup = torch.matmul(frame_verts, R_yup.T.to(device))
        alpha = float(alpha_values[idx]) if idx < len(alpha_values) else 0.95
        mesh = Meshes(
            frame_verts_yup.detach().cpu().numpy(), faces_np,
            name=f"{base_name}-F{frame_id}",
            color=(rgb[0], rgb[1], rgb[2], alpha),
            gui_affine=False, is_selectable=False
        )
        mesh.material.ambient = 0.26
        mesh.material.diffuse = 0.41
        viewer.scene.add(mesh)


def visualize_batch_data(viewer, batch, model, smpl_model, device, obj_geo_root, 
                         show_objects=True, vis_gt_only=False, show_foot_contact=False, 
                         show_obj_traj=False, show_hand_traj=False, use_fk=False, compare_3=False, 
                         pred_offset_np=None, no_trans=False, overlay_frames=None):
    """Visualize one batch sequence."""
    try:
        nodes_to_remove = [
            node for node in viewer.scene.collect_nodes()
            if hasattr(node, "name") and node.name is not None
            and (node.name.startswith("GT-") or node.name.startswith("Pred-") or node.name.startswith("FK-")
                 or "Contact" in node.name or "Indicator" in node.name or "Traj" in node.name)
        ]
        for node in nodes_to_remove:
            try:
                viewer.scene.remove(node)
            except Exception:
                pass
    except Exception:
        pass

    with torch.no_grad():
        config = getattr(viewer, "config", edict({}))
        
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(device)
            else:
                batch_device[key] = value

        human_imu_batch = batch_device.get("human_imu")
        if human_imu_batch is None:
            print("Error: Batch missing 'human_imu'.")
            return
        if human_imu_batch.dim() == 3:
            human_imu_batch = human_imu_batch.unsqueeze(0)
        batch_size, T = human_imu_batch.shape[:2]
        bs = 0
        human_imu_seq = human_imu_batch[bs]
        overlay_frame_ids = _normalize_overlay_frames(overlay_frames, T)
        overlay_mode = overlay_frame_ids is not None
        overlay_alpha_values = _build_alpha_gradient(len(overlay_frame_ids)) if overlay_mode else None
        if overlay_mode:
            print(f"Overlay mesh frames: {overlay_frame_ids}")

        pose_batch = batch_device.get("pose")
        trans_batch = batch_device.get("trans")
        obj_trans_batch = batch_device.get("obj_trans")
        obj_rot_batch = batch_device.get("obj_rot")
        obj_scale_batch = batch_device.get("obj_scale")
        lhand_contact_batch = batch_device.get("lhand_contact")
        rhand_contact_batch = batch_device.get("rhand_contact")
        obj_contact_batch = batch_device.get("obj_contact")

        def _get_bool_sequence(tensor):
            if tensor is None:
                return None
            seq = tensor[bs]
            if seq.dtype != torch.bool:
                seq = seq > 0.5
            return seq

        lhand_contact_seq = _get_bool_sequence(lhand_contact_batch)
        rhand_contact_seq = _get_bool_sequence(rhand_contact_batch)
        obj_contact_seq = _get_bool_sequence(obj_contact_batch)

        has_object_value = batch_device.get("has_object")
        has_object_bool = False
        if has_object_value is not None:
            if isinstance(has_object_value, torch.Tensor):
                has_object_bool = bool(has_object_value[bs].item() if has_object_value.dim() > 0 else has_object_value.item())
            else:
                has_object_bool = bool(has_object_value)
        elif obj_trans_batch is not None:
            has_object_bool = True

        obj_name = "object"
        if "obj_name" in batch_device:
            obj_name_raw = batch_device["obj_name"]
            if isinstance(obj_name_raw, (list, tuple)):
                obj_name_candidate = obj_name_raw[bs]
            else:
                obj_name_candidate = obj_name_raw
            if isinstance(obj_name_candidate, bytes):
                obj_name_candidate = obj_name_candidate.decode("utf-8")
            if obj_name_candidate:
                obj_name = str(obj_name_candidate)

        faces_attr = getattr(smpl_model, "f", None)
        if isinstance(faces_attr, torch.Tensor):
            faces_gt_np = faces_attr.detach().cpu().numpy()
        elif faces_attr is not None:
            faces_gt_np = faces_attr
        else:
            faces_gt_np = smpl_model.faces_tensor.detach().cpu().numpy()

        verts_gt_seq = None
        Jtr_gt_seq = None
        if pose_batch is not None and trans_batch is not None:
            try:
                gt_pose_seq = pose_batch[bs]
                gt_trans_seq = trans_batch[bs]
                if gt_pose_seq.dim() == 2 and gt_trans_seq.dim() == 2:
                    root_orient = gt_pose_seq[:, :3]
                    pose_body = gt_pose_seq[:, 3:]
                    if pose_body.shape[1] < 63:
                        pose_body_padded = torch.zeros(T, 63, device=device, dtype=pose_body.dtype)
                        pose_body_padded[:, :pose_body.shape[1]] = pose_body
                        pose_body = pose_body_padded
                    elif pose_body.shape[1] > 63:
                        pose_body = pose_body[:, :63]
                    body_pose_gt = smpl_model(
                        pose_body=pose_body,
                        root_orient=root_orient,
                        trans=gt_trans_seq
                    )
                    verts_gt_seq = body_pose_gt.v
                    Jtr_gt_seq = body_pose_gt.Jtr
            except Exception as exc:
                print(f"Failed to build GT SMPL mesh: {exc}")

        pred_dict = None
        verts_pred_seq = None
        Jtr_pred_seq = None
        pred_obj_trans_seq = None
        pred_obj_trans_fk_seq = None
        pred_obj_trans_imu_seq = None
        pred_hand_contact_prob_seq = None
        pred_lhand_contact_seq = None
        pred_rhand_contact_seq = None
        model_input = None

        if not vis_gt_only:
            try:
                model_input = build_model_input_dict(batch_device, config, device, add_noise=False)
                compute_fk_flag = bool(use_fk or compare_3)
                model_arch = str(getattr(config, "model_arch", "rnn")).lower()
                if model_arch == "dit":
                    pred_dict = model.inference(model_input, use_object_data=True, compute_fk=compute_fk_flag)
                else:
                    pred_dict = model(model_input, use_object_data=True, compute_fk=compute_fk_flag)
            except Exception as exc:
                print(f"Model inference failed: {exc}")
                pred_dict = None

        if pred_dict is not None:
            p_pred_seq = pred_dict.get("p_pred")
            if p_pred_seq is not None:
                p_pred_seq = p_pred_seq[bs].to(device)
            
            if no_trans:
                pred_root_trans_seq = trans_batch[bs] if trans_batch is not None else None
            else:
                pred_root_trans_all = pred_dict.get("root_trans_pred")
                pred_root_trans_seq = pred_root_trans_all[bs].to(device) if pred_root_trans_all is not None else None
            
            if "pred_obj_trans" in pred_dict:
                pred_obj_trans_seq = pred_dict["pred_obj_trans"][bs].to(device)
            if compare_3 and "pred_obj_trans_fk" in pred_dict:
                pred_obj_trans_fk_seq = pred_dict["pred_obj_trans_fk"][bs].to(device)

            pred_hand_contact_prob_all = pred_dict.get("pred_hand_contact_prob")
            if pred_hand_contact_prob_all is not None:
                pred_hand_contact_prob_seq = pred_hand_contact_prob_all[bs].to(device)
                if pred_hand_contact_prob_seq.shape[-1] >= 2:
                    pred_lhand_contact_seq = pred_hand_contact_prob_seq[:, 0] > 0.5
                    pred_rhand_contact_seq = pred_hand_contact_prob_seq[:, 1] > 0.5
            
            if compare_3 and "pred_obj_vel" in pred_dict and model_input is not None:
                try:
                    pred_obj_vel_seq = pred_dict["pred_obj_vel"][bs].to(device)
                    obj_trans_init = model_input["obj_trans_init"][bs].to(device)
                    delta = pred_obj_vel_seq * (1.0 / FRAME_RATE)
                    disp = torch.zeros_like(delta)
                    if T > 1:
                        cumulative = torch.cumsum(delta, dim=0)
                        disp[1:] = cumulative[:-1]
                    pred_obj_trans_imu_seq = obj_trans_init.unsqueeze(0) + disp
                except Exception:
                    pass
            
            if p_pred_seq is not None and pred_root_trans_seq is not None and model.human_pose_module is not None:
                try:
                    reduced_pose = p_pred_seq.view(T, len(_REDUCED_POSE_NAMES), 6)
                    orientation_6d = human_imu_batch[bs, :, :, -6:]
                    orientation_mat = transforms.rotation_6d_to_matrix(
                        orientation_6d.reshape(-1, 6)
                    ).reshape(T, human_imu_batch.shape[2], 3, 3)
                    orientation_subset = orientation_mat[:, :len(_SENSOR_ROT_INDICES), :, :]
                    human_module = model.human_pose_module
                    full_glb = human_module._reduced_glb_6d_to_full_glb_mat(
                        reduced_pose,
                        orientation_subset.reshape(T, len(_SENSOR_ROT_INDICES), 3, 3)
                    )
                    parents = human_module.smpl_parents.tolist()
                    local_rot = human_module._global2local(full_glb, parents)
                    pose_axis = transforms.matrix_to_axis_angle(
                        local_rot.reshape(T * full_glb.shape[1], 3, 3)
                    ).reshape(T, full_glb.shape[1], 3)
                    root_axis = pose_axis[:, 0, :]
                    pose_body_axis = pose_axis[:, 1:22, :].reshape(T, -1)
                    smpl_pred = smpl_model(
                        pose_body=pose_body_axis,
                        root_orient=root_axis,
                        trans=pred_root_trans_seq
                    )
                    verts_pred_seq = smpl_pred.v
                    Jtr_pred_seq = smpl_pred.Jtr
                except Exception as exc:
                    print(f"Predicted SMPL reconstruction failed: {exc}")

        if pred_offset_np is not None:
            pred_offset = torch.tensor(pred_offset_np, device=device, dtype=torch.float32)
        else:
            pred_offset = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)

        gt_obj_verts_seq = None
        pred_obj_verts_seq_mesh = None
        pred_obj_verts_seq_fk = None
        pred_obj_verts_seq_imu = None
        obj_faces_np = None
        gt_obj_trans_seq = None
        gt_obj_rot_mat_seq = None

        if has_object_bool and obj_trans_batch is not None:
            gt_obj_trans_seq = obj_trans_batch[bs]

        if show_objects and has_object_bool and obj_trans_batch is not None and obj_rot_batch is not None:
            gt_obj_rot_6d_seq = obj_rot_batch[bs]
            gt_obj_rot_mat_seq = transforms.rotation_6d_to_matrix(gt_obj_rot_6d_seq)
            gt_obj_scale_seq = obj_scale_batch[bs] if obj_scale_batch is not None else None
            try:
                gt_obj_verts_seq, obj_faces_np = load_object_geometry(
                    obj_name, gt_obj_rot_mat_seq, gt_obj_trans_seq, gt_obj_scale_seq,
                    device=device, obj_geo_root=obj_geo_root
                )
            except Exception as exc:
                print(f"Failed to load GT object geometry: {exc}")
            
            if not vis_gt_only and pred_obj_trans_seq is not None and obj_faces_np is not None:
                try:
                    pred_obj_verts_seq_mesh, _ = load_object_geometry(
                        obj_name, gt_obj_rot_mat_seq, pred_obj_trans_seq, gt_obj_scale_seq,
                        device=device, obj_geo_root=obj_geo_root
                    )
                except Exception:
                    pass
            
            if not vis_gt_only and compare_3 and pred_obj_trans_fk_seq is not None and obj_faces_np is not None:
                try:
                    pred_obj_verts_seq_fk, _ = load_object_geometry(
                        obj_name, gt_obj_rot_mat_seq, pred_obj_trans_fk_seq, gt_obj_scale_seq,
                        device=device, obj_geo_root=obj_geo_root
                    )
                except Exception:
                    pass
            
            if not vis_gt_only and compare_3 and pred_obj_trans_imu_seq is not None and obj_faces_np is not None:
                try:
                    pred_obj_verts_seq_imu, _ = load_object_geometry(
                        obj_name, gt_obj_rot_mat_seq, pred_obj_trans_imu_seq, gt_obj_scale_seq,
                        device=device, obj_geo_root=obj_geo_root
                    )
                except Exception:
                    pass

        if verts_gt_seq is not None:
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, verts_gt_seq, faces_gt_np, overlay_frame_ids,
                    "GT-Human", GT_HUMAN_COLOR, overlay_alpha_values, device
                )
            else:
                verts_gt_yup = torch.matmul(verts_gt_seq, R_yup.T.to(device))
                gt_human_mesh = Meshes(
                    verts_gt_yup.detach().cpu().numpy(), faces_gt_np,
                    name="GT-Human", color=GT_HUMAN_COLOR,
                    gui_affine=False, is_selectable=False
                )
                gt_human_mesh.material.ambient = 0.26
                gt_human_mesh.material.diffuse = 0.41
                viewer.scene.add(gt_human_mesh)

        if verts_pred_seq is not None and not vis_gt_only:
            verts_pred_shifted = verts_pred_seq + pred_offset
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, verts_pred_shifted, faces_gt_np, overlay_frame_ids,
                    "Pred-Human", (0.9, 0.2, 0.2), overlay_alpha_values, device
                )
            else:
                verts_pred_yup = torch.matmul(verts_pred_shifted, R_yup.T.to(device))
                pred_human_mesh = Meshes(
                    verts_pred_yup.detach().cpu().numpy(), faces_gt_np,
                    name="Pred-Human", color=(0.9, 0.2, 0.2, 0.8),
                    gui_affine=False, is_selectable=False
                )
                viewer.scene.add(pred_human_mesh)

        if gt_obj_verts_seq is not None and obj_faces_np is not None:
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, gt_obj_verts_seq, obj_faces_np, overlay_frame_ids,
                    f"GT-{obj_name}", GT_OBJECT_COLOR, overlay_alpha_values, device
                )
            else:
                gt_obj_verts_yup = torch.matmul(gt_obj_verts_seq, R_yup.T.to(device))
                gt_obj_mesh = Meshes(
                    gt_obj_verts_yup.detach().cpu().numpy(), obj_faces_np,
                    name=f"GT-{obj_name}", color=GT_OBJECT_COLOR,
                    gui_affine=False, is_selectable=False
                )
                gt_obj_mesh.material.ambient = 0.24
                viewer.scene.add(gt_obj_mesh)

        if pred_obj_verts_seq_mesh is not None and obj_faces_np is not None and not vis_gt_only:
            pred_obj_verts_shifted = pred_obj_verts_seq_mesh + pred_offset
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, pred_obj_verts_shifted, obj_faces_np, overlay_frame_ids,
                    f"Pred-{obj_name}", (0.9, 0.2, 0.2), overlay_alpha_values, device
                )
            else:
                pred_obj_verts_yup = torch.matmul(pred_obj_verts_shifted, R_yup.T.to(device))
                pred_obj_mesh = Meshes(
                    pred_obj_verts_yup.detach().cpu().numpy(), obj_faces_np,
                    name=f"Pred-{obj_name}", color=(0.9, 0.2, 0.2, 0.8),
                    gui_affine=False, is_selectable=False
                )
                viewer.scene.add(pred_obj_mesh)

        if compare_3 and pred_obj_verts_seq_imu is not None and obj_faces_np is not None and not vis_gt_only:
            obj_verts_imu_shifted = pred_obj_verts_seq_imu + pred_offset
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, obj_verts_imu_shifted, obj_faces_np, overlay_frame_ids,
                    f"Pred-IMU-{obj_name}", (0.2, 0.2, 0.9), overlay_alpha_values, device
                )
            else:
                obj_verts_imu_yup = torch.matmul(obj_verts_imu_shifted, R_yup.T.to(device))
                imu_mesh = Meshes(
                    obj_verts_imu_yup.detach().cpu().numpy(), obj_faces_np,
                    name=f"Pred-IMU-{obj_name}", color=(0.2, 0.2, 0.9, 0.8),
                    gui_affine=False, is_selectable=False
                )
                viewer.scene.add(imu_mesh)

        if compare_3 and pred_obj_verts_seq_fk is not None and obj_faces_np is not None and not vis_gt_only:
            obj_verts_fk_shifted = pred_obj_verts_seq_fk + pred_offset
            if overlay_mode:
                _add_overlay_meshes(
                    viewer, obj_verts_fk_shifted, obj_faces_np, overlay_frame_ids,
                    f"FK-{obj_name}", (1.0, 1.0, 0.0), overlay_alpha_values, device
                )
            else:
                obj_verts_fk_yup = torch.matmul(obj_verts_fk_shifted, R_yup.T.to(device))
                fk_mesh = Meshes(
                    obj_verts_fk_yup.detach().cpu().numpy(), obj_faces_np,
                    name=f"FK-{obj_name}", color=(1.0, 1.0, 0.0, 0.8),
                    gui_affine=False, is_selectable=False
                )
                viewer.scene.add(fk_mesh)

        if show_obj_traj:
            if gt_obj_trans_seq is not None:
                _add_line_node_if_nonempty(
                    viewer, gt_obj_trans_seq, device, f"GT-{obj_name}-ObjTraj",
                    OBJ_TRAJ_COLORS["gt"], OBJ_TRAJ_RADIUS, "line_strip"
                )
            if not vis_gt_only and pred_obj_trans_seq is not None:
                _add_line_node_if_nonempty(
                    viewer, pred_obj_trans_seq + pred_offset, device, f"Pred-{obj_name}-ObjTraj",
                    OBJ_TRAJ_COLORS["pred"], OBJ_TRAJ_RADIUS, "line_strip"
                )
            if not vis_gt_only and compare_3 and pred_obj_trans_imu_seq is not None:
                _add_line_node_if_nonempty(
                    viewer, pred_obj_trans_imu_seq + pred_offset, device, f"Pred-IMU-{obj_name}-ObjTraj",
                    OBJ_TRAJ_COLORS["pred_imu"], OBJ_TRAJ_RADIUS, "line_strip"
                )
            if not vis_gt_only and compare_3 and pred_obj_trans_fk_seq is not None:
                _add_line_node_if_nonempty(
                    viewer, pred_obj_trans_fk_seq + pred_offset, device, f"FK-{obj_name}-ObjTraj",
                    OBJ_TRAJ_COLORS["fk"], OBJ_TRAJ_RADIUS, "line_strip"
                )

        if show_hand_traj:
            lhand_idx, rhand_idx = 20, 21

            if Jtr_gt_seq is not None:
                if lhand_contact_seq is not None:
                    gt_l_contact_lines, gt_l_non_contact_lines = _split_contact_segments(
                        Jtr_gt_seq[:, lhand_idx], lhand_contact_seq, dash_stride=HAND_DASH_STRIDE
                    )
                    _add_line_node_if_nonempty(
                        viewer, gt_l_contact_lines, device, "GT-LHandTraj-Contact",
                        HAND_TRAJ_COLORS["gt_l_contact"], HAND_TRAJ_RADIUS, "lines"
                    )
                    _add_line_node_if_nonempty(
                        viewer, gt_l_non_contact_lines, device, "GT-LHandTraj-NonContact",
                        HAND_TRAJ_COLORS["gt_l_non_contact"], HAND_TRAJ_RADIUS, "lines"
                    )

                if rhand_contact_seq is not None:
                    gt_r_contact_lines, gt_r_non_contact_lines = _split_contact_segments(
                        Jtr_gt_seq[:, rhand_idx], rhand_contact_seq, dash_stride=HAND_DASH_STRIDE
                    )
                    _add_line_node_if_nonempty(
                        viewer, gt_r_contact_lines, device, "GT-RHandTraj-Contact",
                        HAND_TRAJ_COLORS["gt_r_contact"], HAND_TRAJ_RADIUS, "lines"
                    )
                    _add_line_node_if_nonempty(
                        viewer, gt_r_non_contact_lines, device, "GT-RHandTraj-NonContact",
                        HAND_TRAJ_COLORS["gt_r_non_contact"], HAND_TRAJ_RADIUS, "lines"
                    )

            if not vis_gt_only and Jtr_pred_seq is not None:
                if pred_lhand_contact_seq is not None:
                    pred_l_contact_lines, pred_l_non_contact_lines = _split_contact_segments(
                        Jtr_pred_seq[:, lhand_idx] + pred_offset, pred_lhand_contact_seq, dash_stride=HAND_DASH_STRIDE
                    )
                    _add_line_node_if_nonempty(
                        viewer, pred_l_contact_lines, device, "Pred-LHandTraj-Contact",
                        HAND_TRAJ_COLORS["pred_l_contact"], HAND_TRAJ_RADIUS, "lines"
                    )
                    _add_line_node_if_nonempty(
                        viewer, pred_l_non_contact_lines, device, "Pred-LHandTraj-NonContact",
                        HAND_TRAJ_COLORS["pred_l_non_contact"], HAND_TRAJ_RADIUS, "lines"
                    )

                if pred_rhand_contact_seq is not None:
                    pred_r_contact_lines, pred_r_non_contact_lines = _split_contact_segments(
                        Jtr_pred_seq[:, rhand_idx] + pred_offset, pred_rhand_contact_seq, dash_stride=HAND_DASH_STRIDE
                    )
                    _add_line_node_if_nonempty(
                        viewer, pred_r_contact_lines, device, "Pred-RHandTraj-Contact",
                        HAND_TRAJ_COLORS["pred_r_contact"], HAND_TRAJ_RADIUS, "lines"
                    )
                    _add_line_node_if_nonempty(
                        viewer, pred_r_non_contact_lines, device, "Pred-RHandTraj-NonContact",
                        HAND_TRAJ_COLORS["pred_r_non_contact"], HAND_TRAJ_RADIUS, "lines"
                    )

        viewer.virtual_bone_info = {'has_data': False}


class InteractiveViewer(Viewer):
    def __init__(self, data_list, model, smpl_model, config, device, obj_geo_root, 
                 show_objects=True, vis_gt_only=False, show_foot_contact=False,
                 show_obj_traj=False, show_hand_traj=False, use_fk=False, compare_3=False, 
                 pred_offset=None, no_trans=False, overlay_frames=None, **kwargs):
        super().__init__(**kwargs)
        self.data_list = data_list
        self.current_index = 0
        self.model = model
        self.smpl_model = smpl_model
        self.config = config
        self.device = device
        self.show_objects = show_objects
        self.vis_gt_only = vis_gt_only
        self.show_obj_traj = show_obj_traj
        self.show_hand_traj = show_hand_traj
        self.show_foot_contact = show_foot_contact
        self.obj_geo_root = obj_geo_root
        self.use_fk = use_fk
        self.compare_3 = compare_3
        self.pred_offset = pred_offset
        self.no_trans = no_trans
        self.overlay_frames = overlay_frames
        self.virtual_bone_info = {'has_data': False}
        
        self.visualize_current_sequence()

        light_front = self.scene.lights[1]
        light_front.strength = 1.32 
        light_front.position = (15, 10, 15)
        floor = self.scene.floor
        floor.material.diffuse = 0.1
        floor.tiling = False
        floor.material.color = (132/255, 150/255, 183/255, 1.0)
        floor.material.ambient = 0.30
        self.scene.background_color = (0.5, 0.5, 0.5, 1.0)

    def visualize_current_sequence(self):
        if not self.data_list:
            print("Error: Data list is empty.")
            return
        if 0 <= self.current_index < len(self.data_list):
            entry = self.data_list[self.current_index]
            batch = entry["batch"] if isinstance(entry, dict) and "batch" in entry else entry
            mode_str = " (GT only)" if self.vis_gt_only else " (GT+Pred)"
            seq_file_name = ""
            if isinstance(entry, dict):
                seq_file_name = entry.get("seq_file_name") or os.path.basename(entry.get("seq_file_path", ""))
            if seq_file_name:
                print(f"Visualizing sequence: {seq_file_name}{mode_str}")
            else:
                print(f"Visualizing sequence index: {self.current_index}{mode_str}")
            try:
                visualize_batch_data(
                    self, batch, self.model, self.smpl_model, self.device,
                    self.obj_geo_root, self.show_objects, self.vis_gt_only,
                    self.show_foot_contact, self.show_obj_traj, self.show_hand_traj,
                    self.use_fk, self.compare_3, self.pred_offset, self.no_trans, self.overlay_frames,
                )
                title_base = f"Sequence: {seq_file_name}" if seq_file_name else f"Sequence Index: {self.current_index}/{len(self.data_list)-1}"
                self.title = f"{title_base}{mode_str} (q/e:闁?, Ctrl+q/e:闁?0, Alt+q/e:闁?0)"
            except Exception as e:
                print(f"Error visualizing sequence {self.current_index}: {e}")
                import traceback
                traceback.print_exc()

    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        
        io = imgui.get_io()
        if self.render_gui and (io.want_capture_keyboard or io.want_text_input):
            return
        
        is_press = action == self.wnd.keys.ACTION_PRESS
        
        if is_press:
            ctrl_pressed = modifiers.ctrl
            alt_pressed = modifiers.alt
            
            if key == self.wnd.keys.Q:
                step = 50 if alt_pressed else (10 if ctrl_pressed else 1)
                new_index = max(0, self.current_index - step)
                if new_index != self.current_index:
                    self.current_index = new_index
                    self.visualize_current_sequence()
                    self.scene.current_frame_id = 0
            elif key == self.wnd.keys.E:
                step = 50 if alt_pressed else (10 if ctrl_pressed else 1)
                new_index = min(len(self.data_list) - 1, self.current_index + step)
                if new_index != self.current_index:
                    self.current_index = new_index
                    self.visualize_current_sequence()
                    self.scene.current_frame_id = 0


def main():
    parser = argparse.ArgumentParser(description='Interactive IMUHOI Visualization Tool')
    parser.add_argument('--config', type=str, default='configs/IMUHOI_train.yaml', help='Path to config file')
    parser.add_argument('--smpl_model_path', type=str, default=None, help='Path to SMPL model file')
    parser.add_argument('--test_data_dir', type=str, default='process/processed_split_data_OMOMO/debug', help='Test data directory or a single .pt sequence file')
    parser.add_argument('--obj_geo_root', type=str, default='datasets/OMOMO/captured_objects', help='Root directory of object geometry files')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--no_objects', action='store_true', help='Disable object mesh rendering')
    parser.add_argument('--vis_gt_only', action='store_true', help='Render GT only')
    parser.add_argument('--show_foot_contact', action='store_true', help='Show foot contact')
    parser.add_argument('--show_obj_traj', action='store_true', help='Show object trajectory')
    parser.add_argument('--show_hand_traj', action='store_true', help='Show hand contact trajectory')
    parser.add_argument('--use_fk', action='store_true', help='Enable FK branch')
    parser.add_argument('--compare_3', action='store_true', help='Compare three object branches')
    parser.add_argument('--limit_sequences', type=int, default=None, help='Limit number of loaded sequences')
    parser.add_argument('--pred_offset', type=float, nargs=3, default=[3.0, 0.0, 0.0], help='Prediction translation offset')
    parser.add_argument('--overlay_frames', type=int, nargs='+', default=None, help='Overlay selected 0-based frames in one scene')
    parser.add_argument('--no_trans', action='store_true', help='Enable noTrans mode')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Mode: {'noTrans' if args.no_trans else 'Normal'}")

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    module_paths = None
    if hasattr(config, "pretrained_modules") and config.pretrained_modules:
        module_paths = dict(config.pretrained_modules)
    
    if args.smpl_model_path:
        config.body_model_path = args.smpl_model_path
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    config.device = str(device)

    smpl_model_path = config.get('body_model_path', 'datasets/smpl_models/smplh/neutral/model.npz')
    smpl_model = load_smpl_model(smpl_model_path, device)

    model = load_model(config, device, no_trans=args.no_trans, module_paths=module_paths)

    test_data_input = args.test_data_dir
    if not test_data_input or not os.path.exists(test_data_input):
        print(f"Error: Test dataset path not found: {test_data_input}")
        return

    test_window_size = config.test.get('window', config.train.get('window', 60))
    dataset_debug = config.get('debug', False)

    if os.path.isdir(test_data_input):
        print(f"Loading test dataset from directory: {test_data_input}")
        print("Dataset mode: directory")
        test_dataset = IMUDataset(
            data_dir=test_data_input,
            window_size=test_window_size,
            debug=dataset_debug,
            full_sequence=True
        )
    elif os.path.isfile(test_data_input) and test_data_input.lower().endswith(".pt"):
        target_pt = os.path.normcase(os.path.normpath(os.path.abspath(test_data_input)))
        parent_dir = os.path.dirname(target_pt)
        print(f"Loading test dataset from single file: {target_pt}")
        print("Dataset mode: single-pt")
        test_dataset = IMUDataset(
            data_dir=parent_dir,
            window_size=test_window_size,
            debug=dataset_debug,
            full_sequence=True,
            sequence_paths=[target_pt],
        )
        if len(test_dataset.sequence_info) != 1:
            try:
                pt_data = torch.load(target_pt, map_location="cpu")
            except Exception as exc:
                raise SystemExit(f"Failed to load pt file: {target_pt}. Error: {exc}")
            required_key = "rotation_local_full_gt_list"
            if not isinstance(pt_data, dict) or required_key not in pt_data:
                raise SystemExit(
                    f"Unsupported pt format for --test_data_dir: {target_pt}. "
                    f"Missing required key '{required_key}'."
                )
            raise SystemExit(
                f"Failed to build single-sequence dataset from file: {target_pt}. "
                f"Valid sequence count after filtering: {len(test_dataset.sequence_info)}"
            )
    else:
        raise SystemExit(f"--test_data_dir must be a directory or a .pt file: {test_data_input}")

    def _natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    try:
        test_dataset.sequence_info.sort(
            key=lambda info: _natural_key(os.path.basename(info.get('file_path', '')))
        )
    except Exception:
        pass

    if len(test_dataset) == 0:
        print("Error: Test dataset is empty.")
        return

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=True, drop_last=False
    )

    print(f"Loading data into memory (limit={args.limit_sequences})...")
    data_list = []
    for i, batch in enumerate(test_loader):
        if args.limit_sequences is not None and i >= args.limit_sequences:
            break
        try:
            seq_info_i = test_dataset.sequence_info[i]
            file_path_i = seq_info_i.get('file_path', '')
            file_name_i = os.path.basename(file_path_i) if file_path_i else ''
        except Exception:
            file_path_i, file_name_i = '', ''
        data_list.append({
            'batch': batch,
            'seq_file_path': file_path_i,
            'seq_file_name': file_name_i,
        })
        if i % 50 == 0 and i > 0:
            print(f"  Loaded {i+1} sequences...")
    print(f"Finished loading {len(data_list)} sequences.")

    if not data_list:
        print("Error: No data loaded.")
        return

    print("Initializing Interactive Viewer...")
    pred_offset_np = np.array(args.pred_offset, dtype=np.float32)
    if args.overlay_frames is not None:
        print(f"Overlay frame mode enabled: {args.overlay_frames}")
    
    viewer_instance = InteractiveViewer(
        data_list=data_list,
        model=model,
        smpl_model=smpl_model,
        config=config,
        device=device,
        obj_geo_root=args.obj_geo_root,
        show_objects=(not args.no_objects),
        vis_gt_only=args.vis_gt_only,
        show_foot_contact=args.show_foot_contact,
        show_obj_traj=args.show_obj_traj,
        show_hand_traj=args.show_hand_traj,
        use_fk=args.use_fk,
        compare_3=args.compare_3,
        pred_offset=pred_offset_np,
        no_trans=args.no_trans,
        overlay_frames=args.overlay_frames,
        window_size=(1920, 1080)
    )
    
    print("Viewer Initialized. Controls:")
    print("  q/e: Previous/Next 1 sequence")
    print("  Ctrl+q/e: Previous/Next 10 sequences")
    print("  Alt+q/e: Previous/Next 50 sequences")
    viewer_instance.run()


if __name__ == "__main__":
    main()

