#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo script that loads the preprocessed Noitom/STAG ``data_dict`` (see
``process/process_demo.py``), runs the IMUHOI TransPose model forward, and
visualizes the predicted human body together with the fused object trajectory.

The implementation mirrors the data preparation utilities from
``models/do_train_IMUHOI.py`` and ``eval_IMUHOI.py`` while reusing the viewing
logic from ``vis_IMUHOI.py`` in a simplified form.
"""

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
import pytorch3d.transforms as t3d
import trimesh
from easydict import EasyDict as edict
from aitviewer.viewer import Viewer
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.spheres import Spheres

from human_body_prior.body_model.body_model import BodyModel
from model import IMUHOIModel, load_model
from configs import _REDUCED_POSE_NAMES, _SENSOR_ROT_INDICES


R_YUP = torch.eye(3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMUHOI demo: run inference on preprocessed Noitom/STAG data.")
    parser.add_argument(
        "--config",
        default="configs/IMUHOI_train.yaml",
        help="Model config used to instantiate TransPoseNet.",
    )
    parser.add_argument(
        "--data-dict",
        default="noitom/7IMU/shoe004.pt",
        help="Serialized data_dict produced by process/process_demo.py.",
    )
    parser.add_argument(
        "--body-model",
        default="datasets/smpl_models/smplh/male/model.npz",
        help="Path to the SMPL-H body model required for mesh reconstruction.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Full TransPose checkpoint (.pt). If omitted, provide individual module weights.",
    )
    parser.add_argument("--model-arch", choices=["rnn", "dit"], default=None, help="Select backbone architecture.")
    parser.add_argument("--velocity-module", default='outputs/IMUHOI/joint_train_01112057/best_velocity_contact.pt', help="Pretrained velocity_contact module checkpoint.")
    parser.add_argument("--human-module", default='outputs/IMUHOI/joint_train_01112057/best_human_pose.pt', help="Pretrained human_pose module checkpoint.")
    parser.add_argument("--object-module", default='outputs/IMUHOI/joint_train_12311229/best_object_trans.pt', help="Pretrained object_trans module checkpoint.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for inference (e.g. cuda:0 or cpu).")
    parser.add_argument(
        "--object-mesh",
        default='/mnt/d/a_WORK/Projects/PhD/paper_work/EgoIMU/Rebuttal/shoe_transformed.obj',
        help="Optional triangulated mesh (.obj/.ply/...) that will be placed at the predicted object poses.",
    )
    parser.add_argument(
        "--object-scale",
        type=float,
        default=1,
        help="Uniform scale applied to the provided object mesh.",
    )
    parser.add_argument(
        "--object-size",
        type=float,
        default=0.12,
        help="Edge length (in meters) of the fallback cube mesh when --object-mesh is not specified.",
    )
    parser.add_argument(
        "--use-imu-rotation",
        default=True,
        help="Rotate the rendered object with the orientation coming from obj_imu (default: only translate).",
    )
    parser.add_argument(
        "--compute-fk",
        action="store_true",
        default=True,
        help="Enable FK branch outputs inside the object module (adds slight runtime overhead).",
    )
    parser.add_argument(
        "--save-pred",
        default=None,
        help="Optional .pt path that receives the raw model outputs for offline inspection.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip aitviewer visualization (useful for headless environments when only --save-pred is needed).",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"[demo] CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return requested


def load_config(config_path: str, device: torch.device) -> edict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = edict(cfg_dict)
    cfg.device = str(device)
    if device.type == "cuda":
        gpu_index = device.index if device.index is not None else 0
        cfg.gpus = [gpu_index]
    else:
        cfg.gpus = []
    return cfg


def build_model(cfg: edict, args: argparse.Namespace, device: torch.device) -> IMUHOIModel:
    module_paths = {}
    if args.velocity_module:
        module_paths["velocity_contact"] = args.velocity_module
    if args.human_module:
        module_paths["human_pose"] = args.human_module
    if args.object_module:
        module_paths["object_trans"] = args.object_module

    if not args.checkpoint and not module_paths:
        print("[demo] Warning: no pretrained checkpoints provided, the model will run with random weights.")

    # 使用新的 load_model 函数
    model = load_model(cfg, device, no_trans=False, module_paths=module_paths if module_paths else None)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[demo] Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"[demo] Unexpected keys when loading checkpoint: {unexpected}")
        print(f"[demo] Loaded checkpoint from {args.checkpoint}")

    return model


def load_body_model(path: str, device: torch.device) -> BodyModel:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SMPL-H model not found at {path}")
    smpl = BodyModel(bm_fname=path, num_betas=16).to(device)
    smpl.eval()
    return smpl


def _ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def load_data_dict(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location=device)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a dictionary.")

    result = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            tensor = value.to(device=device)
            if key in {"human_imu"} and tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            result[key] = tensor
        else:
            result[key] = value

    if "use_object_data" not in result:
        result["use_object_data"] = True

    has_object = result.get("has_object")
    if isinstance(has_object, torch.Tensor):
        result["has_object"] = has_object.to(device=device, dtype=torch.bool)
    elif has_object is None:
        result["has_object"] = torch.ones(1, dtype=torch.bool, device=device)
    else:
        has_object_tensor = torch.as_tensor(has_object, dtype=torch.bool, device=device)
        result["has_object"] = has_object_tensor.view(1) if has_object_tensor.dim() == 0 else has_object_tensor

    human_imu = result.get("human_imu")
    if human_imu is None:
        raise KeyError("data_dict must contain 'human_imu'.")

    print(f"[demo] Loaded data_dict ({path}) with sequence length {human_imu.shape[1]}")
    return result


def run_inference(
    cfg: edict,
    model: IMUHOIModel,
    data_dict: Dict[str, torch.Tensor],
    use_object_data: bool,
    compute_fk: bool,
) -> Dict[str, torch.Tensor]:

    with torch.no_grad():
        if cfg.model_arch == 'dit':
            outputs = model.inference(
                data_dict,
                gt_targets=data_dict,
                use_object_data=use_object_data,
                compute_fk=compute_fk,
            )
        else:
            outputs = model(data_dict, use_object_data=use_object_data, compute_fk=compute_fk)
    print("[demo] Model forward pass complete.")
    return outputs


def reconstruct_human_sequence(
    model: IMUHOIModel,
    smpl_model: BodyModel,
    data_dict: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    if model.human_pose_module is None:
        print("[demo] Human pose module is disabled; cannot reconstruct mesh.")
        return None, None

    p_pred = outputs.get("p_pred")
    root_trans = outputs.get("root_trans_pred")
    if p_pred is None or root_trans is None:
        print("[demo] Missing human pose predictions; skip mesh reconstruction.")
        return None, None

    human_imu = data_dict["human_imu"]
    bs = human_imu.shape[0]
    if bs == 0:
        return None, None

    p_seq = p_pred[0]
    trans_seq = root_trans[0]
    seq_len = p_seq.shape[0]

    reduced_pose = p_seq.view(seq_len, len(_REDUCED_POSE_NAMES), 6)
    imu_rot_6d = human_imu[0, :, :, -6:]
    imu_rot_mat = t3d.rotation_6d_to_matrix(imu_rot_6d.reshape(-1, 6)).reshape(
        seq_len, human_imu.shape[2], 3, 3
    )
    imu_subset = imu_rot_mat[:, : len(_SENSOR_ROT_INDICES), :, :]

    human_module = model.human_pose_module
    full_glb = human_module._reduced_glb_6d_to_full_glb_mat(reduced_pose, imu_subset)
    parents = human_module.smpl_parents.tolist()
    local_rot = human_module._global2local(full_glb, parents)
    pose_axis = t3d.matrix_to_axis_angle(local_rot.reshape(seq_len * full_glb.shape[1], 3, 3)).reshape(
        seq_len, full_glb.shape[1], 3
    )
    root_axis = pose_axis[:, 0, :]
    pose_body_axis = pose_axis[:, 1:22, :].reshape(seq_len, -1)

    with torch.no_grad():
        smpl_out = smpl_model(pose_body=pose_body_axis, root_orient=root_axis, trans=trans_seq)
    verts = smpl_out.v
    faces = (
        smpl_model.f.detach().cpu().numpy()
        if torch.is_tensor(smpl_model.f)
        else np.asarray(smpl_model.f, dtype=np.int32)
    )
    print("[demo] Reconstructed predicted human mesh.")
    return verts, faces

### box/shoe
# def load_object_template(
#     mesh_path: Optional[str],
#     mesh_scale: float,
#     fallback_size: float,
#     device: torch.device,
# ) -> Tuple[torch.Tensor, np.ndarray]:
#     if mesh_path:
#         print(f"[demo] 忽略 --object-mesh ({mesh_path})，改用内置长方体。")
#     _ = mesh_scale
#     _ = fallback_size

#     # length = 0.3
#     # width = 0.20
#     # height = 0.15
#     length = 0.11
#     width = 0.28
#     height = 0.05
#     half_len = length / 2.0
#     half_width = width / 2.0
#     top = 0.0
#     bottom = -height
#     vertices = torch.tensor(
#         [
#             [half_len, top, half_width],
#             [half_len, top, -half_width],
#             [-half_len, top, -half_width],
#             [-half_len, top, half_width],
#             [half_len, bottom, half_width],
#             [half_len, bottom, -half_width],
#             [-half_len, bottom, -half_width],
#             [-half_len, bottom, half_width],
#         ],
#         dtype=torch.float32,
#         device=device,
#     )
#     faces = np.array(
#         [
#             [0, 1, 2],
#             [0, 2, 3],
#             [4, 7, 6],
#             [4, 6, 5],
#             [0, 4, 5],
#             [0, 5, 1],
#             [1, 5, 6],
#             [1, 6, 2],
#             [2, 6, 7],
#             [2, 7, 3],
#             [3, 7, 4],
#             [3, 4, 0],
#         ],
#         dtype=np.int32,
#     )
#     print("[demo] Built 0.25×0.20×0.10 m box mesh（原点位于顶面中心，长沿 +X，宽沿 +Z，高沿 +Y）。")
#     return vertices, faces

### wine
# def load_object_template(
#     mesh_path: Optional[str],
#     mesh_scale: float,
#     fallback_size: float,
#     device: torch.device,
#     sections: int = 32  # 圆柱体的圆周分段数，越大越圆滑
# ) -> Tuple[torch.Tensor, np.ndarray]:
#     """Load external mesh if provided, otherwise use stepped-cylinder fallback."""
#     if mesh_path and os.path.isfile(mesh_path):
#         mesh = trimesh.load(mesh_path, force="mesh")
#         if isinstance(mesh, trimesh.Scene):
#             mesh = mesh.dump(concatenate=True)
#         if mesh_scale != 1:
#             mesh.apply_scale(mesh_scale)
#         vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
#         faces = np.asarray(mesh.faces, dtype=np.int32)
#         print(f"[demo] Loaded object mesh from {mesh_path} (verts={len(vertices)}, faces={len(faces)}), scale={mesh_scale}.")
#         return vertices, faces

#     # fallback parameters (meters)
#     _ = fallback_size
#     r1, h1 = 0.016, 0.10  # top section radius/height
#     r2, h2 = 0.04, 0.17  # bottom section radius/height

#     y0, y1, y2 = 0.0, -h1, -(h1 + h2)

#     vertices = [[0, y0, 0]]
#     angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
#     for y, r in [(y0, r1), (y1, r1), (y1, r2), (y2, r2)]:
#         for angle in angles:
#             vertices.append([r * np.cos(angle), y, r * np.sin(angle)])
#     vertices.append([0, y2, 0])

#     v_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)

#     faces = []
#     N = sections
#     for i in range(N):
#         nxt = (i + 1) % N
#         faces.append([0, i + 1, nxt + 1])
#         v1, v2 = i + 1, nxt + 1
#         v3, v4 = i + 1 + N, nxt + 1 + N
#         faces.extend([[v1, v3, v4], [v1, v4, v2]])
#         v1, v2 = i + 1 + N, nxt + 1 + N
#         v3, v4 = i + 1 + 2 * N, nxt + 1 + 2 * N
#         faces.extend([[v1, v3, v4], [v1, v4, v2]])
#         v1, v2 = i + 1 + 2 * N, nxt + 1 + 2 * N
#         v3, v4 = i + 1 + 3 * N, nxt + 1 + 3 * N
#         faces.extend([[v1, v3, v4], [v1, v4, v2]])
#         bottom_center = 4 * N + 1
#         faces.append([bottom_center, nxt + 1 + 3 * N, i + 1 + 3 * N])

#     f_array = np.array(faces, dtype=np.int32)
#     print(f"[demo] Built stepped cylinder fallback mesh (verts={len(vertices)}, faces={len(faces)}).")
#     return v_tensor, f_array

## bottol
# def load_object_template(
#     mesh_path: Optional[str],
#     mesh_scale: float,
#     fallback_size: float,
#     device: torch.device,
#     sections: int = 32
# ) -> Tuple[torch.Tensor, np.ndarray]:
    
#     # 忽略输入参数
#     _ = mesh_path
#     _ = mesh_scale
#     _ = fallback_size

#     # 参数设置 (单位：米)
#     radius = 0.03  # 直径6cm -> 半径0.03m
#     height = 0.16  # 高度16cm
    
#     top_y = 0.0
#     bottom_y = -height

#     vertices = []
    
#     # 1. 生成顶点
#     # 索引 0: 顶面中心点
#     vertices.append([0, top_y, 0])
    
#     # 索引 1 ~ N: 顶面圆周顶点
#     # 索引 N+1 ~ 2N: 底面圆周顶点
#     angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    
#     # 顶层圆周
#     for angle in angles:
#         vertices.append([radius * np.cos(angle), top_y, radius * np.sin(angle)])
        
#     # 底层圆周
#     for angle in angles:
#         vertices.append([radius * np.cos(angle), bottom_y, radius * np.sin(angle)])
            
#     # 索引 2N+1: 底面中心点
#     vertices.append([0, bottom_y, 0])
    
#     v_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
    
#     # 2. 生成面 (Faces)
#     faces = []
#     N = sections
    
#     for i in range(N):
#         next_i = (i + 1) % N
        
#         # a. 顶面封口 (中心点为0)
#         # 顶点顺序 0 -> i+1 -> next_i+1
#         faces.append([0, i + 1, next_i + 1])
        
#         # b. 侧面 (两个三角形组成一个矩形)
#         v_top_curr = i + 1
#         v_top_next = next_i + 1
#         v_bot_curr = i + 1 + N
#         v_bot_next = next_i + 1 + N
        
#         # 三角形1: 顶当前, 底当前, 底下一个
#         faces.append([v_top_curr, v_bot_curr, v_bot_next])
#         # 三角形2: 顶当前, 底下一个, 顶下一个
#         faces.append([v_top_curr, v_bot_next, v_top_next])
        
#         # c. 底面封口 (中心点为 2N + 1)
#         # 注意为了使法线朝外，底面的绕行方向与顶面相反
#         bottom_center = 2 * N + 1
#         faces.append([bottom_center, v_bot_next, v_bot_curr])

#     f_array = np.array(faces, dtype=np.int32)
    
#     print(f"[demo] Built cylinder: d=6cm, h=16cm (Origin at top center).")
#     return v_tensor, f_array


def load_object_template(
    mesh_path: Optional[str],
    mesh_scale: float,
    fallback_size: float,
    device: torch.device,
    sections: int = 32,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load an external object mesh when provided; otherwise return a cube of edge ``fallback_size``.
    The previous template variants are kept above as comments for reference.
    """
    if mesh_path and os.path.isfile(mesh_path):
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if mesh_scale != 1:
            mesh.apply_scale(mesh_scale)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        print(f"[demo] Loaded object mesh from {mesh_path} (verts={len(vertices)}, faces={len(faces)}), scale={mesh_scale}.")
        return vertices, faces

    half = fallback_size / 2.0
    vertices = torch.tensor(
        [
            [ half,  half,  half],
            [ half,  half, -half],
            [ half, -half,  half],
            [ half, -half, -half],
            [-half,  half,  half],
            [-half,  half, -half],
            [-half, -half,  half],
            [-half, -half, -half],
        ],
        dtype=torch.float32,
        device=device,
    )
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2],  # +X
            [4, 6, 7], [4, 7, 5],  # -X
            [0, 2, 6], [0, 6, 4],  # +Z
            [1, 5, 7], [1, 7, 3],  # -Z
            [0, 4, 5], [0, 5, 1],  # +Y
            [2, 3, 7], [2, 7, 6],  # -Y
        ],
        dtype=np.int32,
    )
    print(f"[demo] Using cube fallback mesh (edge={fallback_size} m).")
    return vertices, faces


def build_object_renderables(
    args: argparse.Namespace,
    data_dict: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Optional[torch.Tensor]]:
    pred_obj_trans = outputs.get("pred_obj_trans")
    if pred_obj_trans is None:
        print("[demo] No object predictions available.")
        return {}

    obj_trans_seq = pred_obj_trans[0]
    seq_len = obj_trans_seq.shape[0]

    base_vertices, faces = load_object_template(args.object_mesh, args.object_scale, args.object_size, device)

    if args.use_imu_rotation and "obj_imu" in data_dict:
        obj_imu = data_dict["obj_imu"][0]
        if obj_imu.shape[-1] >= 9:
            rot6d = obj_imu[:, 3:9]
            rot_mat = t3d.rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(rot6d.shape[0], 3, 3)
        else:
            rot_mat = torch.eye(3, device=device).unsqueeze(0).repeat(seq_len, 1, 1)
    else:
        rot_mat = torch.eye(3, device=device).unsqueeze(0).repeat(seq_len, 1, 1)

    # Normalize orientation so the first frame is identity (rest pose)
    if rot_mat.shape[0] > 0:
        r0_inv = rot_mat[0].transpose(0, 1)
        rot_mat = torch.matmul(r0_inv.unsqueeze(0), rot_mat)

    min_len = min(rot_mat.shape[0], seq_len)
    rot_mat = rot_mat[:min_len]
    obj_trans_seq = obj_trans_seq[:min_len]

    verts_seq = torch.einsum("tij,vj->tvi", rot_mat, base_vertices) + obj_trans_seq.unsqueeze(1)
    
    # FK Visualization
    fk_verts_seq = None
    if "pred_obj_trans_fk" in outputs:
        obj_trans_fk_seq = outputs["pred_obj_trans_fk"][0]
        obj_trans_fk_seq = obj_trans_fk_seq[:min_len]
        fk_verts_seq = torch.einsum("tij,vj->tvi", rot_mat, base_vertices) + obj_trans_fk_seq.unsqueeze(1)
        print("[demo] Prepared object FK geometry for visualization.")

    print("[demo] Prepared object geometry for visualization.")
    return {
        "verts": verts_seq,
        "faces": faces,
        "centers": obj_trans_seq.detach().cpu().numpy(),
        "fk_verts": fk_verts_seq,
    }


def visualize_predictions(
    human_verts: Optional[torch.Tensor],
    human_faces: Optional[np.ndarray],
    object_renderables: Dict[str, Optional[torch.Tensor]],
) -> None:
    if human_verts is None and not object_renderables:
        print("[demo] Nothing to visualize.")
        return

    viewer = Viewer(title="IMUHOI Demo")
    viewer.scene.floor.material.diffuse = 0.1
    viewer.scene.floor.tiling = False
    viewer.scene.floor.material.color = (132/255, 150/255, 183/255, 1.0)
    viewer.scene.floor.material.ambient = 0.30
    viewer.scene.background_color = (0.5, 0.5, 0.5, 1.0)
    device = human_verts.device if human_verts is not None else torch.device("cpu")

    if human_verts is not None and human_faces is not None:
        verts_yup = torch.matmul(human_verts, R_YUP.to(device).T)
        mesh = Meshes(
            verts_yup.detach().cpu().numpy(),
            human_faces,
            name="Pred-Human",
            color=(225/255, 166/255, 100/255, 1),
            gui_affine=False,
            is_selectable=False,
        )
        mesh.material.ambient = 0.32
        viewer.scene.add(mesh)

    obj_verts = object_renderables.get("verts")
    obj_faces = object_renderables.get("faces")
    if obj_verts is not None and obj_faces is not None:
        obj_yup = torch.matmul(obj_verts, R_YUP.to(obj_verts.device).T)
        obj_mesh = Meshes(
            obj_yup.detach().cpu().numpy(),
            obj_faces,
            name="Pred-Object",
            color=(164/255, 99/255, 204/255, 1),
            gui_affine=False,
            is_selectable=False,
        )
        obj_mesh.material.ambient = 0.41
        viewer.scene.add(obj_mesh)

    obj_fk_verts = object_renderables.get("fk_verts")
    if obj_fk_verts is not None and obj_faces is not None:
        obj_fk_yup = torch.matmul(obj_fk_verts, R_YUP.to(obj_fk_verts.device).T)
        obj_fk_mesh = Meshes(
            obj_fk_yup.detach().cpu().numpy(),
            obj_faces,
            name="Pred-Object-FK",
            color=(0.2, 0.8, 0.2, 0.6), # Green, semi-transparent
            gui_affine=False,
            is_selectable=False,
        )
        viewer.scene.add(obj_fk_mesh)

    elif object_renderables.get("centers") is not None:
        centers = object_renderables["centers"]
        spheres = Spheres(
            positions=centers,
            radius=0.03,
            name="Pred-Object-Centers",
            color=(0.2, 0.4, 0.9, 0.9),
            gui_affine=False,
            is_selectable=False,
        )
        viewer.scene.add(spheres)

    viewer.run()


def maybe_save_predictions(
    outputs: Dict[str, torch.Tensor],
    save_path: Optional[str],
) -> None:
    if not save_path:
        return
    torch.save(outputs, save_path)
    print(f"[demo] Saved raw predictions to {save_path}")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    cfg = load_config(args.config, device)
    if args.model_arch is not None:
        cfg.model_arch = args.model_arch
    model = build_model(cfg, args, device)
    smpl_model = load_body_model(args.body_model, device)
    data_dict = load_data_dict(args.data_dict, device)

    outputs = run_inference(
        cfg,
        model,
        data_dict,
        use_object_data=data_dict.get("use_object_data", True),
        compute_fk=args.compute_fk,
    )
    foot_contact_watch = outputs.get("contact_pred").detach().cpu().numpy()

    maybe_save_predictions(outputs, args.save_pred)

    human_verts, human_faces = reconstruct_human_sequence(model, smpl_model, data_dict, outputs)
    object_renderables = build_object_renderables(args, data_dict, outputs, device)

    if not args.no_viewer:
        visualize_predictions(human_verts, human_faces, object_renderables)
    else:
        print("[demo] Visualization skipped (--no-viewer).")


if __name__ == "__main__":
    main()
