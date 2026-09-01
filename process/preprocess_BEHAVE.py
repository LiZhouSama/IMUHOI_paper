import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from human_body_prior.body_model.body_model import BodyModel
import pytorch3d.transforms as transforms
from utils.human_pose import (
    GenderedBodyModelLoader,
    build_smpl24_joint_positions,
    ensure_smpl24_local_rotations,
    local_rotations_to_global,
)

try:
    from preprocess import (
        add_contact_postprocess_arguments,
        build_unheld_object_motion_skip_record,
        contact_postprocess_metadata,
        precompute_sequence_features,
        compute_improved_contact_labels,
        compute_foot_contact_labels,
        load_canonical_points_from_mesh_path,
        postprocess_sequence_contact_labels,
        write_contact_skip_manifest,
    )
except ModuleNotFoundError:
    from process.preprocess import (
        add_contact_postprocess_arguments,
        build_unheld_object_motion_skip_record,
        contact_postprocess_metadata,
        precompute_sequence_features,
        compute_improved_contact_labels,
        compute_foot_contact_labels,
        load_canonical_points_from_mesh_path,
        postprocess_sequence_contact_labels,
        write_contact_skip_manifest,
    )

DEFAULT_CHUNK_LEN = 1500      # 视显存/机型可调

R_Y_UP = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0]],
    dtype=np.float32,
)

def iter_slices(T: int, chunk_len: int):
    for s in range(0, T, chunk_len):
        e = min(s + chunk_len, T)
        yield s, e

def npz_get(npz_file: np.lib.npyio.NpzFile, key: str, default=None):
    return npz_file[key] if key in npz_file.files else default


class BodyModelLoader(GenderedBodyModelLoader):
    """BEHAVE ships SMPL-H fits (156-dim poses, 10 betas); the family is still
    re-confirmed per sequence through :meth:`resolve`."""

    def __init__(self, support_dir: str, device: torch.device, num_betas: int = 16) -> None:
        super().__init__(support_dir, device=device, num_betas=num_betas, default_type="smplh")


def infer_object_name(seq_name: str, object_names: List[str]) -> str:
    lowered = seq_name.lower()
    for candidate in sorted(object_names, key=len, reverse=True):
        if candidate.lower() in lowered:
            return candidate
    raise ValueError(f"Unable to infer object name from sequence {seq_name}")


_mesh_cache: Dict[str, str] = {}


def resolve_object_mesh(object_name: str, objects_root: str) -> str:
    if object_name in _mesh_cache:
        return _mesh_cache[object_name]

    candidate_paths: List[str] = []
    for ext in (".ply", ".obj"):
        direct = os.path.join(objects_root, f"{object_name}{ext}")
        if os.path.exists(direct):
            candidate_paths.append(direct)

    obj_dir = os.path.join(objects_root, object_name)
    if os.path.isdir(obj_dir):
        ply_candidates = sorted(glob.glob(os.path.join(obj_dir, "*.ply")))
        obj_candidates = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))
        candidate_paths.extend(ply_candidates + obj_candidates)

    if not candidate_paths:
        raise FileNotFoundError(f"Unable to locate mesh for object '{object_name}' under {objects_root}")

    mesh_path = candidate_paths[0]
    _mesh_cache[object_name] = mesh_path
    return mesh_path


def pad_betas(betas: np.ndarray, target_dim: int, target_frames: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = betas[None, :]
    if betas.shape[0] == 1 and target_frames > 1:
        betas = np.repeat(betas, repeats=target_frames, axis=0)
    if betas.shape[0] < target_frames:
        betas = np.pad(betas, ((0, target_frames - betas.shape[0]), (0, 0)), mode='edge')
    else:
        betas = betas[:target_frames]
    if betas.shape[1] < target_dim:
        betas = np.concatenate([
            betas,
            np.zeros((betas.shape[0], target_dim - betas.shape[1]), dtype=betas.dtype)
        ], axis=1)
    elif betas.shape[1] > target_dim:
        betas = betas[:, :target_dim]
    return betas


def build_smpl_input(root_orient: np.ndarray, pose_body: np.ndarray, trans: np.ndarray,
                     betas: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    smpl_input: Dict[str, torch.Tensor] = {
        "root_orient": torch.tensor(root_orient, dtype=torch.float32, device=device),
        "pose_body": torch.tensor(pose_body, dtype=torch.float32, device=device),
        "trans": torch.tensor(trans, dtype=torch.float32, device=device)
    }
    if betas.size > 0:
        smpl_input["betas"] = torch.tensor(betas, dtype=torch.float32, device=device)
    return smpl_input


def compute_rotation_features_chunk(
    root_orient_chunk: np.ndarray, pose_body_chunk: np.ndarray,
    bm: BodyModel, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    # [B, 3] + [B, 63] -> SMPL-24 local axis-angle and global rotations.
    bdata_poses = np.concatenate([root_orient_chunk, pose_body_chunk], axis=1).astype(np.float32)
    pose_aa = torch.tensor(bdata_poses, dtype=torch.float32, device=device).reshape(
        bdata_poses.shape[0], -1, 3
    )
    local_rot_mat = transforms.axis_angle_to_matrix(pose_aa)
    local_rot_mat = ensure_smpl24_local_rotations(local_rot_mat)
    palm_pose = torch.zeros(
        pose_aa.shape[0], 2, 3, device=device, dtype=pose_aa.dtype
    )
    pose_aa = torch.cat((pose_aa, palm_pose), dim=1)
    rotation_global_matrot = local_rotations_to_global(local_rot_mat)
    return pose_aa.detach().cpu(), rotation_global_matrot.detach().cpu()


def forward_smpl_in_batches(
    bm: BodyModel,
    smpl_input: Dict[str, torch.Tensor],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward SMPL in manageable batches and return joints plus the global minimum |y| vertex height."""
    total_frames = smpl_input["root_orient"].shape[0]
    joint_chunks: List[torch.Tensor] = []
    min_abs_y_value: Optional[float] = None
    with torch.no_grad():
        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            kwargs = {
                key: value[start:end]
                for key, value in smpl_input.items()
            }
            body_out = bm(**kwargs)
            joints = build_smpl24_joint_positions(body_out.Jtr).detach().cpu()
            verts = body_out.v
            batch_min_abs = float(verts[:, :, 1].abs().min().item())
            if min_abs_y_value is None or batch_min_abs < min_abs_y_value:
                min_abs_y_value = batch_min_abs
            joint_chunks.append(joints)
            del body_out
    position_global = torch.cat(joint_chunks, dim=0)
    min_y = torch.tensor(min_abs_y_value if min_abs_y_value is not None else 0.0, dtype=position_global.dtype)
    return position_global, min_y


def compute_foot_floor_offset(
    bm: BodyModel,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    device: torch.device,
) -> float:
    """计算整个序列中双脚关节的最低y坐标，用于地面对齐"""
    T = root_orient.shape[0]
    smpl_input = build_smpl_input(root_orient, pose_body, trans, betas, device)
    
    foot_min_y: Optional[float] = None
    with torch.no_grad():
        kwargs = {key: value for key, value in smpl_input.items()}
        body_out = bm(**kwargs)
        joints = build_smpl24_joint_positions(body_out.Jtr)  # [B, 24, 3]
        # 双脚关节: 7=左脚踝, 8=右脚踝, 10=左脚, 11=右脚
        left_ankle = joints[:, 7, 1]   # [B]
        right_ankle = joints[:, 8, 1]  # [B]
        left_foot = joints[:, 10, 1]   # [B]
        right_foot = joints[:, 11, 1]  # [B]
        foot_min_y = torch.min(torch.stack([left_ankle, right_ankle, left_foot, right_foot], dim=0)).item()
    
    return float(foot_min_y or 0.0)


def process_behave_sequence(seq_dir: str, bm_loader: BodyModelLoader, objects_root: str,
                            object_names: List[str], device: torch.device,
                            output_dir: str, frame_trim: Optional[int] = None,
                            smpl_batch: int = 256,
                            obj_points_count: int = 256,
                            contact_max_hole_frames: int = 30,
                            contact_min_duration_frames: int = 30,
                            skip_object_displacement_without_hand_contact: bool = True,
                            unheld_object_displacement_threshold_m: float = 0.10,
                            skipped_sequences=None) -> Optional[str]:
    seq_name = os.path.basename(seq_dir.rstrip(os.sep))
    output_file = f"{seq_name}.pt"
    out_path = os.path.join(output_dir, output_file)
    smpl_path = os.path.join(seq_dir, "smpl_fit_all.npz")
    obj_path = os.path.join(seq_dir, "object_fit_all.npz")
    if not (os.path.exists(smpl_path) and os.path.exists(obj_path)):
        raise FileNotFoundError(f"Missing expected npz files under {seq_dir}")

    smpl_data = np.load(smpl_path, mmap_mode='r')
    obj_data  = np.load(obj_path,  mmap_mode='r')

    total_frames = smpl_data['poses'].shape[0]
    obj_frames = obj_data['angles'].shape[0]
    T = min(total_frames, obj_frames)
    if frame_trim is not None:
        T = min(T, frame_trim)
    if T <= 0:
        raise ValueError(f"Sequence {seq_name} has no frames after trimming")

    poses = smpl_data['poses'][:T].astype(np.float32)
    trans = smpl_data['trans'][:T].astype(np.float32)
    trans_first_frame = trans[0].copy()
    trans_norm = trans - trans_first_frame

    betas_raw = npz_get(smpl_data, 'betas', np.zeros((1, bm_loader.num_betas), dtype=np.float32))
    betas = pad_betas(betas_raw, bm_loader.num_betas, T)

    gender_raw = npz_get(smpl_data, 'gender', 'neutral')
    gender = str(gender_raw)

    object_angles = obj_data['angles'][:T].astype(np.float32)
    object_trans = obj_data['trans'][:T].astype(np.float32)
    object_name = infer_object_name(seq_name, object_names)

    object_trans_norm = object_trans - trans_first_frame

    mesh_path = resolve_object_mesh(object_name, objects_root)
    mesh_dir = os.path.dirname(mesh_path)
    mesh_basename = os.path.splitext(os.path.basename(mesh_path))[0]
    canonical_points = load_canonical_points_from_mesh_path(
        mesh_path,
        sample_count=int(max(1, obj_points_count)),
        device="cpu",
    )
    # BEHAVE fits are SMPL-H (156-dim poses); confirm from the sequence rather
    # than assuming, so a re-released SMPL-X variant would not silently pass.
    bm, body_model_type = bm_loader.resolve(
        {"poses": poses, "betas": betas_raw, "gender": gender}, gender=gender
    )
    smpl_init_input = {
            "root_orient": torch.zeros_like(torch.from_numpy(poses[:, :3].reshape(-1, 3)).to(device).float()),
            "pose_body": torch.zeros_like(torch.from_numpy(poses[:, 3:66].reshape(-1, 63)).to(device).float()),
            "trans": torch.zeros_like(torch.from_numpy(trans_norm.reshape(-1, 3)).to(device).float())
        }
    Jtr_0 = np.asarray(bm(**smpl_init_input).Jtr[:, 0, :].cpu(), dtype=np.float32)

    r_y_up_torch = torch.from_numpy(R_Y_UP).float().to(device)
    r_y_up_repeat = r_y_up_torch.unsqueeze(0).repeat(T, 1, 1)
    root_ori_t = torch.from_numpy(poses[:, :3]).float().to(device)
    root_ori_mat = transforms.axis_angle_to_matrix(root_ori_t)
    root_ori_t = r_y_up_repeat @ root_ori_mat
    root_ori = transforms.matrix_to_axis_angle(root_ori_t).detach().cpu().numpy().astype(np.float32)
    poses[:, :3] = root_ori
    trans_norm = (trans_norm + Jtr_0) @ R_Y_UP.T - Jtr_0
    object_angles_t = torch.from_numpy(object_angles).float().to(device)
    obj_rot_mat = transforms.axis_angle_to_matrix(object_angles_t)
    obj_rot_mat = torch.matmul(r_y_up_repeat, obj_rot_mat)
    object_angles_t = transforms.matrix_to_axis_angle(obj_rot_mat)
    object_angles = object_angles_t.detach().cpu().numpy().astype(np.float32)
    object_trans_norm = object_trans_norm @ R_Y_UP.T

    # BEHAVE 目前默认没有真实人体/物体 IMU，保持与 IMHD 相同字段结构
    human_imu_tensor: Optional[torch.Tensor] = None
    obj_imu_tensor: Optional[torch.Tensor] = None
    
    # ---------- 计算双脚最低点偏移 ----------
    foot_floor_y = compute_foot_floor_offset(
        bm, poses[:, :3], poses[:, 3:66], trans_norm, betas, device
    )
    # 修正 trans 和 object_trans，使双脚最低点对齐到 y=0
    trans_norm[:, 1] -= foot_floor_y
    object_trans_norm[:, 1] -= foot_floor_y

    # ---------- 分块产出+拼接 ----------
    pose_list        : List[torch.Tensor] = []
    rot_global_list  : List[torch.Tensor] = []
    pos_global_list  : List[torch.Tensor] = []
    lfoot_list       : List[torch.Tensor] = []
    rfoot_list       : List[torch.Tensor] = []
    lhand_list       : List[torch.Tensor] = []
    rhand_list       : List[torch.Tensor] = []
    objcontact_list  : List[torch.Tensor] = []
    obj_trans_all    : List[torch.Tensor] = []
    obj_rot_all      : List[torch.Tensor] = []
    obj_scale_all    : List[torch.Tensor] = []

    with torch.no_grad():
        for s, e in iter_slices(T, DEFAULT_CHUNK_LEN):
            # 切块取数据
            root_orient_slice = poses[s:e, :3].astype(np.float32)
            pose_body_slice   = poses[s:e, 3:66].astype(np.float32)
            trans_slice       = trans_norm[s:e].astype(np.float32)

            object_angles_slice = object_angles[s:e].astype(np.float32)
            object_trans_slice  = object_trans_norm[s:e].astype(np.float32)

            betas_chunk = pad_betas(betas_raw, bm_loader.num_betas, e - s)
            smpl_input = build_smpl_input(root_orient_slice, pose_body_slice, trans_slice, betas_chunk, device)

            # 你的 forward_smpl_in_batches（本 chunk joints）
            pos_chunk, _ = forward_smpl_in_batches(bm, smpl_input, batch_size=smpl_batch)
            # 整段一致：加全局 floor（保持你的 |y| 语义）
            # pos_chunk[:, :, 1] += human_floor_abs
            pos_global_list.append(pos_chunk)

            # 旋转特征（分块）
            pose_chunk, rot_global = compute_rotation_features_chunk(root_orient_slice, pose_body_slice, bm, device)
            pose_list.append(pose_chunk.float())
            rot_global_list.append(rot_global.float())

            # 物体：旋转/平移张量 + 全局 floor
            obj_rot_t   = torch.tensor(R.from_rotvec(object_angles_slice).as_matrix().astype(np.float32),
                                       dtype=torch.float32, device=device)
            obj_trans_t = torch.tensor(object_trans_slice, dtype=torch.float32, device=device)
            # obj_trans_t[:, 1] += object_floor_abs  # 整段一致
            obj_scale_t = torch.ones((e - s,), dtype=torch.float32, device=device)

            # 接触（按 chunk 跑）
            pos_dev = pos_chunk.to(device)
            lfoot_c, rfoot_c = compute_foot_contact_labels(pos_dev)
            lhand_c, rhand_c, obj_c = compute_improved_contact_labels(
                obj_trans_t, obj_rot_t, obj_scale_t,
                pos_dev, mesh_dir, object_name, device, e - s
            )

            lfoot_list.append(lfoot_c.cpu()); rfoot_list.append(rfoot_c.cpu())
            lhand_list.append(lhand_c.cpu()); rhand_list.append(rhand_c.cpu()); objcontact_list.append(obj_c.cpu())
            obj_trans_all.append(obj_trans_t.detach().cpu())
            obj_rot_all.append(obj_rot_t.detach().cpu())
            obj_scale_all.append(obj_scale_t.detach().cpu())

            del smpl_input, pos_chunk, pos_dev, obj_rot_t, obj_trans_t, obj_scale_t
            if device.type == "cuda": torch.cuda.empty_cache()

    # 拼接
    position_global = torch.cat(pos_global_list, dim=0).float()
    pose = torch.cat(pose_list, dim=0).reshape(T, -1).float()
    rotation_global        = torch.cat(rot_global_list, dim=0).float()
    lfoot_contact = torch.cat(lfoot_list, dim=0)
    rfoot_contact = torch.cat(rfoot_list, dim=0)
    lhand_contact_raw = torch.cat(lhand_list, dim=0).bool()
    rhand_contact_raw = torch.cat(rhand_list, dim=0).bool()
    obj_contact_raw = torch.cat(objcontact_list, dim=0).bool()
    obj_trans_cpu = torch.cat(obj_trans_all, dim=0)
    obj_rot_cpu   = torch.cat(obj_rot_all,   dim=0)
    obj_scale_cpu = torch.cat(obj_scale_all, dim=0)
    (
        lhand_contact,
        rhand_contact,
        obj_contact,
        unheld_motion_interval,
    ) = postprocess_sequence_contact_labels(
        lhand_contact_raw,
        rhand_contact_raw,
        obj_contact_raw,
        obj_trans_cpu,
        max_hole_frames=contact_max_hole_frames,
        min_contact_frames=contact_min_duration_frames,
        skip_object_displacement_without_hand_contact=(
            skip_object_displacement_without_hand_contact
        ),
        unheld_object_displacement_threshold_m=(
            unheld_object_displacement_threshold_m
        ),
    )
    if unheld_motion_interval is not None:
        if skipped_sequences is not None:
            skipped_sequences.append(build_unheld_object_motion_skip_record(
                dataset="BEHAVE",
                seq_name=seq_name,
                output_file=output_file,
                obj_name=object_name,
                obj_trans=obj_trans_cpu,
                lhand_contact_raw=lhand_contact_raw,
                rhand_contact_raw=rhand_contact_raw,
                obj_contact_raw=obj_contact_raw,
                lhand_contact=lhand_contact,
                rhand_contact=rhand_contact,
                obj_contact=obj_contact,
                unheld_motion_interval=unheld_motion_interval,
            ))
        if os.path.isfile(out_path):
            os.remove(out_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return None

    output = {
        "seq_name": seq_name,
        "gender": gender,
        "body_model_type": body_model_type,
        "pose": pose,
        "position_global_full_gt_world": position_global,
        "rotation_global": rotation_global,
        "trans": trans_norm,
        "human_imu_real": human_imu_tensor,
        "obj_imu_real": obj_imu_tensor,
        "lfoot_contact": lfoot_contact,
        "rfoot_contact": rfoot_contact,
        "betas": torch.from_numpy(betas_raw).float(),
        "obj_name": object_name,
        "obj_mesh_name": mesh_basename,
        "obj_points_canonical": canonical_points.half().cpu(),
        "has_object": True,
        "obj_scale": obj_scale_cpu,
        "obj_trans": obj_trans_cpu,
        "obj_rot": obj_rot_cpu,
        "lhand_contact": lhand_contact,
        "rhand_contact": rhand_contact,
        "obj_contact": obj_contact,
        "lhand_contact_raw": lhand_contact_raw,
        "rhand_contact_raw": rhand_contact_raw,
        "obj_contact_raw": obj_contact_raw,
        "source_pkls": [os.path.basename(smpl_path), os.path.basename(obj_path)],
        **contact_postprocess_metadata(
            max_hole_frames=contact_max_hole_frames,
            min_contact_frames=contact_min_duration_frames,
            skip_object_displacement_without_hand_contact=(
                skip_object_displacement_without_hand_contact
            ),
            unheld_object_displacement_threshold_m=(
                unheld_object_displacement_threshold_m
            ),
        ),
    }
    output.update(
        precompute_sequence_features(
            position_global,
            rotation_global,
            torch.from_numpy(trans_norm).float(),
            obj_trans=obj_trans_cpu,
            obj_rot=obj_rot_cpu,
            human_imu=human_imu_tensor,
            obj_imu=obj_imu_tensor,
            lhand_contact=lhand_contact,
            rhand_contact=rhand_contact,
            obj_contact=obj_contact,
            has_object=True,
        )
    )

    os.makedirs(output_dir, exist_ok=True)
    torch.save(output, out_path)

    if device.type == "cuda": 
        torch.cuda.empty_cache()
    return out_path



def collect_sequences(dataset_root: str) -> List[str]:
    seq_dirs: List[str] = []
    for entry in sorted(os.listdir(dataset_root)):
        full_path = os.path.join(dataset_root, entry)
        if os.path.isdir(full_path):
            smpl_path = os.path.join(full_path, "smpl_fit_all.npz")
            obj_path = os.path.join(full_path, "object_fit_all.npz")
            if os.path.exists(smpl_path) and os.path.exists(obj_path):
                seq_dirs.append(full_path)
    return seq_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess BEHAVE dataset sequences into pt files")
    parser.add_argument("--dataset_root", type=str, default='datasets/BEHAVE/sequences', help="Root directory of BEHAVE sequences")
    parser.add_argument("--objects_root", type=str, default='datasets/BEHAVE/objects', help="Directory containing BEHAVE object meshes")
    parser.add_argument("--output_dir", type=str, default='process/processed_split_data_BEHAVE', help="Directory to store processed pt files")
    parser.add_argument("--support_dir", type=str, default="datasets/smpl_models", help="Directory with SMPL-H support files")
    parser.add_argument("--frame_trim", type=int, default=None, help="Optional maximum number of frames per sequence")
    parser.add_argument("--device", type=str, default=None, help="Force computation device (cpu or cuda)")
    parser.add_argument("--smpl_batch", type=int, default=256, help="Batch size for SMPL forward pass")
    parser.add_argument(
        "--obj_points_count",
        type=int,
        default=256,
        help="Number of canonical object points to save per sequence.",
    )
    parser.add_argument(
        "--num_betas",
        type=int,
        default=16,
        help="Number of body shape coefficients expected by the SMPL-H model.",
    )
    add_contact_postprocess_arguments(parser)
    parser.set_defaults(skip_object_displacement_without_hand_contact=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm_loader = BodyModelLoader(args.support_dir, device, num_betas=args.num_betas)

    seq_dirs = collect_sequences(args.dataset_root)
    if not seq_dirs:
        raise FileNotFoundError(f"No valid sequences found under {args.dataset_root}")

    object_names = [name for name in os.listdir(args.objects_root) if os.path.isdir(os.path.join(args.objects_root, name))]
    if not object_names:
        raise FileNotFoundError(f"No object subdirectories found in {args.objects_root}")

    processed_count = 0
    skipped_sequences = []
    for seq_dir in tqdm(seq_dirs, desc="Processing BEHAVE sequences"):
        output_path = process_behave_sequence(
            seq_dir,
            bm_loader,
            args.objects_root,
            object_names,
            device,
            args.output_dir,
            args.frame_trim,
            args.smpl_batch,
            args.obj_points_count,
            contact_max_hole_frames=args.contact_max_hole_frames,
            contact_min_duration_frames=args.contact_min_duration_frames,
            skip_object_displacement_without_hand_contact=(
                args.skip_object_displacement_without_hand_contact
            ),
            unheld_object_displacement_threshold_m=(
                args.unheld_object_displacement_threshold_m
            ),
            skipped_sequences=skipped_sequences,
        )
        processed_count += int(output_path is not None)

    manifest_path = write_contact_skip_manifest(
        args.output_dir,
        dataset="BEHAVE",
        source_sequences=len(seq_dirs),
        saved_sequences=processed_count,
        skipped_sequences=skipped_sequences,
        max_hole_frames=args.contact_max_hole_frames,
        min_contact_frames=args.contact_min_duration_frames,
        skip_object_displacement_without_hand_contact=(
            args.skip_object_displacement_without_hand_contact
        ),
        unheld_object_displacement_threshold_m=(
            args.unheld_object_displacement_threshold_m
        ),
    )
    print(
        f"BEHAVE preprocessing finished: saved={processed_count}, "
        f"skipped={len(skipped_sequences)}, manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
