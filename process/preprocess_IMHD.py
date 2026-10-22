import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import re
import glob
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from human_body_prior.body_model.body_model import BodyModel
import pytorch3d.transforms as transforms
from utils.human_pose import (
    build_smpl24_joint_positions,
    ensure_smpl24_local_rotations,
    local_rotations_to_global,
)

try:
    from preprocess import (
        add_contact_postprocess_arguments,
        build_unheld_object_motion_skip_record,
        contact_postprocess_metadata,
        compute_improved_contact_labels,
        compute_object_movement_mask,
        compute_foot_contact_labels,
        load_canonical_points_from_mesh_path,
        precompute_sequence_features,
        postprocess_sequence_contact_labels,
        write_contact_skip_manifest,
    )
except ModuleNotFoundError:
    from process.preprocess import (
        add_contact_postprocess_arguments,
        build_unheld_object_motion_skip_record,
        contact_postprocess_metadata,
        compute_improved_contact_labels,
        compute_object_movement_mask,
        compute_foot_contact_labels,
        load_canonical_points_from_mesh_path,
        precompute_sequence_features,
        postprocess_sequence_contact_labels,
        write_contact_skip_manifest,
    )

DEFAULT_CHUNK_LEN = 4000
CANONICAL_BODY_MODEL_TYPE = "smplh"
CANONICAL_BODY_MODEL_GENDER = "neutral"
CANONICAL_BETAS = "zero"
CANONICAL_SKELETON_VERSION = 1
CANONICAL_BODY_MODEL_RELATIVE_PATH = os.path.join("smplh", "neutral", "model.npz")

ROT_Z90 = np.array(
    [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

IMHD_HEAD_JOINT = 15
IMHD_FOOT_JOINTS = (7, 8, 10, 11)
IMHD_TARGET_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
IMHD_UP_ALIGNMENT_MIN_LEN = 0.25
IMHD_UP_ALIGNMENT_EPS = 1e-6


def _moving_avg_timewise(x, k=3):
    """
    沿时间维做滑动平均：
      - 支持 torch.Tensor / numpy.ndarray
      - 输入形状可为 [T], [T, C], [T, ..., C]（例如 [T, 22, 3]）
      - 仅沿时间维平滑；其它维（关节/通道）互不混合
      - 保持原始形状与（若为浮点）dtype
    """
    is_np = isinstance(x, np.ndarray)
    xt = torch.as_tensor(x)  # 零拷贝包装
    orig_dtype = xt.dtype
    orig_shape = tuple(xt.shape)

    # 只对浮点做平滑；非浮点（如 bool 标签）不要传进来
    if not torch.is_floating_point(xt):
        xt = xt.float()

    # 统一到 [T, C_total]，C_total = 其余维度的乘积（如 22*3）
    if xt.ndim == 1:
        xt = xt[:, None]  # [T, 1]
        rest_shape = (1,)
    else:
        T = xt.shape[0]
        rest_shape = tuple(xt.shape[1:])
        C_total = int(torch.tensor(rest_shape).prod().item())
        xt = xt.reshape(T, C_total)  # [T, C_total]

    T = xt.shape[0]
    if k <= 1 or T <= 2:
        out = xt
    else:
        pad = (k - 1) // 2
        # NCL: [1, C, T]
        x_ncl = xt.transpose(0, 1).unsqueeze(0)
        x_pad = F.pad(x_ncl, (pad, pad), mode="replicate")
        y = F.avg_pool1d(x_pad, kernel_size=k, stride=1)  # 逐通道滑动平均
        out = y.squeeze(0).transpose(0, 1)  # [T, C_total]

    # 还原到原形状
    if len(rest_shape) == 1 and rest_shape[0] == 1 and len(orig_shape) == 1:
        out = out.squeeze(1)  # 还原 [T]
    else:
        out = out.reshape(orig_shape)

    if is_np:
        # numpy：保持浮点 dtype，不是浮点则返回 float32（通常输入已是浮点）
        return out.cpu().numpy().astype(x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32)
    else:
        # torch：尽量还原原始浮点 dtype
        if orig_dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            out = out.to(dtype=orig_dtype)
        return out


def _moving_avg_axis_angle_timewise(aa, k=3):
    """Temporal moving average of axis-angle rotations, performed on quaternions.

    Averaging axis-angle components directly is invalid near |theta| = pi, where
    the representation wraps (theta -> 2*pi - theta with a flipped axis) and a
    component-wise mean yields an unrelated rotation.  Here each rotation track is
    converted to quaternions, made sign-continuous over time, averaged, and
    re-normalised.

    aa: numpy array [T, 3*J] or [T, J, 3] (J independent rotation tracks).
    """
    aa = np.asarray(aa)
    orig_shape = aa.shape
    orig_dtype = aa.dtype if np.issubdtype(aa.dtype, np.floating) else np.float32
    T = orig_shape[0]
    if k <= 1 or T <= 2:
        return aa.astype(orig_dtype, copy=True)
    if orig_shape[-1] != 3 and (aa.ndim != 2 or orig_shape[1] % 3 != 0):
        raise ValueError(f"expected axis-angle data with trailing groups of 3, got {orig_shape}")

    rotvec = aa.reshape(T, -1, 3).astype(np.float64)
    num_tracks = rotvec.shape[1]
    quat = R.from_rotvec(rotvec.reshape(-1, 3)).as_quat().reshape(T, num_tracks, 4)

    # Sign continuity: flip q_t whenever it lies in the opposite hemisphere of q_{t-1}.
    # Flipping q_{t-1} also flips dot(q_t, q_{t-1}), so the cumulative product of the
    # raw per-step signs gives the consistent sign for every frame.
    step_sign = np.sign(np.sum(quat[1:] * quat[:-1], axis=-1))
    step_sign[step_sign == 0] = 1.0
    cum_sign = np.concatenate(
        (np.ones((1, num_tracks)), np.cumprod(step_sign, axis=0)), axis=0
    )
    quat = quat * cum_sign[..., None]

    quat = _moving_avg_timewise(quat, k=k)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    out = R.from_quat(quat.reshape(-1, 4)).as_rotvec().reshape(orig_shape)
    return out.astype(orig_dtype)


# ROT_Z90 = np.array([[1.0, 0.0, 0.0],
#                     [0.0, 1.0, 0.0],
#                     [0.0, 0.0, 1.0]], dtype=np.float32)

GT_FILE_PATTERN = re.compile(r"gt_(\d+)_(\d+)_(-?\d+)\.pkl")


def iter_slices(total: int, chunk_len: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_len):
        end = min(start + chunk_len, total)
        yield start, end


class BodyModelLoader:
    """Load the single canonical neutral, zero-shape SMPL-H model."""

    def __init__(self, support_dir: str, device: torch.device, num_betas: int = 16) -> None:
        self.model_path = os.path.join(support_dir, CANONICAL_BODY_MODEL_RELATIVE_PATH)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                "Canonical SMPL-H model not found; expected "
                f"{self.model_path}"
            )
        self._model = BodyModel(bm_fname=self.model_path, num_betas=int(num_betas))
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._model = self._model.to(device)
        self._num_betas = int(num_betas)

    @property
    def num_betas(self) -> int:
        return self._num_betas

    @property
    def model_type(self) -> str:
        return CANONICAL_BODY_MODEL_TYPE

    def get(self, gender=None, model_type=None) -> BodyModel:
        if model_type is not None and str(model_type).lower() != CANONICAL_BODY_MODEL_TYPE:
            raise ValueError(f"IMHD preprocessing only supports {CANONICAL_BODY_MODEL_TYPE}")
        return self._model

    def resolve(self, seq_data=None, gender=None, model_type=None) -> Tuple[BodyModel, str]:
        return self.get(gender=gender, model_type=model_type), CANONICAL_BODY_MODEL_TYPE


def canonical_body_model_metadata() -> Dict[str, object]:
    return {
        "canonical_body_model_gender": CANONICAL_BODY_MODEL_GENDER,
        "canonical_betas": CANONICAL_BETAS,
        "canonical_skeleton_version": CANONICAL_SKELETON_VERSION,
    }


def infer_object_name(seq_name: str, object_names: List[str]) -> str:
    lowered = seq_name.lower()
    for candidate in sorted(object_names, key=len, reverse=True):
        if candidate.lower() in lowered:
            return candidate
    raise ValueError(f"Unable to infer object name from sequence identifier '{seq_name}'")


_mesh_cache: Dict[str, str] = {}


def resolve_object_mesh(object_name: str, objects_root: str) -> str:
    if object_name in _mesh_cache:
        return _mesh_cache[object_name]
    candidate_paths: List[str] = []
    direct = os.path.join(objects_root, f"{object_name}.obj")
    if os.path.exists(direct):
        candidate_paths.append(direct)
    object_dir = os.path.join(objects_root, object_name)
    if os.path.isdir(object_dir):
        obj_candidates = sorted(glob.glob(os.path.join(object_dir, "*.obj")))
        preferred = [
            path for path in obj_candidates
            if "simplified" in os.path.basename(path) and "transformed" in os.path.basename(path)
        ]
        candidate_paths.extend(preferred or obj_candidates)
    if not candidate_paths:
        raise FileNotFoundError(f"Unable to locate mesh for object '{object_name}' under {objects_root}")
    mesh_path = candidate_paths[0]
    _mesh_cache[object_name] = mesh_path
    return mesh_path


def build_smpl_input(
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "root_orient": torch.tensor(root_orient, dtype=torch.float32, device=device),
        "pose_body": torch.tensor(pose_body, dtype=torch.float32, device=device),
        "trans": torch.tensor(trans, dtype=torch.float32, device=device),
    }


def compute_rotation_features_chunk(
    root_orient_chunk: np.ndarray,
    pose_body_chunk: np.ndarray,
    bm: BodyModel,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    poses = np.concatenate([root_orient_chunk, pose_body_chunk], axis=1).astype(np.float32)
    local_rot_mat = transforms.axis_angle_to_matrix(
        torch.tensor(poses, dtype=torch.float32, device=device).reshape(-1, 3)
    ).reshape(poses.shape[0], -1, 3, 3)
    local_rot_mat = ensure_smpl24_local_rotations(local_rot_mat)
    pose_aa = torch.cat(
        (torch.tensor(poses, dtype=torch.float32, device=device).reshape(poses.shape[0], -1, 3),
         torch.zeros(poses.shape[0], 2, 3, device=device)),
        dim=1,
    )
    rotation_global_matrot = local_rotations_to_global(local_rot_mat)
    return pose_aa.detach().cpu(), rotation_global_matrot.detach().cpu()


def left_multiply_axis_angles(
    axis_angles: np.ndarray,
    rotation: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    rot_t = torch.from_numpy(rotation).float().to(device).unsqueeze(0)
    aa_t = torch.from_numpy(axis_angles).float().to(device)
    mat = transforms.axis_angle_to_matrix(aa_t)
    mat = torch.matmul(rot_t.expand(mat.shape[0], -1, -1), mat)
    return transforms.matrix_to_axis_angle(mat).detach().cpu().numpy().astype(np.float32)


def rotate_smpl_translation(
    trans: np.ndarray,
    rotation: np.ndarray,
    root_rest_offset: np.ndarray,
) -> np.ndarray:
    return ((trans + root_rest_offset) @ rotation.T - root_rest_offset).astype(np.float32)


def rotate_world_vectors(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return (vectors @ rotation.T).astype(np.float32)


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < IMHD_UP_ALIGNMENT_EPS or target_norm < IMHD_UP_ALIGNMENT_EPS:
        return np.eye(3, dtype=np.float32)

    a = source / source_norm
    b = target / target_norm
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-6:
        return np.eye(3, dtype=np.float32)

    if dot < -1.0 + 1e-6:
        basis = np.eye(3, dtype=np.float64)
        axis = np.cross(a, basis[int(np.argmin(np.abs(a)))])
        axis /= np.linalg.norm(axis)
        return R.from_rotvec(axis * np.pi).as_matrix().astype(np.float32)

    axis = np.cross(a, b)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < IMHD_UP_ALIGNMENT_EPS:
        return np.eye(3, dtype=np.float32)
    axis /= axis_norm
    angle = np.arccos(dot)
    return R.from_rotvec(axis * angle).as_matrix().astype(np.float32)


def compute_body_up_vector(
    bm: BodyModel,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Estimate the sequence-level body-up direction from feet center to head."""
    smpl_input = build_smpl_input(root_orient, pose_body, trans, device)
    directions: List[torch.Tensor] = []
    total_frames = root_orient.shape[0]
    with torch.no_grad():
        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            kwargs = {key: value[start:end] for key, value in smpl_input.items()}
            body_out = bm(**kwargs)
            joints = build_smpl24_joint_positions(body_out.Jtr)
            body_up = joints[:, IMHD_HEAD_JOINT, :] - joints[:, IMHD_FOOT_JOINTS, :].mean(dim=1)
            valid = torch.linalg.norm(body_up, dim=1) > IMHD_UP_ALIGNMENT_MIN_LEN
            if valid.any():
                body_up = F.normalize(body_up[valid], dim=1)
                directions.append(body_up.detach().cpu())
            del body_out
    if not directions:
        return IMHD_TARGET_UP.copy()
    up = torch.cat(directions, dim=0).median(dim=0).values.numpy().astype(np.float32)
    norm = float(np.linalg.norm(up))
    if norm < IMHD_UP_ALIGNMENT_EPS:
        return IMHD_TARGET_UP.copy()
    return (up / norm).astype(np.float32)


def compute_up_alignment_rotation(
    bm: BodyModel,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    body_up = compute_body_up_vector(
        bm,
        root_orient,
        pose_body,
        trans,
        device,
        batch_size=batch_size,
    )
    rotation = rotation_matrix_from_vectors(body_up, IMHD_TARGET_UP)
    aligned_up = rotation @ body_up
    alignment_cos = float(np.clip(np.dot(body_up, IMHD_TARGET_UP), -1.0, 1.0))
    return rotation, body_up.astype(np.float32), aligned_up.astype(np.float32), alignment_cos


def forward_smpl_in_batches(
    bm: BodyModel,
    smpl_input: Dict[str, torch.Tensor],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_frames = smpl_input["root_orient"].shape[0]
    joint_chunks: List[torch.Tensor] = []
    min_abs_y: Optional[float] = None
    with torch.no_grad():
        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            kwargs = {key: value[start:end] for key, value in smpl_input.items()}
            body_out = bm(**kwargs)
            joints = build_smpl24_joint_positions(body_out.Jtr).detach().cpu()
            verts = body_out.v
            batch_min_abs = float(verts[:, :, 1].abs().min().item())
            if min_abs_y is None or batch_min_abs < min_abs_y:
                min_abs_y = batch_min_abs
            joint_chunks.append(joints)
            del body_out
    position_global = torch.cat(joint_chunks, dim=0)
    min_y_tensor = torch.tensor(min_abs_y if min_abs_y is not None else 0.0, dtype=position_global.dtype)
    return position_global, min_y_tensor


def compute_foot_floor_offset(
    bm: BodyModel,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """计算整个序列中双脚关节的最低y坐标，用于地面对齐"""
    T = root_orient.shape[0]
    smpl_input = build_smpl_input(root_orient, pose_body, trans, device)
    
    foot_min_y: Optional[float] = None
    with torch.no_grad():
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            kwargs = {key: value[start:end] for key, value in smpl_input.items()}
            body_out = bm(**kwargs)
            joints = build_smpl24_joint_positions(body_out.Jtr)
            # 双脚关节: 7=左脚踝, 8=右脚踝, 10=左脚, 11=右脚
            left_ankle = joints[:, 7, 1]   # [B]
            right_ankle = joints[:, 8, 1]  # [B]
            left_foot = joints[:, 10, 1]   # [B]
            right_foot = joints[:, 11, 1]  # [B]
            
            batch_min = torch.min(torch.stack([left_ankle, right_ankle, left_foot, right_foot], dim=0)).item()
            foot_min_y = batch_min if foot_min_y is None else min(foot_min_y, batch_min)
            del body_out
    
    return float(foot_min_y or 0.0)


@dataclass
class SequenceEntry:
    seq_name: str
    motion_dir: str
    pkl_files: List[str]


def parse_gt_filename(path: str) -> Tuple[int, int, int]:
    name = os.path.basename(path)
    match = GT_FILE_PATTERN.match(name)
    if not match:
        raise ValueError(f"Unexpected ground truth filename: {name}")
    seg_idx, start, end = match.groups()
    return int(seg_idx), int(start), int(end)


def collect_sequences(dataset_root: str) -> List[SequenceEntry]:
    entries: List[SequenceEntry] = []
    for current_root, _, files in os.walk(dataset_root):
        gt_files = [os.path.join(current_root, f) for f in files if GT_FILE_PATTERN.match(f)]
        if not gt_files:
            continue
        rel_path = os.path.relpath(current_root, dataset_root)
        seq_name = "__".join(rel_path.split(os.sep))
        sorted_files = sorted(gt_files, key=lambda path: (parse_gt_filename(path)[1], parse_gt_filename(path)[0]))
        entries.append(SequenceEntry(seq_name=seq_name, motion_dir=current_root, pkl_files=sorted_files))
    entries.sort(key=lambda entry: entry.seq_name)
    return entries


def _resolve_imu_path(gt_path: str) -> Optional[str]:
    """
    Locate the IMU file matching a ground-truth PKL by mirroring the directory
    structure and swapping prefixes:
      - directory: 'ground_truth' -> 'imu_preprocessed'
      - filename: 'gt_*.pkl' -> 'imu_*.pkl'
    """
    marker = f"{os.sep}ground_truth{os.sep}"
    if marker in gt_path:
        imu_dir = gt_path.replace(marker, f"{os.sep}imu_preprocessed{os.sep}", 1)
    elif "ground_truth" in gt_path:
        imu_dir = gt_path.replace("ground_truth", "imu_preprocessed", 1)
    else:
        return None
    imu_name = os.path.basename(gt_path).replace("gt_", "imu_", 1)
    return os.path.join(os.path.dirname(imu_dir), imu_name)


def _load_object_imu(gt_path: str, expected_len: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load object IMU orientation/acceleration for a ground-truth segment if available."""
    imu_path = _resolve_imu_path(gt_path)
    if imu_path is None or not os.path.exists(imu_path):
        return None

    with open(imu_path, "rb") as f:
        imu_record = pickle.load(f)

    ori = imu_record.get("objectImuOri")
    acc = imu_record.get("objectImuAcc")
    if ori is None or acc is None:
        return None

    ori = np.asarray(ori)
    acc = np.asarray(acc)
    if ori.shape[0] != expected_len or acc.shape[0] != expected_len:
        return None

    return ori, acc


def load_imhd_sequence(pkl_files: List[str]) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    accum: Dict[str, List[np.ndarray]] = {
        "objectRot": [],
        "objectTrans": [],
        "smplPose": [],
        "smplTrans": [],
    }
    obj_imu_ori_chunks: List[np.ndarray] = []
    obj_imu_acc_chunks: List[np.ndarray] = []
    obj_imu_missing = False
    human_imu_real: Optional[np.ndarray] = None

    for path in pkl_files:
        with open(path, "rb") as f:
            record = pickle.load(f)
        for key in accum.keys():
            if key not in record:
                raise KeyError(f"Key '{key}' missing from {path}")
            accum[key].append(np.asarray(record[key]))

        imu_pair = _load_object_imu(path, expected_len=record["smplPose"].shape[0])
        if imu_pair is None:
            obj_imu_missing = True
        else:
            ori, acc = imu_pair
            obj_imu_ori_chunks.append(ori)
            obj_imu_acc_chunks.append(acc)

    obj_imu_real: Optional[Dict[str, np.ndarray]] = None
    if (not obj_imu_missing) and obj_imu_ori_chunks and obj_imu_acc_chunks:
        obj_imu_real = {
            "ori": np.concatenate(obj_imu_ori_chunks, axis=0),
            "acc": np.concatenate(obj_imu_acc_chunks, axis=0),
        }
        seq_hint = os.path.basename(pkl_files[0])
        print(f"[IMHD] Loaded real object IMU for {seq_hint}, frames={obj_imu_real['ori'].shape[0]}")

    return (
        {key: np.concatenate(chunks, axis=0) for key, chunks in accum.items()},
        human_imu_real,
        obj_imu_real,
    )


def process_imhd_sequence(
    entry: SequenceEntry,
    bm_loader: BodyModelLoader,
    objects_root: str,
    object_names: List[str],
    device: torch.device,
    output_dir: str,
    frame_trim: Optional[int] = None,
    smpl_batch: int = 256,
    obj_points_count: int = 256,
    contact_max_hole_frames: int = 30,
    contact_min_duration_frames: int = 30,
    skip_object_displacement_without_hand_contact: bool = True,
    unheld_object_displacement_threshold_m: float = 0.10,
    skipped_sequences=None,
) -> Optional[str]:
    output_file = f"{entry.seq_name}.pt"
    out_path = os.path.join(output_dir, output_file)

    data_np, human_imu_real, obj_imu_real = load_imhd_sequence(entry.pkl_files)
    
    # 在降采样前对位置、旋转和姿态数据进行平滑（k=3）
    # 平滑关键的连续数据以减少噪声
    # Translations are averaged directly; axis-angle rotations must be averaged on
    # quaternions (see _moving_avg_axis_angle_timewise), otherwise frames near
    # |theta| = pi turn into ~120 deg single-frame jumps.
    for key in ["smplTrans", "objectTrans"]:
        if key in data_np:
            data_np[key] = _moving_avg_timewise(data_np[key], k=3)
    for key in ["smplPose", "objectRot"]:
        if key in data_np:
            data_np[key] = _moving_avg_axis_angle_timewise(data_np[key], k=3)
    # if obj_imu_real is not None:
    #     obj_imu_real["ori"] = _moving_avg_timewise(obj_imu_real["ori"], k=3)
    #     obj_imu_real["acc"] = _moving_avg_timewise(obj_imu_real["acc"], k=3)
    
    # 降采样：60fps -> 30fps (每隔一帧取一帧)
    for key in data_np.keys():
        data_np[key] = data_np[key][::2]
    if obj_imu_real is not None:
        obj_imu_real["ori"] = obj_imu_real["ori"][::2]
        obj_imu_real["acc"] = obj_imu_real["acc"][::2]
    
    total_frames = data_np["smplPose"].shape[0]
    T = total_frames if frame_trim is None else min(total_frames, frame_trim)
    if T <= 0:
        raise ValueError(f"Sequence {entry.seq_name} has no frames after trimming")

    root_orient = data_np["smplPose"][:T, :3].astype(np.float32)
    pose_body = data_np["smplPose"][:T, 3:].astype(np.float32)
    trans = data_np["smplTrans"][:T].astype(np.float32)

    object_angles = data_np["objectRot"][:T].astype(np.float32)
    object_trans = data_np["objectTrans"][:T].astype(np.float32)

    if T > 0:
        root_offset = trans[0].copy()
        trans -= root_offset
        object_trans -= root_offset

    object_name = infer_object_name(entry.seq_name, object_names)
    mesh_path = resolve_object_mesh(object_name, objects_root)
    mesh_dir = os.path.dirname(mesh_path)
    mesh_basename = os.path.splitext(os.path.basename(mesh_path))[0]
    canonical_points = load_canonical_points_from_mesh_path(
        mesh_path,
        sample_count=int(max(1, obj_points_count)),
        device="cpu",
    )
    bm, body_model_type = bm_loader.resolve(model_type=CANONICAL_BODY_MODEL_TYPE)

    smpl_init_input = {
            "root_orient": torch.zeros_like(torch.from_numpy(root_orient.reshape(-1, 3)).to(device).float()),
            "pose_body": torch.zeros_like(torch.from_numpy(pose_body.reshape(-1, 63)).to(device).float()),
            "trans": torch.zeros_like(torch.from_numpy(trans.reshape(-1, 3)).to(device).float())
        }
    Jtr_0 = np.asarray(bm(**smpl_init_input).Jtr[:, 0, :].cpu(), dtype=np.float32)

    root_orient = left_multiply_axis_angles(root_orient, ROT_Z90, device)
    trans = rotate_smpl_translation(trans, ROT_Z90, Jtr_0)

    object_angles = left_multiply_axis_angles(object_angles, ROT_Z90, device)
    object_trans = rotate_world_vectors(object_trans, ROT_Z90)

    # ---------- 在地面对齐前将序列级身体上方向对齐到 +Y ----------
    up_alignment_rot, body_up_before, aligned_body_up, up_alignment_cos = compute_up_alignment_rotation(
        bm,
        root_orient,
        pose_body,
        trans,
        device,
        batch_size=smpl_batch,
    )
    if not np.allclose(up_alignment_rot, np.eye(3, dtype=np.float32), atol=1e-5):
        root_orient = left_multiply_axis_angles(root_orient, up_alignment_rot, device)
        trans = rotate_smpl_translation(trans, up_alignment_rot, Jtr_0)
        object_angles = left_multiply_axis_angles(object_angles, up_alignment_rot, device)
        object_trans = rotate_world_vectors(object_trans, up_alignment_rot)
        print(
            f"[IMHD] Aligned body up for {entry.seq_name} "
            f"(body_up_before={body_up_before.tolist()}, "
            f"cos_to_y_before={up_alignment_cos:.3f}, "
            f"aligned_up={aligned_body_up.tolist()})"
        )

    # 真实物体 IMU（IMHD 不含人体 IMU）
    human_imu_tensor: Optional[torch.Tensor] = None
    obj_imu_tensor: Optional[torch.Tensor] = None
    if obj_imu_real is not None:
        if obj_imu_real["ori"].shape[0] < T or obj_imu_real["acc"].shape[0] < T:
            obj_imu_real = None
        else:
            obj_ori_np = obj_imu_real["ori"][:T].astype(np.float32)
            obj_acc_np = obj_imu_real["acc"][:T].astype(np.float32)

            obj_ori_t = torch.from_numpy(obj_ori_np).float().to(device)
            obj_ori_mat = transforms.axis_angle_to_matrix(obj_ori_t)
            rot_z90_torch = torch.from_numpy(ROT_Z90).float().to(device).unsqueeze(0)
            obj_ori_mat = torch.matmul(rot_z90_torch.expand(obj_ori_mat.shape[0], -1, -1), obj_ori_mat)
            up_alignment_torch = torch.from_numpy(up_alignment_rot).float().to(device).unsqueeze(0)
            obj_ori_mat = torch.matmul(up_alignment_torch.expand(obj_ori_mat.shape[0], -1, -1), obj_ori_mat)
            obj_ori_6d = transforms.matrix_to_rotation_6d(obj_ori_mat).detach().cpu()

            obj_acc_np = rotate_world_vectors(obj_acc_np, ROT_Z90)
            obj_acc_np = rotate_world_vectors(obj_acc_np, up_alignment_rot)
            obj_acc_t = torch.from_numpy(obj_acc_np).float()

            obj_imu_tensor = torch.cat([obj_acc_t, obj_ori_6d], dim=-1).float()

    # ---------- 计算双脚最低点偏移 ----------
    foot_floor_y = compute_foot_floor_offset(
        bm, root_orient, pose_body, trans, device, batch_size=smpl_batch
    )
    # 修正 trans 和 object_trans，使双脚最低点对齐到 y=0
    trans[:, 1] -= foot_floor_y
    object_trans[:, 1] -= foot_floor_y

    rot_local_chunks: List[torch.Tensor] = []
    rot_global_chunks: List[torch.Tensor] = []
    pos_chunks: List[torch.Tensor] = []
    lfoot_chunks: List[torch.Tensor] = []
    rfoot_chunks: List[torch.Tensor] = []
    lhand_chunks: List[torch.Tensor] = []
    rhand_chunks: List[torch.Tensor] = []
    obj_contact_chunks: List[torch.Tensor] = []
    obj_trans_chunks: List[torch.Tensor] = []
    obj_rot_chunks: List[torch.Tensor] = []
    obj_scale_chunks: List[torch.Tensor] = []

    # obj_move is decided on the full sequence (low-pass + window) and sliced per chunk.
    obj_movement_mask_full = compute_object_movement_mask(
        object_trans[:T], R.from_rotvec(object_angles[:T]).as_matrix()
    )

    with torch.no_grad():
        for start, end in iter_slices(T, DEFAULT_CHUNK_LEN):
            smpl_input = build_smpl_input(
                root_orient[start:end],
                pose_body[start:end],
                trans[start:end],
                device,
            )

            pos_chunk, _ = forward_smpl_in_batches(bm, smpl_input, batch_size=smpl_batch)
            pos_chunks.append(pos_chunk)

            rot_local, rot_global = compute_rotation_features_chunk(
                root_orient[start:end], pose_body[start:end], bm, device
            )
            rot_local_chunks.append(rot_local.float())
            rot_global_chunks.append(rot_global.float())

            obj_rot_mat = R.from_rotvec(object_angles[start:end]).as_matrix().astype(np.float32)
            obj_rot_t = torch.tensor(obj_rot_mat, dtype=torch.float32, device=device)
            obj_trans_t = torch.tensor(object_trans[start:end], dtype=torch.float32, device=device)
            obj_scale_t = torch.ones((end - start,), dtype=torch.float32, device=device)

            pos_dev = pos_chunk.to(device)
            lfoot_c, rfoot_c = compute_foot_contact_labels(pos_dev)
            lhand_c, rhand_c, obj_c = compute_improved_contact_labels(
                obj_trans_t,
                obj_rot_t,
                obj_scale_t,
                pos_dev,
                mesh_dir,
                mesh_basename,
                device,
                end - start,
                obj_movement_mask=obj_movement_mask_full[start:end],
            )

            lfoot_chunks.append(lfoot_c.cpu())
            rfoot_chunks.append(rfoot_c.cpu())
            lhand_chunks.append(lhand_c.cpu())
            rhand_chunks.append(rhand_c.cpu())
            obj_contact_chunks.append(obj_c.cpu())
            obj_trans_chunks.append(obj_trans_t.cpu())
            obj_rot_chunks.append(obj_rot_t.cpu())
            obj_scale_chunks.append(obj_scale_t.cpu())

            del smpl_input, pos_chunk, pos_dev, obj_rot_t, obj_trans_t, obj_scale_t
            if device.type == "cuda":
                torch.cuda.empty_cache()

    position_global = torch.cat(pos_chunks, dim=0).float()
    pose = torch.cat(rot_local_chunks, dim=0).reshape(T, -1).float()
    rotation_global = torch.cat(rot_global_chunks, dim=0).float()
    lfoot_contact = torch.cat(lfoot_chunks, dim=0)
    rfoot_contact = torch.cat(rfoot_chunks, dim=0)
    lhand_contact_raw = torch.cat(lhand_chunks, dim=0).bool()
    rhand_contact_raw = torch.cat(rhand_chunks, dim=0).bool()
    obj_contact_raw = torch.cat(obj_contact_chunks, dim=0).bool()
    obj_trans_cpu = torch.cat(obj_trans_chunks, dim=0)
    obj_rot_cpu = torch.cat(obj_rot_chunks, dim=0)
    obj_scale_cpu = torch.cat(obj_scale_chunks, dim=0)
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
                dataset="IMHD",
                seq_name=entry.seq_name,
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
        "seq_name": entry.seq_name,
        "gender": CANONICAL_BODY_MODEL_GENDER,
        "body_model_type": body_model_type,
        **canonical_body_model_metadata(),
        "pose": pose,
        "position_global_full_gt_world": position_global,
        "rotation_global": rotation_global,
        "trans": trans,
        "human_imu_real": human_imu_tensor,
        "obj_imu_real": obj_imu_tensor,
        "lfoot_contact": lfoot_contact,
        "rfoot_contact": rfoot_contact,
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
        "source_pkls": [os.path.relpath(path, entry.motion_dir) for path in entry.pkl_files],
        "imhd_up_alignment_rotation": torch.from_numpy(up_alignment_rot).float(),
        "imhd_body_up_before_alignment": torch.from_numpy(body_up_before).float(),
        "imhd_aligned_body_up": torch.from_numpy(aligned_body_up).float(),
        "imhd_up_alignment_cos_to_y_before": float(up_alignment_cos),
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
            trans,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess IMHD dataset ground truth into PyTorch tensors.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=r"datasets/IMHD/IMHD Dataset/ground_truth",
        help="Root directory containing IMHD ground-truth PKL files.",
    )
    parser.add_argument(
        "--objects_root",
        type=str,
        default=r"datasets/IMHD/IMHD Dataset/object_templates",
        help="Directory that stores IMHD object template meshes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="process/processed_split_data_IMHD_bps",
        help="Directory where processed sequence tensors will be written.",
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default="datasets/smpl_models",
        help="Directory with SMPL-H support files.",
    )
    parser.add_argument(
        "--frame_trim",
        type=int,
        default=None,
        help="Optionally limit each sequence to this many frames.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force computation device (cpu or cuda).",
    )
    parser.add_argument(
        "--smpl_batch",
        type=int,
        default=64,
        help="Batch size for SMPL forward passes.",
    )
    parser.add_argument(
        "--num_betas",
        type=int,
        default=16,
        help="Number of body shape coefficients expected by the SMPL-H model.",
    )
    parser.add_argument(
        "--obj_points_count",
        type=int,
        default=256,
        help="Number of canonical object points to save per sequence.",
    )
    add_contact_postprocess_arguments(parser)
    parser.set_defaults(skip_object_displacement_without_hand_contact=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bm_loader = BodyModelLoader(args.support_dir, device, num_betas=args.num_betas)
    sequences = collect_sequences(args.dataset_root)
    if not sequences:
        raise FileNotFoundError(f"No IMHD PKL files found under {args.dataset_root}")

    object_names = [
        name for name in os.listdir(args.objects_root) if os.path.isdir(os.path.join(args.objects_root, name))
    ]
    if not object_names:
        raise FileNotFoundError(f"No object templates found in {args.objects_root}")

    processed_count = 0
    skipped_sequences = []
    failed_sequences = []
    for entry in tqdm(sequences, desc="Processing IMHD sequences"):
        try:
            output_path = process_imhd_sequence(
                entry,
                bm_loader,
                args.objects_root,
                object_names,
                device,
                args.output_dir,
                frame_trim=args.frame_trim,
                smpl_batch=args.smpl_batch,
                obj_points_count=args.obj_points_count,
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
        except Exception as exc:
            failed_sequences.append(entry.seq_name)
            print(f"Failed to process {entry.seq_name}: {exc}")

    manifest_path = write_contact_skip_manifest(
        args.output_dir,
        dataset="IMHD",
        source_sequences=len(sequences),
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
        f"IMHD preprocessing finished: saved={processed_count}, "
        f"contact-skipped={len(skipped_sequences)}, "
        f"failed={len(failed_sequences)}, manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
