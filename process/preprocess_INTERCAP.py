import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import local2global_pose
import pytorch3d.transforms as transforms

try:
    from preprocess import (
        _load_mesh_arrays,
        compute_foot_contact_labels,
        compute_improved_contact_labels,
        load_canonical_points_from_mesh_path,
    )
except ModuleNotFoundError:
    from process.preprocess import (
        _load_mesh_arrays,
        compute_foot_contact_labels,
        compute_improved_contact_labels,
        load_canonical_points_from_mesh_path,
    )


R_Y_UP = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0]],
    dtype=np.float32,
)


def iter_slices(total: int, chunk_len: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_len):
        yield start, min(start + chunk_len, total)


def relpath(path: Path) -> str:
    return os.path.relpath(str(path), Path.cwd())


def build_downsample_indices(total_frames: int, source_fps: int, target_fps: int) -> np.ndarray:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"source_fps and target_fps must be positive, got {source_fps}->{target_fps}")
    if source_fps == target_fps:
        return np.arange(total_frames, dtype=np.int64)
    if source_fps < target_fps:
        raise ValueError(f"InterCap source fps {source_fps} is lower than target fps {target_fps}")

    stride_float = float(source_fps) / float(target_fps)
    stride = int(round(stride_float))
    if stride <= 0 or abs(stride_float - stride) > 1e-6:
        raise ValueError(f"InterCap fps conversion {source_fps}->{target_fps} is not an integer stride")
    return np.arange(0, total_frames, stride, dtype=np.int64)


def as_frame_array(value, name: str, width: int) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        arr = np.stack([np.asarray(item, dtype=np.float32).reshape(width) for item in value], axis=0)
        return arr.astype(np.float32)

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == width:
        arr = arr[:, 0, :]
    elif arr.ndim == 2 and arr.shape[-1] == width:
        pass
    elif arr.ndim == 1 and arr.size % width == 0:
        arr = arr.reshape(-1, width)
    else:
        raise ValueError(f"{name} has unsupported shape {arr.shape}, expected [T,{width}]")
    return arr.astype(np.float32)


def reshape_body_pose(value, expected_frames: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (expected_frames, 63):
        return arr.astype(np.float32)

    flat = arr.reshape(-1)
    if flat.size % 63 != 0:
        raise ValueError(f"body_pose has {flat.size} values, not divisible by 63")
    body_pose = flat.reshape(-1, 63)
    if body_pose.shape[0] < expected_frames:
        raise ValueError(f"body_pose has {body_pose.shape[0]} frames, expected {expected_frames}")
    return body_pose[:expected_frames].astype(np.float32)


def pad_betas(betas: np.ndarray, target_dim: int, target_frames: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = betas[None, :]
    betas = betas.reshape(betas.shape[0], -1)

    if betas.shape[0] == 1 and target_frames > 1:
        betas = np.repeat(betas, target_frames, axis=0)
    elif betas.shape[0] < target_frames:
        betas = np.pad(betas, ((0, target_frames - betas.shape[0]), (0, 0)), mode="edge")
    elif betas.shape[0] > target_frames:
        betas = betas[:target_frames]

    if betas.shape[1] < target_dim:
        pad = np.zeros((betas.shape[0], target_dim - betas.shape[1]), dtype=np.float32)
        betas = np.concatenate([betas, pad], axis=1)
    elif betas.shape[1] > target_dim:
        betas = betas[:, :target_dim]
    return betas.astype(np.float32)


class BodyModelLoader:
    def __init__(self, support_dir: str, device: torch.device, num_betas: int = 16) -> None:
        self._device = device
        self._num_betas = int(num_betas)
        self._models: Dict[str, BodyModel] = {}
        for gender in ("male", "female", "neutral"):
            model_path = Path(support_dir) / "smplh" / gender / "model.npz"
            if model_path.exists():
                self._models[gender] = BodyModel(
                    bm_fname=str(model_path),
                    num_betas=self._num_betas,
                ).to(device).eval()
        if not self._models:
            raise FileNotFoundError(f"No SMPL-H body models found under {support_dir}")
        self._default_gender = "neutral" if "neutral" in self._models else "male"

    @property
    def num_betas(self) -> int:
        return self._num_betas

    def get(self, gender: Optional[str]) -> BodyModel:
        if gender is None:
            return self._models[self._default_gender]
        return self._models.get(str(gender).lower(), self._models[self._default_gender])


def collect_intercap_segments(dataset_root: str) -> List[Path]:
    root = Path(dataset_root)
    segments: List[Path] = []
    for res2_path in sorted(root.glob("*/*/Seg_*/res_2.pkl")):
        mesh_dir = res2_path.parent / "Mesh"
        if mesh_dir.is_dir() and next(mesh_dir.glob("*_second_obj.ply"), None) is not None:
            segments.append(res2_path.parent)
        else:
            print(f"Warning: missing InterCap object mesh files under {res2_path.parent}, skipped")
    return segments


def mesh_path_for_frame(mesh_dir: Path, frame_idx: int) -> Optional[Path]:
    for suffix in ("ply", "obj"):
        path = mesh_dir / f"{int(frame_idx):05d}_second_obj.{suffix}"
        if path.exists():
            return path
    return None


def body_mesh_path_for_frame(mesh_dir: Path, frame_idx: int) -> Optional[Path]:
    for suffix in ("ply", "obj"):
        path = mesh_dir / f"{int(frame_idx):05d}_second.{suffix}"
        if path.exists():
            return path
    return None


def choose_canonical_mesh(mesh_dir: Path, source_frame_indices: np.ndarray) -> Tuple[Path, int]:
    for frame_idx in source_frame_indices:
        path = mesh_path_for_frame(mesh_dir, int(frame_idx))
        if path is not None:
            return path, int(frame_idx)

    candidates = sorted(mesh_dir.glob("*_second_obj.ply")) + sorted(mesh_dir.glob("*_second_obj.obj"))
    if not candidates:
        raise FileNotFoundError(f"No InterCap object mesh found under {mesh_dir}")
    frame_idx = int(candidates[0].name.split("_", 1)[0])
    return candidates[0], frame_idx


def load_intercap_sequence(
    segment_dir: Path,
    num_betas: int,
    source_fps: int,
    target_fps: int,
    gender: str,
    max_frames: Optional[int],
) -> Dict[str, object]:
    with open(segment_dir / "res_2.pkl", "rb") as handle:
        data = pickle.load(handle)

    root_orient = as_frame_array(data["global_orient"], "global_orient", 3)
    trans = as_frame_array(data["transl"], "transl", 3)
    pose_body = reshape_body_pose(data["body_pose"], root_orient.shape[0])
    obj_pose = as_frame_array(data["ob_pose"], "ob_pose", 3)
    obj_origin_trans = as_frame_array(data["ob_trans"], "ob_trans", 3)

    raw_T = min(root_orient.shape[0], pose_body.shape[0], trans.shape[0], obj_pose.shape[0], obj_origin_trans.shape[0])
    source_frame_indices = build_downsample_indices(raw_T, int(source_fps), int(target_fps))
    if max_frames is not None:
        source_frame_indices = source_frame_indices[: int(max_frames)]
    if source_frame_indices.shape[0] <= 0:
        raise ValueError(f"{segment_dir} has no frames after downsampling")

    betas = pad_betas(np.asarray(data["betas"], dtype=np.float32), num_betas, raw_T)
    mesh_path, canonical_mesh_frame_index = choose_canonical_mesh(segment_dir / "Mesh", source_frame_indices)

    return {
        "root_orient": root_orient[source_frame_indices].astype(np.float32),
        "pose_body": pose_body[source_frame_indices].astype(np.float32),
        "trans": trans[source_frame_indices].astype(np.float32),
        "betas": betas[source_frame_indices].astype(np.float32),
        "obj_pose": obj_pose[source_frame_indices].astype(np.float32),
        "obj_origin_trans": obj_origin_trans[source_frame_indices].astype(np.float32),
        "source_frame_indices": source_frame_indices.astype(np.int64),
        "source_mocap_frame_rate": int(source_fps),
        "mocap_frame_rate": int(target_fps),
        "gender": str(gender).lower(),
        "canonical_mesh_path": mesh_path,
        "canonical_mesh_frame_index": int(canonical_mesh_frame_index),
    }


def compute_motion_features(
    bm: BodyModel,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    device: torch.device,
    chunk_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position_chunks: List[torch.Tensor] = []
    local_chunks: List[torch.Tensor] = []
    global_chunks: List[torch.Tensor] = []

    kintree_table = bm.kintree_table[0].long()[:22]
    kintree_table[0] = -1

    with torch.no_grad():
        for start, end in iter_slices(root_orient.shape[0], chunk_len):
            root_t = torch.tensor(root_orient[start:end], dtype=torch.float32, device=device)
            body_t = torch.tensor(pose_body[start:end], dtype=torch.float32, device=device)
            trans_t = torch.tensor(trans[start:end], dtype=torch.float32, device=device)
            betas_t = torch.tensor(betas[start:end], dtype=torch.float32, device=device)

            body_out = bm(
                root_orient=root_t,
                pose_body=body_t,
                trans=trans_t,
                betas=betas_t,
            )
            position_chunks.append(body_out.Jtr[:, :22, :].detach().cpu().float())

            pose_aa = torch.cat([root_t, body_t], dim=1)
            local_rot_mat = transforms.axis_angle_to_matrix(pose_aa.reshape(-1, 3)).reshape(
                pose_aa.shape[0], -1, 3, 3
            )
            local_chunks.append(transforms.matrix_to_rotation_6d(local_rot_mat).reshape(pose_aa.shape[0], -1).detach().cpu().float())
            global_chunks.append(
                local2global_pose(local_rot_mat.reshape(local_rot_mat.shape[0], -1, 9), kintree_table)
                .reshape(local_rot_mat.shape[0], -1, 3, 3)
                .detach()
                .cpu()
                .float()
            )

            del body_out, root_t, body_t, trans_t, betas_t, pose_aa, local_rot_mat
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return torch.cat(position_chunks, dim=0), torch.cat(local_chunks, dim=0), torch.cat(global_chunks, dim=0)


def apply_y_up_flip(seq: Dict[str, object], bm: BodyModel, device: torch.device) -> None:
    root_orient = seq["root_orient"]
    pose_body = seq["pose_body"]
    trans = seq["trans"]
    obj_pose = seq["obj_pose"]
    obj_origin_trans = seq["obj_origin_trans"]

    smpl_init_input = {
        "root_orient": torch.zeros_like(torch.from_numpy(root_orient.reshape(-1, 3)).to(device).float()),
        "pose_body": torch.zeros_like(torch.from_numpy(pose_body.reshape(-1, 63)).to(device).float()),
        "trans": torch.zeros_like(torch.from_numpy(trans.reshape(-1, 3)).to(device).float()),
    }
    with torch.no_grad():
        jtr_0 = np.asarray(bm(**smpl_init_input).Jtr[:, 0, :].cpu(), dtype=np.float32)

    r_y_up_torch = torch.from_numpy(R_Y_UP).float().to(device)
    r_y_up_repeat = r_y_up_torch.unsqueeze(0).repeat(root_orient.shape[0], 1, 1)

    root_orient_t = torch.from_numpy(root_orient).float().to(device)
    root_rot_mat = transforms.axis_angle_to_matrix(root_orient_t)
    root_rot_mat = torch.matmul(r_y_up_repeat, root_rot_mat)
    seq["root_orient"] = transforms.matrix_to_axis_angle(root_rot_mat).detach().cpu().numpy().astype(np.float32)
    seq["trans"] = ((trans + jtr_0) @ R_Y_UP.T - jtr_0).astype(np.float32)

    obj_pose_t = torch.from_numpy(obj_pose).float().to(device)
    obj_rot_mat = transforms.axis_angle_to_matrix(obj_pose_t)
    obj_rot_mat = torch.matmul(r_y_up_repeat, obj_rot_mat)
    seq["obj_pose"] = transforms.matrix_to_axis_angle(obj_rot_mat).detach().cpu().numpy().astype(np.float32)
    seq["obj_origin_trans"] = (obj_origin_trans @ R_Y_UP.T).astype(np.float32)


def body_mesh_floor_y(mesh_path: Path, y_up_flip: bool) -> float:
    vertices, _ = _load_mesh_arrays(str(mesh_path))
    if y_up_flip:
        vertices = vertices @ R_Y_UP.T
    return float(vertices[:, 1].min())


def estimate_body_mesh_y_bridge(
    segment_dir: Path,
    source_frame_indices: np.ndarray,
    position_global: torch.Tensor,
    y_up_flip: bool,
    sample_count: int,
) -> Tuple[float, int, float, float]:
    """Estimate the raw InterCap body-mesh to generated SMPL-H y-offset.

    InterCap object poses are in the same coordinate frame as Mesh/*_second.ply,
    while this preprocessor regenerates the human body with SMPL-H.  The two
    body surfaces have a stable vertical offset, so bridge the object frame to
    the generated-human frame without using object-ground assumptions.
    """
    if sample_count <= 0 or len(source_frame_indices) == 0:
        return 0.0, 0, float("nan"), float("nan")

    mesh_dir = segment_dir / "Mesh"
    sample_total = min(int(sample_count), len(source_frame_indices))
    sample_rows = np.unique(np.linspace(0, len(source_frame_indices) - 1, sample_total).round().astype(np.int64))
    smpl_floor_per_frame = position_global[:, [7, 8, 10, 11], 1].amin(dim=1).detach().cpu().numpy()

    deltas: List[float] = []
    raw_floors: List[float] = []
    smpl_floors: List[float] = []
    for row in sample_rows:
        body_mesh_path = body_mesh_path_for_frame(mesh_dir, int(source_frame_indices[int(row)]))
        if body_mesh_path is None:
            continue
        raw_floor = body_mesh_floor_y(body_mesh_path, y_up_flip=y_up_flip)
        smpl_floor = float(smpl_floor_per_frame[int(row)])
        raw_floors.append(raw_floor)
        smpl_floors.append(smpl_floor)
        deltas.append(smpl_floor - raw_floor)

    if not deltas:
        print(f"Warning: no InterCap body meshes available for y bridge under {mesh_dir}; using zero bridge")
        return 0.0, 0, float("nan"), float("nan")

    return (
        float(np.median(np.asarray(deltas, dtype=np.float32))),
        len(deltas),
        float(np.mean(np.asarray(raw_floors, dtype=np.float32))),
        float(np.mean(np.asarray(smpl_floors, dtype=np.float32))),
    )


def process_intercap_sequence(
    segment_dir: Path,
    bm_loader: BodyModelLoader,
    output_dir: Path,
    device: torch.device,
    chunk_len: int,
    obj_points_count: int,
    source_fps: int,
    target_fps: int,
    gender: str,
    max_frames: Optional[int],
    floor_align: bool,
    y_up_flip: bool,
    body_mesh_y_bridge: bool,
    body_bridge_sample_count: int,
    skip_existing: bool,
) -> Path:
    subject_id = segment_dir.parts[-3]
    object_id = segment_dir.parts[-2]
    segment_id = segment_dir.name
    seq_name = f"intercap_{subject_id}_{object_id}_{segment_id}"
    out_path = output_dir / f"{seq_name}.pt"
    if skip_existing and out_path.exists():
        return out_path

    seq = load_intercap_sequence(
        segment_dir=segment_dir,
        num_betas=bm_loader.num_betas,
        source_fps=source_fps,
        target_fps=target_fps,
        gender=gender,
        max_frames=max_frames,
    )
    bm = bm_loader.get(seq["gender"])
    if y_up_flip:
        apply_y_up_flip(seq, bm, device)
    T = int(seq["root_orient"].shape[0])

    position_global, rotation_local, rotation_global = compute_motion_features(
        bm=bm,
        root_orient=seq["root_orient"],
        pose_body=seq["pose_body"],
        trans=seq["trans"],
        betas=seq["betas"],
        device=device,
        chunk_len=max(1, int(chunk_len)),
    )

    mesh_path = Path(seq["canonical_mesh_path"])
    mesh_rel = relpath(mesh_path)
    mesh_vertices, _ = _load_mesh_arrays(str(mesh_path))
    obj_mesh_centroid = mesh_vertices.mean(axis=0).astype(np.float32)

    obj_rot = R.from_rotvec(seq["obj_pose"]).as_matrix().astype(np.float32)
    trans = seq["trans"].copy()
    obj_origin_trans = seq["obj_origin_trans"].copy()
    body_mesh_y_bridge_offset = 0.0
    body_mesh_y_bridge_samples = 0
    raw_body_mesh_floor_y = float("nan")
    smplh_foot_floor_y = float("nan")
    if body_mesh_y_bridge:
        (
            body_mesh_y_bridge_offset,
            body_mesh_y_bridge_samples,
            raw_body_mesh_floor_y,
            smplh_foot_floor_y,
        ) = estimate_body_mesh_y_bridge(
            segment_dir=segment_dir,
            source_frame_indices=seq["source_frame_indices"],
            position_global=position_global,
            y_up_flip=y_up_flip,
            sample_count=int(body_bridge_sample_count),
        )
        obj_origin_trans[:, 1] += body_mesh_y_bridge_offset

    obj_center_offset = np.einsum("tij,j->ti", obj_rot, obj_mesh_centroid).astype(np.float32)
    obj_trans = (obj_origin_trans + obj_center_offset).astype(np.float32)

    if floor_align:
        foot_floor_y = float(position_global[:, [7, 8, 10, 11], 1].min().item())
        position_global[:, :, 1] -= foot_floor_y
        trans[:, 1] -= foot_floor_y
        obj_origin_trans[:, 1] -= foot_floor_y
        obj_trans[:, 1] -= foot_floor_y

    lfoot_contact, rfoot_contact = compute_foot_contact_labels(position_global.to(device))

    obj_rot_t = torch.tensor(obj_rot, dtype=torch.float32, device=device)
    obj_trans_t = torch.tensor(obj_trans, dtype=torch.float32, device=device)
    obj_scale_t = torch.ones(T, dtype=torch.float32, device=device)
    lhand_contact, rhand_contact, obj_contact = compute_improved_contact_labels(
        obj_trans_t,
        obj_rot_t,
        obj_scale_t,
        position_global.to(device),
        str(mesh_path.parent),
        str(mesh_path),
        device,
        T,
    )

    canonical_points = load_canonical_points_from_mesh_path(
        str(mesh_path),
        sample_count=int(max(1, obj_points_count)),
        device="cpu",
    )

    output = {
        "seq_name": seq_name,
        "dataset": "InterCap",
        "subject_id": subject_id,
        "object_id": object_id,
        "segment_id": segment_id,
        "gender": seq["gender"],
        "source_mocap_frame_rate": seq["source_mocap_frame_rate"],
        "mocap_frame_rate": seq["mocap_frame_rate"],
        "source_frame_indices": torch.from_numpy(seq["source_frame_indices"]).long(),
        "rotation_local_full_gt_list": rotation_local.float(),
        "position_global_full_gt_world": position_global.float(),
        "rotation_global": rotation_global.float(),
        "trans": torch.from_numpy(trans).float(),
        "human_imu_real": None,
        "obj_imu_real": None,
        "lfoot_contact": lfoot_contact.cpu().float(),
        "rfoot_contact": rfoot_contact.cpu().float(),
        "betas": torch.from_numpy(seq["betas"]).float(),
        "obj_name": mesh_rel,
        "obj_mesh_name": mesh_path.stem,
        "obj_mesh_path": mesh_rel,
        "canonical_mesh_frame_index": int(seq["canonical_mesh_frame_index"]),
        "y_up_flip": bool(y_up_flip),
        "body_mesh_y_bridge": bool(body_mesh_y_bridge),
        "body_mesh_y_bridge_offset": float(body_mesh_y_bridge_offset),
        "body_mesh_y_bridge_samples": int(body_mesh_y_bridge_samples),
        "raw_body_mesh_floor_y": float(raw_body_mesh_floor_y),
        "smplh_foot_floor_y": float(smplh_foot_floor_y),
        "obj_points_canonical": canonical_points.half().cpu(),
        "obj_points_sample_count": int(canonical_points.shape[0]),
        "has_object": True,
        "obj_scale": torch.ones(T, dtype=torch.float32),
        "obj_trans": obj_trans_t.detach().cpu().float(),
        "obj_origin_trans": torch.from_numpy(obj_origin_trans).float(),
        "obj_mesh_centroid": torch.from_numpy(obj_mesh_centroid).float(),
        "obj_center_offset": torch.from_numpy(obj_center_offset).float(),
        "obj_rot": obj_rot_t.detach().cpu().float(),
        "obj_com_pos": obj_trans_t.detach().cpu().float(),
        "lhand_contact": lhand_contact.cpu().bool(),
        "rhand_contact": rhand_contact.cpu().bool(),
        "obj_contact": obj_contact.cpu().bool(),
        "source_files": [
            relpath(segment_dir / "res_2.pkl"),
            mesh_rel,
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess InterCap into IMUHOI-compatible pt files")
    parser.add_argument("--dataset_root", type=str, default="datasets/InterCap")
    parser.add_argument("--output_dir", type=str, default="process/processed_split_data_INTERCAP")
    parser.add_argument("--support_dir", type=str, default="datasets/smpl_models")
    parser.add_argument("--device", type=str, default='cuda', help="cpu/cuda override")
    parser.add_argument("--chunk_len", type=int, default=1024, help="SMPL forward chunk length")
    parser.add_argument("--obj_points_count", type=int, default=256)
    parser.add_argument("--num_betas", type=int, default=16)
    parser.add_argument("--source_fps", type=int, default=30)
    parser.add_argument("--target_fps", type=int, default=30)
    parser.add_argument("--gender", type=str, default="neutral")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--object_ids", nargs="*", default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_floor_align", action="store_true")
    parser.add_argument("--no_y_up_flip", action="store_true", help="Disable InterCap Y-up flip")
    parser.add_argument("--no_body_mesh_y_bridge", action="store_true", help="Disable InterCap raw body mesh to SMPL-H y bridge")
    parser.add_argument("--body_bridge_sample_count", type=int, default=16, help="Frames sampled to estimate the InterCap y bridge")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    bm_loader = BodyModelLoader(args.support_dir, device, num_betas=args.num_betas)
    segments = collect_intercap_segments(args.dataset_root)
    if args.subjects:
        subjects = set(args.subjects)
        segments = [path for path in segments if path.parts[-3] in subjects]
    if args.object_ids:
        object_ids = set(args.object_ids)
        segments = [path for path in segments if path.parts[-2] in object_ids]
    if args.max_sequences is not None:
        segments = segments[: int(args.max_sequences)]
    if not segments:
        raise FileNotFoundError(f"No InterCap segments found under {args.dataset_root}")

    processed_count = 0
    for segment_dir in tqdm(segments, desc="Processing InterCap sequences"):
        process_intercap_sequence(
            segment_dir=segment_dir,
            bm_loader=bm_loader,
            output_dir=output_root,
            device=device,
            chunk_len=args.chunk_len,
            obj_points_count=args.obj_points_count,
            source_fps=args.source_fps,
            target_fps=args.target_fps,
            gender=args.gender,
            max_frames=args.max_frames,
            floor_align=not args.no_floor_align,
            y_up_flip=not args.no_y_up_flip,
            body_mesh_y_bridge=not args.no_body_mesh_y_bridge,
            body_bridge_sample_count=args.body_bridge_sample_count,
            skip_existing=args.skip_existing,
        )
        processed_count += 1

    print("InterCap preprocessing finished")
    print(f"  processed: {processed_count} sequences -> {output_root}")
    print("  output layout: flat top-level .pt files, ready for process/split_dataset.py")


if __name__ == "__main__":
    main()
