import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import local2global_pose
import pytorch3d.transforms as transforms

_OBJECT_MESH_CENTROID_CACHE: Dict[Tuple[str, str], np.ndarray] = {}

try:
    from preprocess import (
        compute_foot_contact_labels,
        compute_improved_contact_labels,
        load_object_canonical_points,
        resolve_object_mesh_path,
    )
except ModuleNotFoundError:
    from process.preprocess import (
        compute_foot_contact_labels,
        compute_improved_contact_labels,
        load_object_canonical_points,
        resolve_object_mesh_path,
    )


def iter_slices(total: int, chunk_len: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_len):
        yield start, min(start + chunk_len, total)


def scalar_to_str(value, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 0:
        return default
    return str(arr.reshape(-1)[0])


def subject_from_sequence(seq_name: str) -> str:
    return seq_name.split("_", 1)[0]


def object_from_sequence(seq_name: str) -> str:
    if "_" not in seq_name:
        raise ValueError(f"Unable to infer HODome object name from sequence '{seq_name}'")
    return seq_name.split("_", 1)[1]


def load_object_mesh_centroid(obj_name: str, objects_root: str) -> np.ndarray:
    cache_key = (str(objects_root), str(obj_name))
    cached = _OBJECT_MESH_CENTROID_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    mesh_path = resolve_object_mesh_path(obj_name, objects_root)
    if mesh_path is None:
        print(f"Warning: missing HODome object mesh for {obj_name}; using zero centroid")
        centroid = np.zeros(3, dtype=np.float32)
    else:
        mesh = trimesh.load_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            print(f"Warning: invalid HODome object mesh vertices for {mesh_path}; using zero centroid")
            centroid = np.zeros(3, dtype=np.float32)
        else:
            centroid = vertices.mean(axis=0).astype(np.float32)

    _OBJECT_MESH_CENTROID_CACHE[cache_key] = centroid
    return centroid.copy()


def build_downsample_indices(total_frames: int, source_fps: int, target_fps: int) -> np.ndarray:
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")
    if source_fps == target_fps:
        return np.arange(total_frames, dtype=np.int64)
    if source_fps < target_fps:
        raise ValueError(f"HODome source fps {source_fps} is lower than target fps {target_fps}")

    step_float = float(source_fps) / float(target_fps)
    step = int(round(step_float))
    if step <= 0 or abs(step_float - step) > 1e-6:
        raise ValueError(f"HODome fps conversion {source_fps}->{target_fps} is not an integer stride")
    return np.arange(0, total_frames, step, dtype=np.int64)


def pad_betas(betas: np.ndarray, target_dim: int, target_frames: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 3:
        betas = betas[:, 0, :]
    elif betas.ndim == 1:
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


def collect_hodome_pairs(mocap_ground_dir: str) -> List[Tuple[str, Path, Path]]:
    root = Path(mocap_ground_dir)
    pairs: List[Tuple[str, Path, Path]] = []
    for human_path in sorted(root.glob("*_human.npz")):
        seq_name = human_path.name[:-len("_human.npz")]
        object_path = root / f"{seq_name}_object.npz"
        if object_path.exists():
            pairs.append((seq_name, human_path, object_path))
        else:
            print(f"Warning: missing object npz for {seq_name}, skipped")
    return pairs


def load_hodome_npz_pair(
    seq_name: str,
    human_path: Path,
    object_path: Path,
    num_betas: int,
    target_fps: int = 30,
    max_frames: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    human = np.load(human_path, allow_pickle=True)
    obj = np.load(object_path, allow_pickle=True)

    poses = np.asarray(human["poses"], dtype=np.float32)
    if poses.ndim == 3:
        poses = poses[:, 0, :]
    poses = poses.reshape(poses.shape[0], -1)

    root_orient = np.asarray(human["Rh"], dtype=np.float32).reshape(poses.shape[0], -1, 3)[:, 0, :]
    trans = np.asarray(human["Th"], dtype=np.float32).reshape(poses.shape[0], -1, 3)[:, 0, :]

    obj_rot = np.asarray(obj["object_R"], dtype=np.float32).reshape(obj["object_R"].shape[0], 3, 3)
    obj_trans = np.asarray(obj["object_T"], dtype=np.float32).reshape(obj["object_T"].shape[0], -1, 3)[:, 0, :]

    source_fps = int(np.asarray(human["mocap_frame_rate"]).item())
    object_source_fps = int(np.asarray(obj["mocap_frame_rate"]).item())
    if object_source_fps != source_fps:
        raise ValueError(
            f"Sequence {seq_name} has mismatched human/object fps: {source_fps} vs {object_source_fps}"
        )

    raw_T = min(poses.shape[0], root_orient.shape[0], trans.shape[0], obj_rot.shape[0], obj_trans.shape[0])
    source_frame_indices = build_downsample_indices(raw_T, source_fps, int(target_fps))
    if max_frames is not None:
        source_frame_indices = source_frame_indices[: int(max_frames)]
    T = int(source_frame_indices.shape[0])
    if T <= 0:
        raise ValueError(f"Sequence {seq_name} has no valid frames")

    poses = poses[source_frame_indices]
    root_orient = root_orient[source_frame_indices]
    trans = trans[source_frame_indices]
    obj_rot = obj_rot[source_frame_indices]
    obj_trans = obj_trans[source_frame_indices]
    raw_betas = pad_betas(np.asarray(human["betas"], dtype=np.float32), num_betas, raw_T)
    betas = raw_betas[source_frame_indices]

    if poses.shape[1] < 66:
        raise ValueError(f"Sequence {seq_name} has pose dim {poses.shape[1]}, expected at least 66")

    return {
        "seq_name": seq_name,
        "gender": scalar_to_str(human["gender"], default="neutral"),
        "source_mocap_frame_rate": source_fps,
        "mocap_frame_rate": int(target_fps),
        "source_frame_indices": source_frame_indices.astype(np.int64),
        "root_orient": root_orient.astype(np.float32),
        "pose_body": poses[:, 3:66].astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": betas,
        "obj_rot": obj_rot.astype(np.float32),
        "obj_trans": obj_trans.astype(np.float32),
        "obj_name": object_from_sequence(seq_name),
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
            root_chunk = root_orient[start:end].astype(np.float32)
            body_chunk = pose_body[start:end].astype(np.float32)
            trans_chunk = trans[start:end].astype(np.float32)
            betas_chunk = betas[start:end].astype(np.float32)

            root_t = torch.tensor(root_chunk, dtype=torch.float32, device=device)
            body_t = torch.tensor(body_chunk, dtype=torch.float32, device=device)
            trans_t = torch.tensor(trans_chunk, dtype=torch.float32, device=device)
            betas_t = torch.tensor(betas_chunk, dtype=torch.float32, device=device)

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
            rotation_local_6d = transforms.matrix_to_rotation_6d(local_rot_mat).reshape(pose_aa.shape[0], -1)
            rotation_global = local2global_pose(
                local_rot_mat.reshape(local_rot_mat.shape[0], -1, 9),
                kintree_table,
            ).reshape(local_rot_mat.shape[0], -1, 3, 3)

            local_chunks.append(rotation_local_6d.detach().cpu().float())
            global_chunks.append(rotation_global.detach().cpu().float())

            del body_out, root_t, body_t, trans_t, betas_t, pose_aa, local_rot_mat
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return (
        torch.cat(position_chunks, dim=0),
        torch.cat(local_chunks, dim=0),
        torch.cat(global_chunks, dim=0),
    )


def process_hodome_sequence(
    seq_name: str,
    human_path: Path,
    object_path: Path,
    bm_loader: BodyModelLoader,
    objects_root: str,
    output_dir: Path,
    device: torch.device,
    chunk_len: int,
    obj_points_count: int,
    target_fps: int = 30,
    max_frames: Optional[int] = None,
    floor_align: bool = True,
    skip_existing: bool = False,
) -> Path:
    out_path = output_dir / f"{seq_name}.pt"
    if skip_existing and out_path.exists():
        return out_path

    seq = load_hodome_npz_pair(
        seq_name,
        human_path,
        object_path,
        num_betas=bm_loader.num_betas,
        target_fps=target_fps,
        max_frames=max_frames,
    )
    bm = bm_loader.get(seq["gender"])
    T = seq["root_orient"].shape[0]

    position_global, rotation_local, rotation_global = compute_motion_features(
        bm=bm,
        root_orient=seq["root_orient"],
        pose_body=seq["pose_body"],
        trans=seq["trans"],
        betas=seq["betas"],
        device=device,
        chunk_len=chunk_len,
    )

    trans = seq["trans"].copy()
    obj_origin_trans = seq["obj_trans"].copy()
    obj_mesh_centroid = load_object_mesh_centroid(seq["obj_name"], objects_root)
    obj_center_offset = np.einsum("tij,j->ti", seq["obj_rot"], obj_mesh_centroid).astype(np.float32)
    obj_trans = (obj_origin_trans + obj_center_offset).astype(np.float32)
    if floor_align:
        foot_floor_y = float(position_global[:, [7, 8, 10, 11], 1].min().item())
        position_global[:, :, 1] -= foot_floor_y
        trans[:, 1] -= foot_floor_y
        obj_origin_trans[:, 1] -= foot_floor_y
        obj_trans[:, 1] -= foot_floor_y

    lfoot_contact, rfoot_contact = compute_foot_contact_labels(position_global.to(device))

    obj_rot_t = torch.tensor(seq["obj_rot"], dtype=torch.float32, device=device)
    obj_trans_t = torch.tensor(obj_trans, dtype=torch.float32, device=device)
    obj_scale_t = torch.ones(T, dtype=torch.float32, device=device)
    lhand_contact, rhand_contact, obj_contact = compute_improved_contact_labels(
        obj_trans_t,
        obj_rot_t,
        obj_scale_t,
        position_global.to(device),
        objects_root,
        seq["obj_name"],
        device,
        T,
    )

    canonical_points = load_object_canonical_points(
        seq["obj_name"],
        obj_geo_root=objects_root,
        sample_count=int(max(1, obj_points_count)),
        device="cpu",
    )
    if canonical_points is None:
        canonical_points = torch.zeros(int(max(1, obj_points_count)), 3, dtype=torch.float32)

    mesh_path = resolve_object_mesh_path(seq["obj_name"], objects_root)
    mesh_name = Path(mesh_path).stem if mesh_path else seq["obj_name"]

    output = {
        "seq_name": seq_name,
        "dataset": "HODome",
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
        "obj_name": seq["obj_name"],
        "obj_mesh_name": mesh_name,
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
        "source_pkls": [human_path.name, object_path.name],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess HODome mocap_ground npz files into IMUHOI pt files")
    parser.add_argument("--mocap_ground_dir", type=str, default="datasets/HODome/mocap_ground")
    parser.add_argument("--objects_root", type=str, default="datasets/HODome/scaned_object")
    parser.add_argument("--output_dir", type=str, default="process/processed_split_data_HODOME")
    parser.add_argument("--support_dir", type=str, default="datasets/smpl_models")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda override")
    parser.add_argument("--chunk_len", type=int, default=1024, help="SMPL forward chunk length")
    parser.add_argument("--obj_points_count", type=int, default=256)
    parser.add_argument("--num_betas", type=int, default=16)
    parser.add_argument("--target_fps", type=int, default=30, help="Output fps; HODome mocap_ground is 60fps")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject filter")
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_floor_align", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    bm_loader = BodyModelLoader(args.support_dir, device, num_betas=args.num_betas)
    pairs = collect_hodome_pairs(args.mocap_ground_dir)
    if args.subjects:
        subjects = set(args.subjects)
        pairs = [item for item in pairs if subject_from_sequence(item[0]) in subjects]
    if args.max_sequences is not None:
        pairs = pairs[: int(args.max_sequences)]
    if not pairs:
        raise FileNotFoundError(f"No HODome human/object npz pairs found under {args.mocap_ground_dir}")

    processed_count = 0
    for seq_name, human_path, object_path in tqdm(pairs, desc="Processing HODome sequences"):
        process_hodome_sequence(
            seq_name=seq_name,
            human_path=human_path,
            object_path=object_path,
            bm_loader=bm_loader,
            objects_root=args.objects_root,
            output_dir=output_root,
            device=device,
            chunk_len=max(1, int(args.chunk_len)),
            obj_points_count=args.obj_points_count,
            target_fps=args.target_fps,
            max_frames=args.max_frames,
            floor_align=not args.no_floor_align,
            skip_existing=args.skip_existing,
        )
        processed_count += 1

    print("HODome preprocessing finished")
    print(f"  processed: {processed_count} sequences -> {output_root}")
    print("  output layout: flat top-level .pt files, ready for process/split_dataset.py")


if __name__ == "__main__":
    main()
