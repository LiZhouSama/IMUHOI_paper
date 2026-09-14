import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from human_body_prior.body_model.body_model import BodyModel
import pytorch3d.transforms as transforms
from utils.human_pose import (
    build_smpl24_joint_positions,
    ensure_smpl24_local_rotations,
    local_rotations_to_global,
)

_OBJECT_MESH_CENTROID_CACHE: Dict[Tuple[str, str], np.ndarray] = {}
CANONICAL_BODY_MODEL_TYPE = "smplh"
CANONICAL_BODY_MODEL_GENDER = "neutral"
CANONICAL_BETAS = "zero"
CANONICAL_SKELETON_VERSION = 1
CANONICAL_BODY_MODEL_RELATIVE_PATH = os.path.join("smplh", "neutral", "model.npz")

try:
    from preprocess import (
        add_contact_postprocess_arguments,
        build_unheld_object_motion_skip_record,
        contact_postprocess_metadata,
        compute_foot_contact_labels,
        compute_improved_contact_labels,
        load_object_canonical_points,
        precompute_sequence_features,
        postprocess_sequence_contact_labels,
        resolve_object_mesh_path,
        saved_contact_postprocess_matches,
        write_contact_skip_manifest,
    )
except ModuleNotFoundError:
    from process.preprocess import (
        add_contact_postprocess_arguments,
        build_unheld_object_motion_skip_record,
        contact_postprocess_metadata,
        compute_foot_contact_labels,
        compute_improved_contact_labels,
        load_object_canonical_points,
        precompute_sequence_features,
        postprocess_sequence_contact_labels,
        resolve_object_mesh_path,
        saved_contact_postprocess_matches,
        write_contact_skip_manifest,
    )


def iter_slices(total: int, chunk_len: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_len):
        yield start, min(start + chunk_len, total)


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
            raise ValueError(f"HODome preprocessing only supports {CANONICAL_BODY_MODEL_TYPE}")
        return self._model

    def resolve(self, seq_data=None, gender=None, model_type=None) -> Tuple[BodyModel, str]:
        return self.get(gender=gender, model_type=model_type), CANONICAL_BODY_MODEL_TYPE


def canonical_body_model_metadata() -> Dict[str, object]:
    return {
        "canonical_body_model_gender": CANONICAL_BODY_MODEL_GENDER,
        "canonical_betas": CANONICAL_BETAS,
        "canonical_skeleton_version": CANONICAL_SKELETON_VERSION,
    }


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
    if poses.shape[1] < 66:
        raise ValueError(f"Sequence {seq_name} has pose dim {poses.shape[1]}, expected at least 66")

    return {
        "seq_name": seq_name,
        "gender": CANONICAL_BODY_MODEL_GENDER,
        "source_mocap_frame_rate": source_fps,
        "mocap_frame_rate": int(target_fps),
        "source_frame_indices": source_frame_indices.astype(np.int64),
        "root_orient": root_orient.astype(np.float32),
        "pose_body": poses[:, 3:66].astype(np.float32),
        "trans": trans.astype(np.float32),
        "obj_rot": obj_rot.astype(np.float32),
        "obj_trans": obj_trans.astype(np.float32),
        "obj_name": object_from_sequence(seq_name),
    }


def compute_motion_features(
    bm: BodyModel,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    device: torch.device,
    chunk_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position_chunks: List[torch.Tensor] = []
    local_chunks: List[torch.Tensor] = []
    global_chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for start, end in iter_slices(root_orient.shape[0], chunk_len):
            root_chunk = root_orient[start:end].astype(np.float32)
            body_chunk = pose_body[start:end].astype(np.float32)
            trans_chunk = trans[start:end].astype(np.float32)

            root_t = torch.tensor(root_chunk, dtype=torch.float32, device=device)
            body_t = torch.tensor(body_chunk, dtype=torch.float32, device=device)
            trans_t = torch.tensor(trans_chunk, dtype=torch.float32, device=device)

            body_out = bm(
                root_orient=root_t,
                pose_body=body_t,
                trans=trans_t,
            )
            position_chunks.append(
                build_smpl24_joint_positions(body_out.Jtr).detach().cpu().float()
            )

            pose_aa = torch.cat([root_t, body_t], dim=1)
            local_rot_mat = transforms.axis_angle_to_matrix(pose_aa.reshape(-1, 3)).reshape(
                pose_aa.shape[0], -1, 3, 3
            )
            local_rot_mat = ensure_smpl24_local_rotations(local_rot_mat)
            pose_aa = pose_aa.reshape(pose_aa.shape[0], -1, 3)
            pose_aa = torch.cat(
                (pose_aa, torch.zeros(pose_aa.shape[0], 2, 3, device=device)),
                dim=1,
            )
            rotation_global = local_rotations_to_global(local_rot_mat)

            local_chunks.append(pose_aa.detach().cpu().float())
            global_chunks.append(rotation_global.detach().cpu().float())

            del body_out, root_t, body_t, trans_t, pose_aa, local_rot_mat
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
    contact_max_hole_frames: int = 30,
    contact_min_duration_frames: int = 30,
    skip_object_displacement_without_hand_contact: bool = True,
    unheld_object_displacement_threshold_m: float = 0.10,
    skipped_sequences=None,
) -> Optional[Path]:
    out_path = output_dir / f"{seq_name}.pt"
    contact_metadata_kwargs = {
        "max_hole_frames": contact_max_hole_frames,
        "min_contact_frames": contact_min_duration_frames,
        "skip_object_displacement_without_hand_contact": (
            skip_object_displacement_without_hand_contact
        ),
        "unheld_object_displacement_threshold_m": (
            unheld_object_displacement_threshold_m
        ),
    }
    if (
        skip_existing
        and saved_contact_postprocess_matches(
            str(out_path),
            **contact_metadata_kwargs,
        )
    ):
        return out_path

    seq = load_hodome_npz_pair(
        seq_name,
        human_path,
        object_path,
        target_fps=target_fps,
        max_frames=max_frames,
    )
    bm, body_model_type = bm_loader.resolve(model_type=CANONICAL_BODY_MODEL_TYPE)
    T = seq["root_orient"].shape[0]

    position_global, rotation_local, rotation_global = compute_motion_features(
        bm=bm,
        root_orient=seq["root_orient"],
        pose_body=seq["pose_body"],
        trans=seq["trans"],
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
    lhand_contact_raw, rhand_contact_raw, obj_contact_raw = compute_improved_contact_labels(
        obj_trans_t,
        obj_rot_t,
        obj_scale_t,
        position_global.to(device),
        objects_root,
        seq["obj_name"],
        device,
        T,
    )
    (
        lhand_contact,
        rhand_contact,
        obj_contact,
        unheld_motion_interval,
    ) = postprocess_sequence_contact_labels(
        lhand_contact_raw,
        rhand_contact_raw,
        obj_contact_raw,
        obj_trans_t,
        **contact_metadata_kwargs,
    )
    if unheld_motion_interval is not None:
        if skipped_sequences is not None:
            skipped_sequences.append(build_unheld_object_motion_skip_record(
                dataset="HODome",
                seq_name=seq_name,
                output_file=out_path.name,
                obj_name=seq["obj_name"],
                obj_trans=obj_trans_t,
                lhand_contact_raw=lhand_contact_raw,
                rhand_contact_raw=rhand_contact_raw,
                obj_contact_raw=obj_contact_raw,
                lhand_contact=lhand_contact,
                rhand_contact=rhand_contact,
                obj_contact=obj_contact,
                unheld_motion_interval=unheld_motion_interval,
            ))
        if out_path.exists():
            out_path.unlink()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return None

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
        "gender": CANONICAL_BODY_MODEL_GENDER,
        "body_model_type": body_model_type,
        **canonical_body_model_metadata(),
        "source_mocap_frame_rate": seq["source_mocap_frame_rate"],
        "mocap_frame_rate": seq["mocap_frame_rate"],
        "source_frame_indices": torch.from_numpy(seq["source_frame_indices"]).long(),
        "pose": rotation_local.reshape(T, -1).float(),
        "position_global_full_gt_world": position_global.float(),
        "rotation_global": rotation_global.float(),
        "trans": torch.from_numpy(trans).float(),
        "human_imu_real": None,
        "obj_imu_real": None,
        "lfoot_contact": lfoot_contact.cpu().float(),
        "rfoot_contact": rfoot_contact.cpu().float(),
        "obj_name": seq["obj_name"],
        "obj_mesh_name": mesh_name,
        "obj_points_canonical": canonical_points.half().cpu(),
        "has_object": True,
        "obj_scale": torch.ones(T, dtype=torch.float32),
        "obj_trans": obj_trans_t.detach().cpu().float(),
        "obj_origin_trans": torch.from_numpy(obj_origin_trans).float(),
        "obj_mesh_centroid": torch.from_numpy(obj_mesh_centroid).float(),
        "obj_center_offset": torch.from_numpy(obj_center_offset).float(),
        "obj_rot": obj_rot_t.detach().cpu().float(),
        "lhand_contact": lhand_contact.cpu().bool(),
        "rhand_contact": rhand_contact.cpu().bool(),
        "obj_contact": obj_contact.cpu().bool(),
        "lhand_contact_raw": lhand_contact_raw.cpu().bool(),
        "rhand_contact_raw": rhand_contact_raw.cpu().bool(),
        "obj_contact_raw": obj_contact_raw.cpu().bool(),
        "source_pkls": [human_path.name, object_path.name],
        **contact_postprocess_metadata(**contact_metadata_kwargs),
    }
    output.update(
        precompute_sequence_features(
            position_global,
            rotation_global,
            torch.from_numpy(trans).float(),
            obj_trans=obj_trans_t,
            obj_rot=obj_rot_t,
            lhand_contact=lhand_contact,
            rhand_contact=rhand_contact,
            obj_contact=obj_contact,
            has_object=True,
        )
    )

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
    add_contact_postprocess_arguments(parser)
    parser.set_defaults(skip_object_displacement_without_hand_contact=False)
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
    skipped_sequences = []
    for seq_name, human_path, object_path in tqdm(pairs, desc="Processing HODome sequences"):
        output_path = process_hodome_sequence(
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
        str(output_root),
        dataset="HODome",
        source_sequences=len(pairs),
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

    print("HODome preprocessing finished")
    print(f"  processed: {processed_count} sequences -> {output_root}")
    print(f"  skipped: {len(skipped_sequences)} sequences")
    print(f"  skip manifest: {manifest_path}")
    print("  output layout: flat top-level .pt files, ready for process/split_dataset.py")


if __name__ == "__main__":
    main()
