import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import re
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from utils.human_pose import (
    CANONICAL_BODY_MODEL_GENDER,
    GenderedBodyModelLoader,
    build_smpl24_joint_positions,
    canonical_skeleton_metadata,
    ensure_smpl24_local_rotations,
    local_rotations_to_global,
    normalize_body_model_type,
)

from human_body_prior.body_model.body_model import BodyModel
import pytorch3d.transforms as transforms

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
        split_unified_by_json,
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
        split_unified_by_json,
        write_contact_skip_manifest,
    )


FBX_BINARY_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
FBX_TICKS_PER_SECOND = 46186158000
FRAME_LINE_RE = re.compile(
    r"^(?P<seq>\d+_\d+):(?P<frames>\d+)\s+"
    r"(?P<object>.*?)\s+"
    r"(?P<weight>heavy|medium|light)\s+"
    r"(?P<action>\S+)\s*$"
)


@dataclass
class PAHOISequence:
    seq_name: str
    subject_id: str
    concat_npz_path: Path
    raw_npz_path: Optional[Path]
    object_fbx_path: Path
    worldpos_npy_path: Optional[Path]
    worldpos_csv_path: Optional[Path]


@dataclass
class FbxNode:
    name: str
    properties: List[object]
    children: List["FbxNode"]


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


class BodyModelLoader(GenderedBodyModelLoader):
    """Load only the neutral, zero-shape SMPL-X canonical body model."""

    def __init__(self, support_dir: str, device: torch.device, num_betas: int = 16) -> None:
        super().__init__(support_dir, device=device, num_betas=num_betas, default_type="smplx")

    @staticmethod
    def _require_smplx(model_type: Optional[str]) -> str:
        resolved = normalize_body_model_type(model_type) if model_type is not None else "smplx"
        if resolved != "smplx":
            raise ValueError(f"PAHOI canonical preprocessing requires SMPL-X, got {model_type!r}")
        return resolved

    def get(self, gender=None, model_type=None):
        """Ignore source gender and return the unique neutral SMPL-X instance."""
        self._require_smplx(model_type)
        return self.registry.get_canonical("smplx")

    def resolve(self, seq_data=None, gender=None, model_type=None):
        """Validate the source family while fixing the canonical model to neutral SMPL-X."""
        if model_type is None and seq_data is not None:
            model_type = self.registry.infer_type(
                seq_data,
                gender=CANONICAL_BODY_MODEL_GENDER,
            )
        resolved = self._require_smplx(model_type)
        self._model_type = resolved
        return self.get(model_type=resolved), resolved


def collect_pahoi_sequences(dataset_root: str) -> List[PAHOISequence]:
    mocap_root = Path(dataset_root) / "Mocap_data"
    fbx_root = mocap_root / "cap_res_fbx"
    if not mocap_root.exists():
        raise FileNotFoundError(f"PAHOI Mocap_data directory not found: {mocap_root}")
    if not fbx_root.exists():
        raise FileNotFoundError(f"PAHOI cap_res_fbx directory not found: {fbx_root}")

    sequences: List[PAHOISequence] = []
    for subject_label in ("s1", "s2"):
        subject_dir = mocap_root / f"cap_res_bvh_{subject_label}"
        if not subject_dir.exists():
            raise FileNotFoundError(f"PAHOI subject directory not found: {subject_dir}")

        for seq_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            seq_name = seq_dir.name
            concat_npz = seq_dir / f"{seq_name}_reg_finger_concat.npz"
            raw_npz = seq_dir / f"{seq_name}_reg_finger.npz"
            object_fbx = fbx_root / f"{seq_name}_o.fbx"
            worldpos_npy = seq_dir / f"{seq_name}.npy"
            worldpos_csv = seq_dir / f"{seq_name}_worldpos.csv"
            for required in (concat_npz, object_fbx):
                if not required.exists():
                    raise FileNotFoundError(f"Missing PAHOI sequence file: {required}")
            sequences.append(
                PAHOISequence(
                    seq_name=seq_name,
                    subject_id=seq_name.split("_", 1)[0],
                    concat_npz_path=concat_npz,
                    raw_npz_path=raw_npz if raw_npz.exists() else None,
                    object_fbx_path=object_fbx,
                    worldpos_npy_path=worldpos_npy if worldpos_npy.exists() else None,
                    worldpos_csv_path=worldpos_csv if worldpos_csv.exists() else None,
                )
            )
    return sequences


def parse_sequence_metadata(dataset_root: str) -> Dict[str, Dict[str, object]]:
    mocap_root = Path(dataset_root) / "Mocap_data"
    metadata: Dict[str, Dict[str, object]] = {}
    for filename in ("frames.txt", "frame_sub2.txt"):
        path = mocap_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing PAHOI frame metadata file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                match = FRAME_LINE_RE.match(text)
                if match is None:
                    raise ValueError(f"Unable to parse {path}:{line_no}: {line.rstrip()}")
                seq_name = match.group("seq")
                item = {
                    "expected_frame_count": int(match.group("frames")),
                    "object_label": match.group("object").strip(),
                    "object_weight": match.group("weight").strip(),
                    "action": match.group("action").strip(),
                    "metadata_source": filename,
                }
                previous = metadata.get(seq_name)
                if previous is not None:
                    comparable_previous = {
                        key: previous[key]
                        for key in ("expected_frame_count", "object_label", "object_weight", "action")
                    }
                    comparable_item = {
                        key: item[key]
                        for key in ("expected_frame_count", "object_label", "object_weight", "action")
                    }
                    if comparable_previous != comparable_item:
                        raise ValueError(
                            f"Conflicting PAHOI metadata for {seq_name}: {previous} vs {item}"
                        )
                    continue
                metadata[seq_name] = item
    return metadata


def _fbx_node_header_size(version: int) -> int:
    return 25 if version >= 7500 else 13


def _is_null_fbx_record(data: bytes, offset: int, version: int) -> bool:
    header_size = _fbx_node_header_size(version)
    if offset + header_size > len(data):
        return True
    return data[offset:offset + header_size] == b"\x00" * header_size


def _read_fbx_property(data: bytes, offset: int) -> Tuple[object, int]:
    prop_type = chr(data[offset])
    offset += 1

    if prop_type == "Y":
        return struct.unpack_from("<h", data, offset)[0], offset + 2
    if prop_type == "C":
        return bool(struct.unpack_from("<?", data, offset)[0]), offset + 1
    if prop_type == "I":
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    if prop_type == "F":
        return struct.unpack_from("<f", data, offset)[0], offset + 4
    if prop_type == "D":
        return struct.unpack_from("<d", data, offset)[0], offset + 8
    if prop_type == "L":
        return struct.unpack_from("<q", data, offset)[0], offset + 8
    if prop_type in ("S", "R"):
        length = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        raw = data[offset:offset + length]
        offset += length
        if prop_type == "S":
            return raw.decode("utf-8", errors="ignore"), offset
        return raw, offset
    if prop_type in ("f", "d", "l", "i", "b"):
        array_len, encoding, encoded_len = struct.unpack_from("<III", data, offset)
        offset += 12
        raw = data[offset:offset + encoded_len]
        offset += encoded_len
        if encoding == 1:
            raw = zlib.decompress(raw)
        elif encoding != 0:
            raise ValueError(f"Unsupported FBX array encoding: {encoding}")

        dtype_map = {
            "f": np.dtype("<f4"),
            "d": np.dtype("<f8"),
            "l": np.dtype("<i8"),
            "i": np.dtype("<i4"),
            "b": np.dtype("u1"),
        }
        array = np.frombuffer(raw, dtype=dtype_map[prop_type], count=array_len).copy()
        if prop_type == "b":
            array = array.astype(bool)
        return array, offset

    raise ValueError(f"Unsupported FBX property type: {prop_type}")


def _parse_fbx_node(data: bytes, offset: int, version: int) -> Tuple[Optional[FbxNode], int]:
    header_size = _fbx_node_header_size(version)
    if offset + header_size > len(data) or _is_null_fbx_record(data, offset, version):
        return None, offset + header_size

    if version >= 7500:
        end_offset, num_props, prop_list_len, name_len = struct.unpack_from("<QQQB", data, offset)
    else:
        end_offset, num_props, prop_list_len, name_len = struct.unpack_from("<IIIB", data, offset)
    offset += header_size
    if end_offset == 0:
        return None, offset

    name = data[offset:offset + name_len].decode("utf-8", errors="ignore")
    offset += name_len

    props_start = offset
    properties: List[object] = []
    for _ in range(num_props):
        prop, offset = _read_fbx_property(data, offset)
        properties.append(prop)
    props_end = props_start + int(prop_list_len)
    if offset != props_end:
        offset = props_end

    children: List[FbxNode] = []
    while offset < end_offset:
        if _is_null_fbx_record(data, offset, version):
            offset += header_size
            break
        child, offset = _parse_fbx_node(data, offset, version)
        if child is not None:
            children.append(child)

    return FbxNode(name=name, properties=properties, children=children), int(end_offset)


def _parse_binary_fbx(data: bytes) -> List[FbxNode]:
    if not data.startswith(FBX_BINARY_MAGIC):
        raise ValueError("Only binary FBX files are supported")
    version = struct.unpack_from("<I", data, 23)[0]
    offset = 27
    nodes: List[FbxNode] = []
    while offset < len(data):
        if _is_null_fbx_record(data, offset, version):
            break
        node, offset = _parse_fbx_node(data, offset, version)
        if node is None:
            break
        nodes.append(node)
    return nodes


def _walk_fbx_nodes(nodes: Iterable[FbxNode]) -> Iterable[FbxNode]:
    for node in nodes:
        yield node
        yield from _walk_fbx_nodes(node.children)


def _find_child(node: FbxNode, name: str) -> Optional[FbxNode]:
    for child in node.children:
        if child.name == name:
            return child
    return None


def _property_as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _extract_animation_curves(nodes: List[FbxNode]) -> Dict[int, Dict[str, np.ndarray]]:
    curves: Dict[int, Dict[str, np.ndarray]] = {}
    for node in _walk_fbx_nodes(nodes):
        if node.name != "AnimationCurve" or not node.properties:
            continue
        curve_id = int(node.properties[0])
        values_node = _find_child(node, "KeyValueFloat")
        times_node = _find_child(node, "KeyTime")
        if values_node is None or not values_node.properties:
            raise ValueError(f"AnimationCurve {curve_id} has no KeyValueFloat")
        values = np.asarray(values_node.properties[0], dtype=np.float32).reshape(-1)
        if times_node is not None and times_node.properties:
            times = np.asarray(times_node.properties[0], dtype=np.int64).reshape(-1)
            if times.shape[0] != values.shape[0]:
                raise ValueError(f"AnimationCurve {curve_id} KeyTime/KeyValueFloat length mismatch")
        else:
            times = np.arange(values.shape[0], dtype=np.int64)
        curves[curve_id] = {"values": values, "times": times}
    if not curves:
        raise ValueError("No AnimationCurve nodes found in object FBX")
    return curves


def _extract_fbx_connections(nodes: List[FbxNode]) -> List[Tuple[str, int, int, str]]:
    connections: List[Tuple[str, int, int, str]] = []
    for node in _walk_fbx_nodes(nodes):
        if node.name != "C" or len(node.properties) < 3:
            continue
        relation = _property_as_str(node.properties[0])
        src = int(node.properties[1])
        dst = int(node.properties[2])
        prop_name = _property_as_str(node.properties[3]) if len(node.properties) > 3 else ""
        connections.append((relation, src, dst, prop_name))
    return connections


def parse_object_fbx_animation(fbx_path: Path, rotation_order: str = "xyz") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = fbx_path.read_bytes()
    nodes = _parse_binary_fbx(data)
    curves = _extract_animation_curves(nodes)
    connections = _extract_fbx_connections(nodes)

    curve_node_channel: Dict[int, str] = {}
    curve_to_node_axis: Dict[int, Tuple[int, str]] = {}
    for _, src, dst, prop_name in connections:
        if prop_name in ("Lcl Translation", "Lcl Rotation"):
            curve_node_channel[src] = prop_name
        elif src in curves and prop_name.startswith("d|"):
            axis = prop_name.split("|", 1)[1].upper()
            curve_to_node_axis[src] = (dst, axis)

    translation_axes: Dict[str, np.ndarray] = {}
    rotation_axes: Dict[str, np.ndarray] = {}
    axis_names = {"X", "Y", "Z"}
    for curve_id, (curve_node_id, axis) in curve_to_node_axis.items():
        if axis not in axis_names:
            continue
        channel = curve_node_channel.get(curve_node_id)
        if channel == "Lcl Translation":
            translation_axes[axis] = curves[curve_id]["values"]
        elif channel == "Lcl Rotation":
            rotation_axes[axis] = curves[curve_id]["values"]

    missing_translation = axis_names - set(translation_axes)
    missing_rotation = axis_names - set(rotation_axes)
    if missing_translation or missing_rotation:
        raise ValueError(
            f"FBX {fbx_path} missing animation curves: "
            f"translation={sorted(missing_translation)}, rotation={sorted(missing_rotation)}"
        )

    lengths = {arr.shape[0] for arr in list(translation_axes.values()) + list(rotation_axes.values())}
    if len(lengths) != 1:
        raise ValueError(f"FBX {fbx_path} animation curve lengths are inconsistent: {sorted(lengths)}")
    T = int(lengths.pop())
    if T <= 0:
        raise ValueError(f"FBX {fbx_path} contains no object animation frames")

    translation_cm = np.stack([translation_axes[axis] for axis in ("X", "Y", "Z")], axis=1)
    rotation_deg = np.stack([rotation_axes[axis] for axis in ("X", "Y", "Z")], axis=1)
    translation_m = (translation_cm * 0.01).astype(np.float32)
    rotation_mat = Rotation.from_euler(rotation_order, rotation_deg, degrees=True).as_matrix().astype(np.float32)

    first_times = next(iter(curves.values()))["times"]
    return translation_m, rotation_mat, first_times.astype(np.int64)


def load_pahoi_human_sequence(
    seq: PAHOISequence,
    num_betas: int,
) -> Dict[str, object]:
    # Keep `num_betas` in the public helper signature for compatibility with
    # existing callers. Source shape coefficients are intentionally not loaded:
    # all saved geometry is generated with the canonical zero-betas model.
    del num_betas
    with np.load(seq.concat_npz_path, allow_pickle=True) as concat_data:
        poses = np.asarray(concat_data["poses"], dtype=np.float32)
        trans = np.asarray(concat_data["trans"], dtype=np.float32)
        mocap_frame_rate = int(np.asarray(concat_data["mocap_frame_rate"]).item())
        if "surface_model_type" in concat_data:
            source_model_type = scalar_to_str(
                concat_data["surface_model_type"],
                default="smplx",
            )
        else:
            source_model_type = "smplx"

    if poses.ndim == 3:
        poses = poses[:, 0, :]
    poses = poses.reshape(poses.shape[0], -1)
    trans = trans.reshape(trans.shape[0], -1, 3)[:, 0, :]
    if poses.shape[1] < 66:
        raise ValueError(f"{seq.seq_name} pose dim {poses.shape[1]} is smaller than required 66")
    if normalize_body_model_type(source_model_type) != "smplx":
        raise ValueError(
            f"{seq.seq_name} declares {source_model_type!r}, but PAHOI requires SMPL-X poses"
        )

    return {
        "seq_name": seq.seq_name,
        "subject_id": seq.subject_id,
        "gender": CANONICAL_BODY_MODEL_GENDER,
        "mocap_frame_rate": mocap_frame_rate,
        "surface_model_type": "smplx",
        "poses": poses.astype(np.float32),
        "root_orient": poses[:, :3].astype(np.float32),
        "pose_body": poses[:, 3:66].astype(np.float32),
        "trans": trans.astype(np.float32),
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
            root_t = torch.tensor(root_orient[start:end], dtype=torch.float32, device=device)
            body_t = torch.tensor(pose_body[start:end], dtype=torch.float32, device=device)
            trans_t = torch.tensor(trans[start:end], dtype=torch.float32, device=device)
            betas_t = torch.zeros(
                end - start,
                int(bm.num_betas),
                dtype=torch.float32,
                device=device,
            )

            body_out = bm(
                root_orient=root_t,
                pose_body=body_t,
                trans=trans_t,
                betas=betas_t,
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

            del body_out, root_t, body_t, trans_t, betas_t, pose_aa, local_rot_mat
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return (
        torch.cat(position_chunks, dim=0),
        torch.cat(local_chunks, dim=0),
        torch.cat(global_chunks, dim=0),
    )


def load_bvh_hips_worldpos(seq: PAHOISequence, frame_count: int) -> np.ndarray:
    """Load PAHOI BVH/worldpos Hips positions in meters."""
    if seq.worldpos_npy_path is not None:
        worldpos = np.load(seq.worldpos_npy_path, allow_pickle=False)
        hips = np.asarray(worldpos[:frame_count, 0, :], dtype=np.float32)
    elif seq.worldpos_csv_path is not None:
        worldpos = np.genfromtxt(seq.worldpos_csv_path, delimiter=",", skip_header=1, dtype=np.float32)
        hips = np.asarray(worldpos[:frame_count, 1:4], dtype=np.float32) * 0.01
    else:
        raise FileNotFoundError(f"Missing PAHOI worldpos file for {seq.seq_name}")

    if hips.shape != (frame_count, 3):
        raise ValueError(f"{seq.seq_name} worldpos Hips shape {hips.shape}, expected {(frame_count, 3)}")
    return hips.astype(np.float32)


def _validate_raw_lengths(
    seq_name: str,
    human: Dict[str, object],
    obj_trans: np.ndarray,
    obj_rot: np.ndarray,
    metadata: Dict[str, object],
) -> int:
    human_T = int(human["poses"].shape[0])
    trans_T = int(human["trans"].shape[0])
    obj_T = int(obj_trans.shape[0])
    obj_rot_T = int(obj_rot.shape[0])
    expected_T = int(metadata["expected_frame_count"])
    lengths = {
        "poses": human_T,
        "trans": trans_T,
        "obj_trans": obj_T,
        "obj_rot": obj_rot_T,
        "expected": expected_T,
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{seq_name} frame count mismatch: {lengths}")
    return expected_T


def process_pahoi_sequence(
    seq: PAHOISequence,
    metadata: Dict[str, object],
    bm_loader: BodyModelLoader,
    object_mesh_root: str,
    output_dir: Path,
    device: torch.device,
    chunk_len: int,
    obj_points_count: int,
    fbx_rotation_order: str,
    max_frames: Optional[int] = None,
    floor_align: bool = True,
    skip_existing: bool = False,
    contact_max_hole_frames: int = 30,
    contact_min_duration_frames: int = 30,
    skip_object_displacement_without_hand_contact: bool = True,
    unheld_object_displacement_threshold_m: float = 0.10,
    skipped_sequences=None,
) -> Optional[Path]:
    out_path = output_dir / f"{seq.seq_name}.pt"
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

    human = load_pahoi_human_sequence(seq, num_betas=bm_loader.num_betas)
    obj_trans, obj_rot, fbx_key_times = parse_object_fbx_animation(
        seq.object_fbx_path,
        rotation_order=fbx_rotation_order,
    )
    raw_T = _validate_raw_lengths(seq.seq_name, human, obj_trans, obj_rot, metadata)

    if max_frames is not None:
        T = min(raw_T, int(max_frames))
    else:
        T = raw_T
    if T <= 0:
        raise ValueError(f"{seq.seq_name} has no frames after trimming")

    root_orient = human["root_orient"][:T].astype(np.float32)
    pose_body = human["pose_body"][:T].astype(np.float32)
    trans = human["trans"][:T].astype(np.float32).copy()
    obj_trans = obj_trans[:T].astype(np.float32).copy()
    obj_rot = obj_rot[:T].astype(np.float32)
    source_frame_indices = np.arange(T, dtype=np.int64)

    # The source family was validated while loading; the canonical loader can
    # only return the neutral SMPL-X instance.
    bm, body_model_type = bm_loader.resolve(human)
    position_global, rotation_local, rotation_global = compute_motion_features(
        bm=bm,
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans,
        device=device,
        chunk_len=max(1, int(chunk_len)),
    )

    obj_trans_bvh_world = obj_trans.copy()
    bvh_hips_worldpos = load_bvh_hips_worldpos(seq, T)
    smpl_root_world = position_global[:, 0, :].numpy().astype(np.float32)
    bvh_to_smpl_offset = (smpl_root_world - bvh_hips_worldpos).astype(np.float32)
    obj_trans = (obj_trans_bvh_world + bvh_to_smpl_offset).astype(np.float32)

    if floor_align:
        foot_floor_y = float(position_global[:, [7, 8, 10, 11], 1].min().item())
        position_global[:, :, 1] -= foot_floor_y
        trans[:, 1] -= foot_floor_y
        smpl_root_world[:, 1] -= foot_floor_y
        bvh_to_smpl_offset[:, 1] -= foot_floor_y
        obj_trans[:, 1] -= foot_floor_y

    object_label = str(metadata["object_label"])
    mesh_path = resolve_object_mesh_path(object_label, object_mesh_root)
    if mesh_path is None:
        raise FileNotFoundError(f"Missing PAHOI object mesh for '{object_label}' under {object_mesh_root}")
    mesh_name = Path(mesh_path).stem

    lfoot_contact, rfoot_contact = compute_foot_contact_labels(position_global.to(device))

    obj_rot_t = torch.tensor(obj_rot, dtype=torch.float32, device=device)
    obj_trans_t = torch.tensor(obj_trans, dtype=torch.float32, device=device)
    obj_scale_t = torch.ones(T, dtype=torch.float32, device=device)
    lhand_contact_raw, rhand_contact_raw, obj_contact_raw = compute_improved_contact_labels(
        obj_trans_t,
        obj_rot_t,
        obj_scale_t,
        position_global.to(device),
        object_mesh_root,
        object_label,
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
                dataset="PAHOI",
                seq_name=seq.seq_name,
                output_file=out_path.name,
                obj_name=object_label,
                obj_trans=obj_trans_t,
                lhand_contact_raw=lhand_contact_raw,
                rhand_contact_raw=rhand_contact_raw,
                obj_contact_raw=obj_contact_raw,
                lhand_contact=lhand_contact,
                rhand_contact=rhand_contact,
                obj_contact=obj_contact,
                unheld_motion_interval=unheld_motion_interval,
                extra={
                    "subject_id": seq.subject_id,
                    "action": str(metadata["action"]),
                },
            ))
        if out_path.exists():
            out_path.unlink()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return None

    canonical_points = load_object_canonical_points(
        object_label,
        obj_geo_root=object_mesh_root,
        sample_count=int(max(1, obj_points_count)),
        device="cpu",
    )
    if canonical_points is None:
        raise RuntimeError(f"Failed to load canonical PAHOI object points for '{object_label}'")

    output = {
        "seq_name": seq.seq_name,
        "dataset": "PAHOI",
        "subject_id": seq.subject_id,
        "surface_model_type": "smplx",
        **canonical_skeleton_metadata(body_model_type),
        "source_mocap_frame_rate": human["mocap_frame_rate"],
        "mocap_frame_rate": human["mocap_frame_rate"],
        "source_frame_indices": torch.from_numpy(source_frame_indices).long(),
        "expected_frame_count": int(metadata["expected_frame_count"]),
        "object_label": object_label,
        "object_weight": str(metadata["object_weight"]),
        "action": str(metadata["action"]),
        "pose": rotation_local.reshape(T, -1).float(),
        "position_global_full_gt_world": position_global.float(),
        "rotation_global": rotation_global.float(),
        "trans": torch.from_numpy(trans).float(),
        "human_imu_real": None,
        "obj_imu_real": None,
        "lfoot_contact": lfoot_contact.cpu().float(),
        "rfoot_contact": rfoot_contact.cpu().float(),
        "obj_name": object_label,
        "obj_mesh_name": mesh_name,
        "obj_mesh_path": os.path.relpath(mesh_path, Path.cwd()),
        "obj_points_canonical": canonical_points.half().cpu(),
        "has_object": True,
        "obj_scale": torch.ones(T, dtype=torch.float32),
        "obj_trans": obj_trans_t.detach().cpu().float(),
        "obj_trans_bvh_world": torch.from_numpy(obj_trans_bvh_world).float(),
        "bvh_hips_worldpos": torch.from_numpy(bvh_hips_worldpos).float(),
        "smpl_root_world": torch.from_numpy(smpl_root_world).float(),
        "bvh_to_smpl_offset": torch.from_numpy(bvh_to_smpl_offset).float(),
        "obj_rot": obj_rot_t.detach().cpu().float(),
        "lhand_contact": lhand_contact.cpu().bool(),
        "rhand_contact": rhand_contact.cpu().bool(),
        "obj_contact": obj_contact.cpu().bool(),
        "lhand_contact_raw": lhand_contact_raw.cpu().bool(),
        "rhand_contact_raw": rhand_contact_raw.cpu().bool(),
        "obj_contact_raw": obj_contact_raw.cpu().bool(),
        "source_files": [
            str(path)
            for path in (
                seq.concat_npz_path,
                seq.raw_npz_path,
                seq.object_fbx_path,
                seq.worldpos_npy_path,
                seq.worldpos_csv_path,
            )
            if path is not None
        ],
        "fbx_rotation_order": str(fbx_rotation_order),
        "fbx_key_times": torch.from_numpy(fbx_key_times[:T]).long(),
        "fbx_seconds": torch.from_numpy((fbx_key_times[:T].astype(np.float64) / FBX_TICKS_PER_SECOND).astype(np.float32)),
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
    parser = argparse.ArgumentParser(description="Preprocess PAHOI into flat IMUHOI-compatible pt files")
    parser.add_argument("--dataset_root", type=str, default="datasets/PAHOI/PA-HOI_Dataset")
    parser.add_argument("--object_mesh_root", type=str, default="datasets/PAHOI/Object_mesh")
    parser.add_argument("--output_dir", type=str, default="process/processed_split_data_PAHOI")
    parser.add_argument("--support_dir", type=str, default="datasets/smpl_models")
    parser.add_argument("--device", type=str, default='cuda', help="cpu/cuda override")
    parser.add_argument("--obj_points_count", type=int, default=256)
    parser.add_argument("--chunk_len", type=int, default=1024)
    parser.add_argument("--num_betas", type=int, default=16)
    parser.add_argument("--fbx_rotation_order", type=str, default="xyz")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject filter, e.g. 1 2")
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_floor_align", action="store_true")
    parser.add_argument(
        "--split_info",
        type=str,
        default=None,
        help=(
            "Existing split JSON (train_files/test_files) used to split the flat "
            "output into train/ and test/. PAHOI sequences are short, so they are "
            "processed and split directly instead of going through seg_sample.py."
        ),
    )
    parser.add_argument(
        "--split_output_dir",
        type=str,
        default=None,
        help="Root that receives train/ and test/ (default: --output_dir)",
    )
    parser.add_argument(
        "--allow_unassigned",
        action="store_true",
        help=(
            "Keep sequences the reference split does not label in the flat output "
            "dir and record them under unassigned_files instead of failing."
        ),
    )
    parser.add_argument(
        "--copy_after_split",
        action="store_true",
        help="Copy instead of move when applying --split_info",
    )
    parser.add_argument("--fail_fast", dest="fail_fast", action="store_true", default=True)
    parser.add_argument("--no_fail_fast", dest="fail_fast", action="store_false")
    add_contact_postprocess_arguments(parser)
    parser.set_defaults(skip_object_displacement_without_hand_contact=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = parse_sequence_metadata(args.dataset_root)
    sequences = collect_pahoi_sequences(args.dataset_root)
    if args.subjects:
        subjects = {str(subject).replace("s", "") for subject in args.subjects}
        sequences = [seq for seq in sequences if seq.subject_id in subjects]
    if args.max_sequences is not None:
        sequences = sequences[: int(args.max_sequences)]
    if not sequences:
        raise FileNotFoundError(f"No PAHOI sequences found under {args.dataset_root}")

    for seq in sequences:
        if seq.seq_name not in metadata:
            raise KeyError(f"No PAHOI frame metadata found for {seq.seq_name}")

    bm_loader = BodyModelLoader(args.support_dir, device, num_betas=args.num_betas)

    processed_count = 0
    skipped_sequences = []
    errors: List[Tuple[str, Exception]] = []
    for seq in tqdm(sequences, desc="Processing PAHOI sequences"):
        try:
            output_path = process_pahoi_sequence(
                seq=seq,
                metadata=metadata[seq.seq_name],
                bm_loader=bm_loader,
                object_mesh_root=args.object_mesh_root,
                output_dir=output_root,
                device=device,
                chunk_len=args.chunk_len,
                obj_points_count=args.obj_points_count,
                fbx_rotation_order=args.fbx_rotation_order,
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
        except Exception as exc:
            if args.fail_fast:
                raise
            errors.append((seq.seq_name, exc))
            print(f"Warning: skipped {seq.seq_name}: {exc}")

    manifest_path = write_contact_skip_manifest(
        str(output_root),
        dataset="PAHOI",
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

    print("PAHOI preprocessing finished")
    print(f"  processed: {processed_count} sequences -> {output_root}")
    print(f"  contact-skipped: {len(skipped_sequences)} sequences")
    print(f"  skip manifest: {manifest_path}")
    if errors:
        print(f"  skipped: {len(errors)} sequences")

    if args.split_info:
        split_root = Path(args.split_output_dir) if args.split_output_dir else output_root
        split_root.mkdir(parents=True, exist_ok=True)
        split_unified_by_json(
            str(output_root),
            args.split_info,
            str(split_root / "train"),
            str(split_root / "test"),
            copy_files=args.copy_after_split,
            allowed_missing_files={
                record["output_file"] for record in skipped_sequences
            },
            output_split_info_path=str(split_root / "split_info.json"),
            enforce_numeric_names=False,
            allow_unassigned=args.allow_unassigned,
        )
        if Path(manifest_path).resolve() != (split_root / "skipped_sequences.json").resolve():
            shutil.copy2(manifest_path, split_root / "skipped_sequences.json")
        print(f"  split layout: {split_root}/train, {split_root}/test")
    else:
        print("  output layout: flat top-level .pt files, ready for process/split_dataset.py")


if __name__ == "__main__":
    main()
