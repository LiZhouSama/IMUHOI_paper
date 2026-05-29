import argparse
import os
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import local2global_pose
import pytorch3d.transforms as transforms

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
    with np.load(seq.concat_npz_path, allow_pickle=True) as concat_data:
        poses = np.asarray(concat_data["poses"], dtype=np.float32)
        trans = np.asarray(concat_data["trans"], dtype=np.float32)
        gender = scalar_to_str(concat_data["gender"], default="neutral")
        mocap_frame_rate = int(np.asarray(concat_data["mocap_frame_rate"]).item())
        if "surface_model_type" in concat_data:
            surface_model_type = scalar_to_str(concat_data["surface_model_type"], default="smplx")
        else:
            surface_model_type = "smplx"
        concat_betas = np.asarray(concat_data["betas"], dtype=np.float32)

    if poses.ndim == 3:
        poses = poses[:, 0, :]
    poses = poses.reshape(poses.shape[0], -1)
    trans = trans.reshape(trans.shape[0], -1, 3)[:, 0, :]
    if poses.shape[1] < 66:
        raise ValueError(f"{seq.seq_name} pose dim {poses.shape[1]} is smaller than required 66")

    betas_source = concat_betas
    if seq.raw_npz_path is not None:
        with np.load(seq.raw_npz_path, allow_pickle=True) as raw_data:
            if "betas" in raw_data:
                betas_source = np.asarray(raw_data["betas"], dtype=np.float32)
    betas = pad_betas(betas_source, num_betas, poses.shape[0])

    return {
        "seq_name": seq.seq_name,
        "subject_id": seq.subject_id,
        "gender": gender,
        "mocap_frame_rate": mocap_frame_rate,
        "surface_model_type": surface_model_type,
        "poses": poses.astype(np.float32),
        "root_orient": poses[:, :3].astype(np.float32),
        "pose_body": poses[:, 3:66].astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": betas,
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
) -> Path:
    out_path = output_dir / f"{seq.seq_name}.pt"
    if skip_existing and out_path.exists():
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
    betas = human["betas"][:T].astype(np.float32)
    obj_trans = obj_trans[:T].astype(np.float32).copy()
    obj_rot = obj_rot[:T].astype(np.float32)
    source_frame_indices = np.arange(T, dtype=np.int64)

    bm = bm_loader.get(human["gender"])
    position_global, rotation_local, rotation_global = compute_motion_features(
        bm=bm,
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans,
        betas=betas,
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
    lhand_contact, rhand_contact, obj_contact = compute_improved_contact_labels(
        obj_trans_t,
        obj_rot_t,
        obj_scale_t,
        position_global.to(device),
        object_mesh_root,
        object_label,
        device,
        T,
    )

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
        "gender": human["gender"],
        "surface_model_type": human["surface_model_type"],
        "source_mocap_frame_rate": human["mocap_frame_rate"],
        "mocap_frame_rate": human["mocap_frame_rate"],
        "source_frame_indices": torch.from_numpy(source_frame_indices).long(),
        "expected_frame_count": int(metadata["expected_frame_count"]),
        "object_label": object_label,
        "object_weight": str(metadata["object_weight"]),
        "action": str(metadata["action"]),
        "rotation_local_full_gt_list": rotation_local.float(),
        "position_global_full_gt_world": position_global.float(),
        "rotation_global": rotation_global.float(),
        "trans": torch.from_numpy(trans).float(),
        "human_imu_real": None,
        "obj_imu_real": None,
        "lfoot_contact": lfoot_contact.cpu().float(),
        "rfoot_contact": rfoot_contact.cpu().float(),
        "betas": torch.from_numpy(betas).float(),
        "obj_name": object_label,
        "obj_mesh_name": mesh_name,
        "obj_mesh_path": os.path.relpath(mesh_path, Path.cwd()),
        "obj_points_canonical": canonical_points.half().cpu(),
        "obj_points_sample_count": int(canonical_points.shape[0]),
        "has_object": True,
        "obj_scale": torch.ones(T, dtype=torch.float32),
        "obj_trans": obj_trans_t.detach().cpu().float(),
        "obj_trans_bvh_world": torch.from_numpy(obj_trans_bvh_world).float(),
        "bvh_hips_worldpos": torch.from_numpy(bvh_hips_worldpos).float(),
        "smpl_root_world": torch.from_numpy(smpl_root_world).float(),
        "bvh_to_smpl_offset": torch.from_numpy(bvh_to_smpl_offset).float(),
        "obj_rot": obj_rot_t.detach().cpu().float(),
        "obj_com_pos": obj_trans_t.detach().cpu().float(),
        "lhand_contact": lhand_contact.cpu().bool(),
        "rhand_contact": rhand_contact.cpu().bool(),
        "obj_contact": obj_contact.cpu().bool(),
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
    }

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
    parser.add_argument("--fail_fast", dest="fail_fast", action="store_true", default=True)
    parser.add_argument("--no_fail_fast", dest="fail_fast", action="store_false")
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
    errors: List[Tuple[str, Exception]] = []
    for seq in tqdm(sequences, desc="Processing PAHOI sequences"):
        try:
            process_pahoi_sequence(
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
            )
            processed_count += 1
        except Exception as exc:
            if args.fail_fast:
                raise
            errors.append((seq.seq_name, exc))
            print(f"Warning: skipped {seq.seq_name}: {exc}")

    print("PAHOI preprocessing finished")
    print(f"  processed: {processed_count} sequences -> {output_root}")
    print("  output layout: flat top-level .pt files, ready for process/split_dataset.py")
    if errors:
        print(f"  skipped: {len(errors)} sequences")


if __name__ == "__main__":
    main()
