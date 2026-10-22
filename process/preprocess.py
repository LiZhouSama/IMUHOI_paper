import argparse
import os
import sys
import json
import shutil
import pickle
import re
import struct
import zlib
import numpy as np
import torch
import joblib
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel
import multiprocessing as mp
from functools import partial
import torch.nn.functional as F
import trimesh
import glob
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as SciPyRotation

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.human_pose import (
    CANONICAL_BODY_MODEL_GENDER,
    BodyModelRegistry,
    body_model_type_of,
    build_smpl24_joint_positions,
    canonical_skeleton_metadata,
    ensure_smpl24_local_rotations,
    infer_body_model_type_from_pose_width,
    local_rotations_to_global,
    normalize_body_model_type,
    select_hand_anchor_positions,
)
from configs import (
    FRAME_RATE,
    _REDUCED_INDICES,
    _SENSOR_POS_INDICES,
    _SENSOR_ROT_INDICES,
    _VEL_SELECTION_INDICES,
)
from dataset.dataset_IMUHOI import (
    _build_human_imu_root,
    _build_object_imu,
    _compute_joint_velocity_root,
    build_contact_endpoint_labels,
    compute_obj_direction_supervision,
)
from utils.utils import _central_diff
from utils.rotation_conversions import matrix_to_rotation_6d

contact_threh = 0.25
_CANONICAL_OBJ_POINTS_CACHE = {}
_MESH_ARRAY_CACHE = {}
_FBX_BINARY_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
_MAX_CONTACT_MESH_VERTICES = 4096
_MESH_EXTENSIONS = {".obj", ".ply", ".fbx"}
_INTERCAP_OBJECT_MESH_RE = re.compile(r"^(\d+)_second_obj\.(?:ply|obj)$", re.IGNORECASE)

# Keep the sequence-level label rule in preprocessing aligned with the
# geometry used by the RNN data loader.  The preprocessor still stores the
# raw labels so this heuristic remains reversible for later experiments.
_PALM_BONE_VARIATION_EPS = 1e-6
_OMOMO_CANONICAL_BODY_MODEL_TYPE = "smplx"
_BODY_MODEL_TYPE_FIELDS = (
    "surface_model_type",
    "body_model_type",
    "model_type",
    "model",
    "smpl_type",
)
_FULL_POSE_FIELDS = ("poses", "pose", "full_pose")


def _infer_body_model_type_without_subject_shape(seq_data, registry):
    """Infer only the model family; never consume source gender or betas."""
    for field in _BODY_MODEL_TYPE_FIELDS:
        value = seq_data.get(field)
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", "ignore")
        if isinstance(value, str):
            model_type = normalize_body_model_type(value)
            if model_type is not None:
                return model_type

    for field in _FULL_POSE_FIELDS:
        value = seq_data.get(field)
        if value is None:
            continue
        value = np.asarray(value)
        if value.ndim == 0:
            continue
        model_type = infer_body_model_type_from_pose_width(value.shape[-1])
        if model_type is not None:
            return model_type

    rest_offsets = seq_data.get("rest_offsets")
    if rest_offsets is not None:
        neutral_candidates = tuple(
            model_type
            for model_type in ("smplx", "smplh", "smpl")
            if CANONICAL_BODY_MODEL_GENDER
            in registry.available_genders(model_type)
        )
        if neutral_candidates:
            zero_betas = np.zeros(int(registry.num_betas), dtype=np.float32)
            # Subject-shaped offsets need not fall within the registry's strict
            # reconstruction tolerance when compared to a zero-shape skeleton.
            # Family identification only needs the nearest neutral template.
            model_type = registry.match_type_by_rest_offsets(
                rest_offsets,
                betas=zero_betas,
                gender=CANONICAL_BODY_MODEL_GENDER,
                candidates=neutral_candidates,
                tolerance_m=float("inf"),
            )
            if model_type is not None:
                return model_type

    return normalize_body_model_type(registry.default_type) or "smplh"


def _resolve_omomo_canonical_body_model(body_models, seq_data):
    """Resolve OMOMO's SMPL-X neutral model without gender fallback."""
    if not isinstance(body_models, BodyModelRegistry):
        raise TypeError(
            "OMOMO canonical preprocessing requires BodyModelRegistry so the "
            "neutral model path can be verified without gender fallback."
        )

    model_type = _infer_body_model_type_without_subject_shape(seq_data, body_models)
    if model_type != _OMOMO_CANONICAL_BODY_MODEL_TYPE:
        raise ValueError(
            "OMOMO must use the SMPL-X family, but shape-independent family "
            f"detection resolved {model_type!r}."
        )
    if CANONICAL_BODY_MODEL_GENDER not in body_models.available_genders(model_type):
        raise FileNotFoundError(
            f"OMOMO requires a neutral {model_type} model under "
            f"{body_models.support_dir}; gender fallback is disabled."
        )

    get_canonical = getattr(body_models, "get_canonical", None)
    if callable(get_canonical):
        body_model = get_canonical(model_type)
    else:
        body_model = body_models.get(model_type, CANONICAL_BODY_MODEL_GENDER)
    loaded_type = body_model_type_of(body_model)
    if loaded_type is not None and loaded_type != model_type:
        raise ValueError(
            f"Expected a {model_type} body model, got {loaded_type}."
        )
    return body_model, model_type


def _body_model_num_betas(body_model, default=16):
    value = getattr(body_model, "num_betas", default)
    if isinstance(value, torch.Tensor):
        value = value.item()
    value = int(value)
    if value <= 0:
        raise ValueError(f"BodyModel num_betas must be positive, got {value}")
    return value


def _forward_canonical_body_model(body_model, root_orient, pose_body, trans):
    """Run the neutral BodyModel with explicit, asserted zero betas."""
    batch_size = int(root_orient.shape[0])
    if int(pose_body.shape[0]) != batch_size or int(trans.shape[0]) != batch_size:
        raise ValueError("BodyModel inputs must have the same batch dimension")
    betas = torch.zeros(
        batch_size,
        _body_model_num_betas(body_model),
        device=root_orient.device,
        dtype=root_orient.dtype,
    )
    if torch.count_nonzero(betas).item() != 0:
        raise AssertionError("Canonical BodyModel betas must be exactly zero")
    return body_model(
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans,
        betas=betas,
    )


_PAHOI_OBJECT_ALIASES = {
    "cup noodles": "noodles_cup",
    "noodles cup": "noodles_cup",
    "dictionary": "oxford_dictionary",
    "oxford dictionary": "oxford_dictionary",
    "yoga ball": "yogaball",
    "yogaball": "yogaball",
    "balloon": "small_balloon",
    "big balloon": "big_balloon",
}


def precompute_sequence_features(
    position_global,
    rotation_global,
    trans,
    obj_trans=None,
    obj_rot=None,
    human_imu=None,
    obj_imu=None,
    lhand_contact=None,
    rhand_contact=None,
    obj_contact=None,
    has_object=True,
):
    """Build all deterministic full-sequence fields consumed by IMUHOI."""
    position_global = torch.as_tensor(position_global).float()
    sequence_length = int(position_global.shape[0])
    device = position_global.device
    dtype = position_global.dtype
    rotation_global = torch.as_tensor(
        rotation_global, device=device, dtype=dtype
    )

    if obj_trans is None:
        obj_trans = torch.zeros(sequence_length, 3, device=device, dtype=dtype)
    else:
        obj_trans = torch.as_tensor(obj_trans, device=device, dtype=dtype)
        if obj_trans.dim() == 3 and obj_trans.shape[-1] == 1:
            obj_trans = obj_trans.squeeze(-1)
    if obj_rot is None:
        obj_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(
            sequence_length, -1, -1
        )
    else:
        obj_rot = torch.as_tensor(obj_rot, device=device, dtype=dtype)
    obj_rot_6d = matrix_to_rotation_6d(obj_rot.reshape(-1, 3, 3)).reshape(
        sequence_length, 6
    )

    has_object = bool(has_object)
    if not has_object:
        obj_trans = torch.zeros_like(obj_trans)
        obj_rot_6d = matrix_to_rotation_6d(
            torch.eye(3, device=device, dtype=dtype).expand(sequence_length, -1, -1)
        )

    def _contacts(value):
        if value is None:
            return torch.zeros(sequence_length, device=device, dtype=torch.bool)
        return torch.as_tensor(value, device=device).reshape(-1)[:sequence_length].bool()

    lhand_contact = _contacts(lhand_contact)
    rhand_contact = _contacts(rhand_contact)
    obj_contact = _contacts(obj_contact)
    if not has_object:
        lhand_contact = torch.zeros_like(lhand_contact)
        rhand_contact = torch.zeros_like(rhand_contact)
        obj_contact = torch.zeros_like(obj_contact)

    if human_imu is None:
        human_imu = _build_human_imu_root(
            position_global,
            rotation_global,
            _SENSOR_POS_INDICES,
            _SENSOR_ROT_INDICES,
            FRAME_RATE,
        )
    else:
        human_imu = torch.as_tensor(human_imu, device=device, dtype=dtype)
        if human_imu.dim() == 3 and human_imu.shape[1] == 1:
            human_imu = human_imu.squeeze(1)

    if obj_imu is None and has_object:
        obj_imu = _build_object_imu(obj_rot_6d, obj_trans, FRAME_RATE)
    elif obj_imu is None:
        obj_imu = torch.zeros(sequence_length, 9, device=device, dtype=dtype)
    else:
        obj_imu = torch.as_tensor(obj_imu, device=device, dtype=dtype)
        if obj_imu.dim() == 3 and obj_imu.shape[1] == 1:
            obj_imu = obj_imu.squeeze(1)

    ori_root_reduced = rotation_global[:, :1].transpose(-1, -2).matmul(
        rotation_global[:, _REDUCED_INDICES]
    )
    joint_vel_root = _compute_joint_velocity_root(
        rotation_global, position_global, FRAME_RATE
    )
    root_vel = _central_diff(position_global[:, 0, :], 1.0 / FRAME_RATE)
    sensor_vel_root = joint_vel_root[:, _VEL_SELECTION_INDICES]
    sensor_vel_glb = _central_diff(
        position_global[:, _SENSOR_POS_INDICES, :], 1.0 / FRAME_RATE
    )
    obj_vel = (
        _central_diff(obj_trans, 1.0 / FRAME_RATE)
        if has_object
        else torch.zeros(sequence_length, 3, device=device, dtype=dtype)
    )

    lhand_obj_direction = torch.zeros(
        sequence_length, 3, device=device, dtype=dtype
    )
    rhand_obj_direction = torch.zeros_like(lhand_obj_direction)
    lhand_lb = torch.zeros(sequence_length, device=device, dtype=dtype)
    rhand_lb = torch.zeros_like(lhand_lb)
    if has_object:
        direction = compute_obj_direction_supervision(
            position_global,
            obj_trans,
            obj_rot_6d,
            lhand_contact,
            rhand_contact,
        )
        lhand_obj_direction = direction["lhand_obj_direction"]
        rhand_obj_direction = direction["rhand_obj_direction"]
        hand_positions = select_hand_anchor_positions(position_global)
        lhand_lb = torch.linalg.vector_norm(
            obj_trans - hand_positions[:, 0, :], dim=-1
        ) * lhand_contact.to(dtype)
        rhand_lb = torch.linalg.vector_norm(
            obj_trans - hand_positions[:, 1, :], dim=-1
        ) * rhand_contact.to(dtype)

    contact_endpoint, contact_endpoint_gauss = build_contact_endpoint_labels(
        lhand_contact,
        rhand_contact,
        obj_contact,
        device=device,
        dtype=dtype,
    )

    features = {
        "human_imu": human_imu.detach().cpu().float(),
        "obj_imu": obj_imu.detach().cpu().float(),
        "ori_root_reduced": ori_root_reduced.detach().cpu().float(),
        "root_vel": root_vel.detach().cpu().float(),
        "obj_vel": obj_vel.detach().cpu().float(),
        "sensor_vel_root": sensor_vel_root.detach().cpu().float(),
        "sensor_vel_glb": sensor_vel_glb.detach().cpu().float(),
        "obj_rot_6d": obj_rot_6d.detach().cpu().float(),
        "lhand_obj_direction": lhand_obj_direction.detach().cpu().float(),
        "rhand_obj_direction": rhand_obj_direction.detach().cpu().float(),
        "lhand_lb": lhand_lb.detach().cpu().float(),
        "rhand_lb": rhand_lb.detach().cpu().float(),
        "contact_endpoint": contact_endpoint.detach().cpu().float(),
        "contact_endpoint_gauss": contact_endpoint_gauss.detach().cpu().float(),
    }
    return features


def _normalize_object_name(name):
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _object_name_variants(obj_name):
    """Return likely mesh-name variants while preserving legacy exact names first."""
    if obj_name is None:
        return []

    raw = str(obj_name).strip()
    normalized = _normalize_object_name(raw)
    variants = []
    seen = set()

    def add(value):
        if value is None:
            return
        value = str(value).strip()
        if not value:
            return
        for candidate in (value, value.replace(" ", "_")):
            if candidate and candidate not in seen:
                seen.add(candidate)
                variants.append(candidate)

    add(raw)
    add(normalized)
    add(_PAHOI_OBJECT_ALIASES.get(normalized))

    if normalized.startswith("empty "):
        stripped = normalized[len("empty "):]
        add(stripped)
        add(_PAHOI_OBJECT_ALIASES.get(stripped))

    return variants


def _is_mesh_file(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in _MESH_EXTENSIONS


def _intercap_object_meshes(mesh_dir):
    if not os.path.isdir(mesh_dir):
        return []
    candidates = []
    for pattern in ("*_second_obj.ply", "*_second_obj.obj"):
        candidates.extend(sorted(glob.glob(os.path.join(mesh_dir, pattern))))
    return [path for path in candidates if _INTERCAP_OBJECT_MESH_RE.match(os.path.basename(path))]


def _resolve_direct_or_intercap_mesh_path(path_like):
    if not path_like:
        return None

    path_like = os.path.normpath(str(path_like))
    if _is_mesh_file(path_like):
        return path_like

    if os.path.isdir(path_like):
        intercap_candidates = _intercap_object_meshes(path_like)
        if intercap_candidates:
            return intercap_candidates[0]

    return None


def resolve_object_mesh_path(obj_name, obj_geo_root='./datasets/OMOMO/captured_objects'):
    """Resolve object mesh path across flat and per-object directory layouts."""
    if obj_name is None:
        return None

    obj_name = str(obj_name)
    obj_geo_root = str(obj_geo_root) if obj_geo_root else ""

    # Direct or relative mesh paths are used by datasets with per-sequence meshes,
    # such as InterCap.
    direct_path = _resolve_direct_or_intercap_mesh_path(obj_name)
    if direct_path is not None:
        return direct_path
    if obj_geo_root:
        root_direct = _resolve_direct_or_intercap_mesh_path(obj_geo_root)
        if root_direct is not None:
            return root_direct
        joined_direct = _resolve_direct_or_intercap_mesh_path(os.path.join(obj_geo_root, obj_name))
        if joined_direct is not None:
            return joined_direct

    if not obj_geo_root:
        return None

    obj_variants = _object_name_variants(obj_name)
    candidate_paths = []
    seen = set()

    def add(path):
        if not path:
            return
        norm_path = os.path.normpath(path)
        if norm_path not in seen:
            seen.add(norm_path)
            candidate_paths.append(norm_path)

    # Legacy flat layouts used by OMOMO / IMHD plus PAHOI's flat Object_mesh/*.fbx.
    for name_variant in obj_variants:
        for filename in (
            f"{name_variant}_cleaned_simplified.obj",
            f"{name_variant}_simplified_transformed.obj",
            f"{name_variant}.obj",
            f"{name_variant}.ply",
            f"{name_variant}.fbx",
        ):
            add(os.path.join(obj_geo_root, filename))

    # HODome and several other datasets store meshes under one directory per object.
    for name_variant in obj_variants:
        obj_dir = os.path.join(obj_geo_root, name_variant)
        if os.path.isdir(obj_dir):
            for filename in (
                f"{name_variant}_cleaned_simplified.obj",
                f"{name_variant}_simplified_transformed.obj",
                f"{name_variant}_face1000.obj",
                f"{name_variant}_face1000_wotex.obj",
                f"{name_variant}.obj",
                f"{name_variant}.ply",
                f"{name_variant}.fbx",
            ):
                add(os.path.join(obj_dir, filename))

            # Prefer light/simplified meshes for contact distance and canonical points.
            for pattern in (
                "*cleaned*simplified*.obj",
                "*simplified*transformed*.obj",
                f"{name_variant}*face1000*.obj",
                "*face1000*.obj",
                f"{name_variant}*.obj",
                "*.obj",
                f"{name_variant}*.ply",
                "*.ply",
                f"{name_variant}*.fbx",
                "*.fbx",
            ):
                for path in sorted(glob.glob(os.path.join(obj_dir, pattern))):
                    add(path)

    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def _quat_xyzw_to_matrix(quat_xyzw):
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _read_pahoi_xml_transform(mesh_path):
    xml_path = os.path.splitext(mesh_path)[0] + ".xml"
    if not os.path.exists(xml_path):
        return None

    root = ET.parse(xml_path).getroot()
    trans_node = root.find("trans")
    if trans_node is None:
        return None

    def _vec_from_node(node_name, default):
        node = trans_node.find(node_name)
        if node is None:
            return np.asarray(default, dtype=np.float32)
        return np.asarray(
            [
                float(node.attrib.get("x", default[0])),
                float(node.attrib.get("y", default[1])),
                float(node.attrib.get("z", default[2])),
            ],
            dtype=np.float32,
        )

    pos = _vec_from_node("pos", (0.0, 0.0, 0.0))
    scale = _vec_from_node("scale", (1.0, 1.0, 1.0))
    quat_node = trans_node.find("quat")
    if quat_node is None:
        rot = np.eye(3, dtype=np.float32)
    else:
        quat = np.asarray(
            [
                float(quat_node.attrib.get("x", 0.0)),
                float(quat_node.attrib.get("y", 0.0)),
                float(quat_node.attrib.get("z", 0.0)),
                float(quat_node.attrib.get("w", 1.0)),
            ],
            dtype=np.float32,
        )
        rot = _quat_xyzw_to_matrix(quat)
    return pos, rot, scale


class _FbxNode:
    def __init__(self, name, properties, children):
        self.name = name
        self.properties = properties
        self.children = children


def _fbx_node_header_size(version):
    return 25 if version >= 7500 else 13


def _is_null_fbx_record(data, offset, version):
    header_size = _fbx_node_header_size(version)
    if offset + header_size > len(data):
        return True
    return data[offset:offset + header_size] == b"\x00" * header_size


def _read_fbx_property(data, offset):
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


def _parse_fbx_node(data, offset, version):
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
    properties = []
    for _ in range(num_props):
        prop, offset = _read_fbx_property(data, offset)
        properties.append(prop)
    props_end = props_start + int(prop_list_len)
    if offset != props_end:
        offset = props_end

    children = []
    while offset < end_offset:
        if _is_null_fbx_record(data, offset, version):
            offset += header_size
            break
        child, offset = _parse_fbx_node(data, offset, version)
        if child is not None:
            children.append(child)

    return _FbxNode(name, properties, children), int(end_offset)


def _parse_binary_fbx_nodes(data):
    if not data.startswith(_FBX_BINARY_MAGIC):
        raise ValueError("Only binary FBX files are supported")
    version = struct.unpack_from("<I", data, 23)[0]
    offset = 27
    nodes = []
    while offset < len(data):
        if _is_null_fbx_record(data, offset, version):
            break
        node, offset = _parse_fbx_node(data, offset, version)
        if node is None:
            break
        nodes.append(node)
    return nodes


def _walk_fbx_nodes(nodes):
    for node in nodes:
        yield node
        yield from _walk_fbx_nodes(node.children)


def _find_fbx_child(node, name):
    for child in node.children:
        if child.name == name:
            return child
    return None


def _triangulate_fbx_polygon_indices(poly_indices):
    faces = []
    polygon = []
    for raw_idx in np.asarray(poly_indices, dtype=np.int64).reshape(-1):
        idx = int(raw_idx)
        is_last = idx < 0
        if is_last:
            idx = -idx - 1
        polygon.append(idx)
        if is_last:
            if len(polygon) >= 3:
                for i in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[i], polygon[i + 1]])
            polygon = []
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)


def _fbx_transform_from_model(model_node):
    translation = np.zeros(3, dtype=np.float32)
    rotation = np.zeros(3, dtype=np.float32)
    scaling = np.ones(3, dtype=np.float32)

    props_node = _find_fbx_child(model_node, "Properties70")
    if props_node is None:
        return translation, rotation, scaling

    for prop_node in props_node.children:
        if prop_node.name != "P" or len(prop_node.properties) < 7:
            continue
        prop_name = str(prop_node.properties[0])
        values = np.asarray(prop_node.properties[4:7], dtype=np.float32)
        if prop_name == "Lcl Translation":
            translation = values
        elif prop_name == "Lcl Rotation":
            rotation = values
        elif prop_name == "Lcl Scaling":
            scaling = values

    return translation, rotation, scaling


def _load_mesh_arrays_with_binary_fbx(obj_mesh_path):
    nodes = _parse_binary_fbx_nodes(open(obj_mesh_path, "rb").read())
    node_by_id = {}
    geometry_nodes = []
    connections = []

    for node in _walk_fbx_nodes(nodes):
        if node.properties and isinstance(node.properties[0], (int, np.integer)):
            node_by_id[int(node.properties[0])] = node
        if node.name == "Geometry" and len(node.properties) >= 3 and str(node.properties[2]) == "Mesh":
            geometry_nodes.append(node)
        elif node.name == "C" and len(node.properties) >= 3:
            relation = str(node.properties[0])
            if relation == "OO":
                connections.append((int(node.properties[1]), int(node.properties[2])))

    if not geometry_nodes:
        raise ValueError(f"No mesh Geometry nodes found in FBX: {obj_mesh_path}")

    geometry_to_model = {
        src: dst
        for src, dst in connections
        if src in node_by_id and node_by_id[src].name == "Geometry"
    }

    all_vertices = []
    all_faces = []
    vertex_offset = 0
    for geometry_node in geometry_nodes:
        vertices_node = _find_fbx_child(geometry_node, "Vertices")
        indices_node = _find_fbx_child(geometry_node, "PolygonVertexIndex")
        if vertices_node is None or indices_node is None:
            continue
        if not vertices_node.properties or not indices_node.properties:
            continue

        vertices = np.asarray(vertices_node.properties[0], dtype=np.float32).reshape(-1, 3)
        faces = _triangulate_fbx_polygon_indices(indices_node.properties[0])
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            continue

        geometry_id = int(geometry_node.properties[0])
        model_node = node_by_id.get(geometry_to_model.get(geometry_id))
        if model_node is not None and model_node.name == "Model":
            translation, rotation, scaling = _fbx_transform_from_model(model_node)
            rot_mat = SciPyRotation.from_euler("xyz", rotation, degrees=True).as_matrix().astype(np.float32)
            vertices = ((vertices * scaling[None, :]) @ rot_mat.T) + translation[None, :]

        all_vertices.append(vertices)
        all_faces.append(faces + vertex_offset)
        vertex_offset += vertices.shape[0]

    if not all_vertices:
        raise ValueError(f"No polygon mesh data found in FBX: {obj_mesh_path}")

    vertices = np.concatenate(all_vertices, axis=0).astype(np.float32) * 0.01
    faces = np.concatenate(all_faces, axis=0).astype(np.int64)
    return vertices, faces


def _load_mesh_arrays_with_trimesh(obj_mesh_path):
    mesh = trimesh.load_mesh(obj_mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [geom for geom in mesh.geometry.values() if hasattr(geom, "vertices")]
        if not geometries:
            raise ValueError(f"No mesh geometry in scene: {obj_mesh_path}")
        mesh = trimesh.util.concatenate(geometries)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3), dtype=np.int64)), dtype=np.int64)
    return vertices, faces


def _load_mesh_arrays_with_open3d(obj_mesh_path):
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError("Open3D is required to load FBX object meshes") from exc

    mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"Open3D loaded no vertices from {obj_mesh_path}")
    return vertices, faces


def _intercap_pose_from_mesh_path(obj_mesh_path):
    match = _INTERCAP_OBJECT_MESH_RE.match(os.path.basename(obj_mesh_path))
    if match is None:
        return None

    mesh_dir = os.path.dirname(obj_mesh_path)
    if os.path.basename(mesh_dir) != "Mesh":
        return None

    res2_path = os.path.join(os.path.dirname(mesh_dir), "res_2.pkl")
    if not os.path.exists(res2_path):
        return None

    frame_idx = int(match.group(1))
    try:
        with open(res2_path, "rb") as handle:
            data = pickle.load(handle)
        ob_pose = data.get("ob_pose")
        ob_trans = data.get("ob_trans")
        if ob_pose is None or ob_trans is None or frame_idx >= len(ob_pose) or frame_idx >= len(ob_trans):
            return None

        rot = SciPyRotation.from_rotvec(np.asarray(ob_pose[frame_idx], dtype=np.float32).reshape(3)).as_matrix()
        trans = np.asarray(ob_trans[frame_idx], dtype=np.float32).reshape(3)
        return rot.astype(np.float32), trans.astype(np.float32)
    except Exception:
        return None


def _canonicalize_intercap_object_mesh(obj_mesh_path, vertices):
    pose = _intercap_pose_from_mesh_path(obj_mesh_path)
    if pose is None:
        return vertices

    rot, trans = pose
    return ((vertices.astype(np.float32) - trans[None, :]) @ rot).astype(np.float32)


def _load_mesh_arrays(obj_mesh_path):
    """Load vertices/faces, applying PAHOI static XML transforms before centering."""
    key = os.path.abspath(obj_mesh_path)
    cached = _MESH_ARRAY_CACHE.get(key)
    if cached is not None:
        verts, faces = cached
        return verts.copy(), faces.copy()

    loaded_from_binary_fbx = False
    if os.path.splitext(obj_mesh_path)[1].lower() == ".fbx":
        try:
            vertices, faces = _load_mesh_arrays_with_binary_fbx(obj_mesh_path)
            loaded_from_binary_fbx = True
        except Exception:
            loaded_from_binary_fbx = False

    if not loaded_from_binary_fbx:
        try:
            vertices, faces = _load_mesh_arrays_with_trimesh(obj_mesh_path)
        except Exception:
            vertices, faces = _load_mesh_arrays_with_open3d(obj_mesh_path)

    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"Invalid object mesh vertices for {obj_mesh_path}")

    if os.path.splitext(obj_mesh_path)[1].lower() == ".fbx" and not loaded_from_binary_fbx:
        # PAHOI FBX meshes are authored in centimeters; body/object tracks are in meters.
        vertices = vertices * 0.01

    vertices = _canonicalize_intercap_object_mesh(obj_mesh_path, vertices)

    transform = _read_pahoi_xml_transform(obj_mesh_path)
    if transform is not None:
        pos, rot, scale = transform
        vertices = ((vertices * scale[None, :]) @ rot.T) + pos[None, :]

    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int64)
    _MESH_ARRAY_CACHE[key] = (vertices.copy(), faces.copy())
    return vertices, faces


def _boolean_runs(mask):
    """Return ``(value, start, end)`` runs for a one-dimensional boolean tensor."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError("contact labels must be a torch.Tensor")
    if mask.ndim != 1:
        raise ValueError("contact labels must have shape [T]")
    if mask.numel() == 0:
        return []

    values = mask.detach().bool().cpu().tolist()
    runs = []
    start = 0
    current = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value != current:
            runs.append((current, start, index - 1))
            start = index
            current = value
    runs.append((current, start, len(values) - 1))
    return runs


def postprocess_contact_labels(
    contact_labels,
    max_hole_frames=30,
    min_contact_frames=30,
):
    """Fill short internal holes, then remove short contact episodes.

    Hole filling is performed first so contact fragments separated only by a
    short threshold dropout are treated as one interaction.  Leading/trailing
    non-contact regions are never filled.  A contact segment of exactly
    ``min_contact_frames`` is retained.
    """
    max_hole_frames = int(max_hole_frames)
    min_contact_frames = int(min_contact_frames)
    if max_hole_frames < 0:
        raise ValueError("max_hole_frames must be non-negative")
    if min_contact_frames < 1:
        raise ValueError("min_contact_frames must be at least 1")

    filled = contact_labels.bool().clone()
    sequence_length = int(filled.numel())
    if sequence_length == 0:
        return filled

    for is_contact, start, end in _boolean_runs(filled):
        run_length = end - start + 1
        is_internal_hole = not is_contact and start > 0 and end < sequence_length - 1
        if is_internal_hole and run_length <= max_hole_frames:
            filled[start:end + 1] = True

    filtered = torch.zeros_like(filled)
    for is_contact, start, end in _boolean_runs(filled):
        if is_contact and end - start + 1 >= min_contact_frames:
            filtered[start:end + 1] = True
    return filtered


def remove_short_contact_runs(contact_labels, min_frames):
    """Drop True runs shorter than ``min_frames``; never fills gaps."""
    min_frames = int(min_frames)
    if min_frames < 1:
        raise ValueError("min_frames must be at least 1")
    cleaned = torch.zeros_like(contact_labels.bool())
    for is_contact, start, end in _boolean_runs(contact_labels.bool()):
        if is_contact and end - start + 1 >= min_frames:
            cleaned[start:end + 1] = True
    return cleaned


def find_unheld_object_displacement_interval(
    obj_trans,
    lhand_contact,
    rhand_contact,
    min_displacement_m=0.10,
):
    """Find a no-hand-contact interval whose object displacement is too large.

    Hand contact splits the sequence into independent intervals.  Within each
    interval where neither hand is in contact, the maximum pairwise Euclidean
    distance between object positions is measured.  This detects throws and
    motion caused by other body parts even if hand contact exists elsewhere in
    the same sequence.
    """
    if not isinstance(obj_trans, torch.Tensor) or obj_trans.ndim != 2 or obj_trans.shape[1] != 3:
        raise ValueError("obj_trans must be a torch.Tensor with shape [T, 3]")
    if not isinstance(lhand_contact, torch.Tensor) or not isinstance(rhand_contact, torch.Tensor):
        raise TypeError("hand contact labels must be torch.Tensor instances")
    if lhand_contact.ndim != 1 or rhand_contact.ndim != 1:
        raise ValueError("hand contact labels must have shape [T]")
    if obj_trans.shape[0] != lhand_contact.shape[0] or obj_trans.shape[0] != rhand_contact.shape[0]:
        raise ValueError("object translation and hand contact sequence lengths must match")

    min_displacement_m = float(min_displacement_m)
    if min_displacement_m <= 0:
        raise ValueError("min_displacement_m must be positive")
    if not bool(torch.isfinite(obj_trans).all()):
        raise ValueError("obj_trans contains non-finite values")

    no_hand_contact = ~(lhand_contact.bool() | rhand_contact.bool())
    for is_unheld, interval_start, interval_end in _boolean_runs(no_hand_contact):
        interval_frames = interval_end - interval_start + 1
        if not is_unheld or interval_frames < 2:
            continue

        positions = obj_trans[interval_start:interval_end + 1]
        max_distance = positions.new_tensor(-1.0)
        local_start = 0
        local_end = 0
        distance_block_frames = 1024
        for block_start in range(0, interval_frames, distance_block_frames):
            block_end = min(block_start + distance_block_frames, interval_frames)
            pairwise_block = torch.cdist(
                positions[block_start:block_end],
                positions,
            )
            block_max, block_flat_index = pairwise_block.reshape(-1).max(dim=0)
            if block_max > max_distance:
                max_distance = block_max
                flat_index = int(block_flat_index.item())
                local_start = block_start + flat_index // interval_frames
                local_end = flat_index % interval_frames
        max_distance_m = float(max_distance.item())
        if max_distance_m < min_displacement_m:
            continue

        motion_start = interval_start + min(local_start, local_end)
        motion_end = interval_start + max(local_start, local_end)
        return {
            "interval_start_frame": int(interval_start),
            "interval_end_frame": int(interval_end),
            "interval_frames": int(interval_frames),
            "motion_start_frame": int(motion_start),
            "motion_end_frame": int(motion_end),
            "max_displacement_m": max_distance_m,
            "displacement_threshold_m": min_displacement_m,
        }

    return None


# v2: object-movement (obj_move) detection switched from per-frame pose diffs to
# a jitter-robust criterion (zero-phase low-pass + windowed net displacement).
# v3: obj_move cleanup no longer fills holes; it only drops raw moving runs
# shorter than OBJ_MOVE_MIN_RUN_FRAMES.  Hole filling / minimum duration
# (contact_max_hole_frames / contact_min_duration_frames) apply to hands only.
CONTACT_POSTPROCESS_VERSION = 3

# Object-movement criterion.  Vision-fitted object poses (e.g. BEHAVE) jitter by
# ~0.2-1 mm / ~0.3 deg per frame and drift slowly at 2-4 Hz, which the old
# per-frame thresholds (0.8 mm, |dR|_F 0.005) flagged as motion on ~40% of
# static frames.  Fitting noise does not accumulate over a window while real
# motion does, so movement is detected from the net displacement across a short
# window of the low-passed trajectory.  Only the obj_move decision uses the
# filtered signal; saved GT poses are untouched.
OBJ_MOTION_LOWPASS_HZ = 6.0
OBJ_MOTION_LOWPASS_ORDER = 4
OBJ_MOTION_HALF_WINDOW_FRAMES = 5
OBJ_MOTION_TRANS_THRESHOLD_M = 0.01
OBJ_MOTION_ROT_THRESHOLD_DEG = 3.0
# The windowed criterion's "static" is strict (net < 1 cm and < 3 deg), so raw
# static frames are trusted as-is: even gaps of a few frames inside a carry are
# near-still pauses.  Filling them would let 1-2 frame rotation-jitter spikes act
# as bridges that merge long static stretches into motion.  Only short isolated
# moving runs (jitter spikes) are removed.
OBJ_MOVE_MIN_RUN_FRAMES = 15


def _zero_phase_lowpass(values, cutoff_hz, fps, order=OBJ_MOTION_LOWPASS_ORDER):
    """Zero-phase Butterworth low-pass along axis 0 (numpy, float64)."""
    from scipy.signal import butter, filtfilt

    values = np.asarray(values, dtype=np.float64)
    nyquist = 0.5 * float(fps)
    if cutoff_hz is None or cutoff_hz <= 0 or cutoff_hz >= nyquist or values.shape[0] < 2:
        return values.copy()
    b, a = butter(int(order), float(cutoff_hz) / nyquist)
    padlen = min(values.shape[0] - 1, 3 * max(len(a), len(b)))
    return filtfilt(b, a, values, axis=0, padlen=padlen)


def _lowpass_rotation_matrices(rot, cutoff_hz, fps):
    """Low-pass a rotation track [T, 3, 3] on sign-continuous quaternions."""
    quat = SciPyRotation.from_matrix(np.asarray(rot, dtype=np.float64)).as_quat()
    if quat.shape[0] > 1:
        step_sign = np.sign(np.sum(quat[1:] * quat[:-1], axis=-1))
        step_sign[step_sign == 0] = 1.0
        quat = quat * np.concatenate(([1.0], np.cumprod(step_sign)))[:, None]
    quat = _zero_phase_lowpass(quat, cutoff_hz, fps)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    return SciPyRotation.from_quat(quat).as_matrix()


def compute_object_motion_stats(
    obj_trans,
    obj_rot,
    fps=FRAME_RATE,
    lowpass_hz=OBJ_MOTION_LOWPASS_HZ,
    half_window_frames=OBJ_MOTION_HALF_WINDOW_FRAMES,
):
    """Windowed net object motion used by the obj_move criterion.

    The object pose is low-passed (zero-phase), then for every frame t the net
    translation (m) and net rotation angle (deg) between frames
    t - half_window_frames and t + half_window_frames (clamped) are measured.

    Args:
        obj_trans: [T, 3] object translation (torch or numpy).
        obj_rot: [T, 3, 3] object rotation matrices (torch or numpy).
    Returns:
        (net_trans_m, net_rot_deg): two float64 numpy arrays of shape [T].
    """
    trans = (
        obj_trans.detach().cpu().double().numpy()
        if isinstance(obj_trans, torch.Tensor)
        else np.asarray(obj_trans, dtype=np.float64)
    ).reshape(-1, 3)
    rot = (
        obj_rot.detach().cpu().double().numpy()
        if isinstance(obj_rot, torch.Tensor)
        else np.asarray(obj_rot, dtype=np.float64)
    ).reshape(-1, 3, 3)
    if trans.shape[0] != rot.shape[0]:
        raise ValueError("obj_trans and obj_rot must have the same length")
    num_frames = trans.shape[0]
    if num_frames <= 1:
        return np.zeros(num_frames), np.zeros(num_frames)

    trans_lp = _zero_phase_lowpass(trans, lowpass_hz, fps)
    rot_lp = _lowpass_rotation_matrices(rot, lowpass_hz, fps)

    half_window = max(1, int(half_window_frames))
    frame_index = np.arange(num_frames)
    start = np.clip(frame_index - half_window, 0, num_frames - 1)
    end = np.clip(frame_index + half_window, 0, num_frames - 1)
    net_trans = np.linalg.norm(trans_lp[end] - trans_lp[start], axis=-1)
    relative_rot = np.einsum("tji,tjk->tik", rot_lp[start], rot_lp[end])
    net_rot_deg = np.degrees(SciPyRotation.from_matrix(relative_rot).magnitude())
    return net_trans, net_rot_deg


def compute_object_movement_mask(
    obj_trans,
    obj_rot,
    fps=FRAME_RATE,
    lowpass_hz=OBJ_MOTION_LOWPASS_HZ,
    half_window_frames=OBJ_MOTION_HALF_WINDOW_FRAMES,
    trans_threshold_m=OBJ_MOTION_TRANS_THRESHOLD_M,
    rot_threshold_deg=OBJ_MOTION_ROT_THRESHOLD_DEG,
):
    """Per-frame object-movement mask robust to pose-fitting jitter.

    Frame t is moving when the windowed net motion from
    ``compute_object_motion_stats`` exceeds ``trans_threshold_m`` or
    ``rot_threshold_deg``.

    Returns:
        torch.bool tensor [T] on the device of ``obj_trans`` when it is a tensor.
    """
    device = obj_trans.device if isinstance(obj_trans, torch.Tensor) else torch.device("cpu")
    net_trans, net_rot_deg = compute_object_motion_stats(
        obj_trans, obj_rot, fps=fps, lowpass_hz=lowpass_hz,
        half_window_frames=half_window_frames,
    )
    moving = (net_trans > float(trans_threshold_m)) | (net_rot_deg > float(rot_threshold_deg))
    return torch.from_numpy(moving).to(device=device, dtype=torch.bool)


def postprocess_sequence_contact_labels(
    lhand_contact_raw,
    rhand_contact_raw,
    obj_contact_raw,
    obj_trans,
    max_hole_frames=30,
    min_contact_frames=30,
    skip_object_displacement_without_hand_contact=True,
    unheld_object_displacement_threshold_m=0.10,
):
    """Apply the canonical full-sequence contact cleanup and exclusion test."""
    labels = (lhand_contact_raw, rhand_contact_raw, obj_contact_raw)
    if any(not isinstance(label, torch.Tensor) or label.ndim != 1 for label in labels):
        raise ValueError("raw contact labels must be one-dimensional torch.Tensor values")
    if not (labels[0].shape == labels[1].shape == labels[2].shape):
        raise ValueError("left/right/object raw contact labels must have the same shape")
    if obj_trans.shape[0] != labels[0].shape[0]:
        raise ValueError("object translation and contact labels must have the same length")

    lhand_contact = postprocess_contact_labels(
        lhand_contact_raw,
        max_hole_frames=max_hole_frames,
        min_contact_frames=min_contact_frames,
    )
    rhand_contact = postprocess_contact_labels(
        rhand_contact_raw,
        max_hole_frames=max_hole_frames,
        min_contact_frames=min_contact_frames,
    )
    obj_contact = remove_short_contact_runs(obj_contact_raw, OBJ_MOVE_MIN_RUN_FRAMES)

    unheld_motion_interval = None
    if skip_object_displacement_without_hand_contact:
        unheld_motion_interval = find_unheld_object_displacement_interval(
            obj_trans,
            lhand_contact,
            rhand_contact,
            min_displacement_m=unheld_object_displacement_threshold_m,
        )
    return (
        lhand_contact,
        rhand_contact,
        obj_contact,
        unheld_motion_interval,
    )


def build_unheld_object_motion_skip_record(
    *,
    dataset,
    seq_name,
    output_file,
    obj_name,
    obj_trans,
    lhand_contact_raw,
    rhand_contact_raw,
    obj_contact_raw,
    lhand_contact,
    rhand_contact,
    obj_contact,
    unheld_motion_interval,
    extra=None,
):
    """Build a consistent manifest record for a sequence-level exclusion."""
    if unheld_motion_interval is None:
        raise ValueError("unheld_motion_interval is required for a skip record")
    record = {
        "dataset": str(dataset),
        "seq_name": str(seq_name),
        "output_file": str(output_file),
        "obj_name": str(obj_name),
        "reason": "object_displacement_without_hand_contact",
        "sequence_frames": int(obj_trans.shape[0]),
        "obj_motion_frames": int(obj_contact.sum().item()),
        "lhand_contact_frames": int(lhand_contact.sum().item()),
        "rhand_contact_frames": int(rhand_contact.sum().item()),
        "obj_motion_raw_frames": int(obj_contact_raw.sum().item()),
        "lhand_contact_raw_frames": int(lhand_contact_raw.sum().item()),
        "rhand_contact_raw_frames": int(rhand_contact_raw.sum().item()),
    }
    record.update(dict(unheld_motion_interval))
    if extra:
        record.update(dict(extra))
    return record


def contact_postprocess_metadata(
    *,
    max_hole_frames,
    min_contact_frames,
    skip_object_displacement_without_hand_contact,
    unheld_object_displacement_threshold_m,
):
    return {
        "contact_postprocess_version": int(CONTACT_POSTPROCESS_VERSION),
        "obj_motion_lowpass_hz": float(OBJ_MOTION_LOWPASS_HZ),
        "obj_motion_half_window_frames": int(OBJ_MOTION_HALF_WINDOW_FRAMES),
        "obj_motion_trans_threshold_m": float(OBJ_MOTION_TRANS_THRESHOLD_M),
        "obj_motion_rot_threshold_deg": float(OBJ_MOTION_ROT_THRESHOLD_DEG),
        "obj_move_min_run_frames": int(OBJ_MOVE_MIN_RUN_FRAMES),
        "contact_max_hole_frames": int(max_hole_frames),
        "contact_min_duration_frames": int(min_contact_frames),
        "skip_object_displacement_without_hand_contact": bool(
            skip_object_displacement_without_hand_contact
        ),
        "unheld_object_displacement_threshold_m": float(
            unheld_object_displacement_threshold_m
        ),
    }


def saved_contact_postprocess_matches(path, **expected_kwargs):
    """Return whether an existing pt file was produced with this configuration."""
    if not os.path.isfile(path):
        return False
    try:
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if not isinstance(saved, dict):
        return False
    expected = contact_postprocess_metadata(**expected_kwargs)
    return all(saved.get(key) == value for key, value in expected.items())


def add_contact_postprocess_arguments(parser):
    """Add the shared sequence-contact cleanup CLI arguments."""
    parser.add_argument(
        "--contact-max-hole-frames",
        type=int,
        default=30,
        help="Fill internal no-contact holes up to this many frames.",
    )
    parser.add_argument(
        "--contact-min-duration-frames",
        type=int,
        default=30,
        help="Remove contact segments shorter than this many frames.",
    )
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--skip-object-displacement-without-hand-contact",
        dest="skip_object_displacement_without_hand_contact",
        action="store_true",
    )
    skip_group.add_argument(
        "--keep-object-displacement-without-hand-contact",
        dest="skip_object_displacement_without_hand_contact",
        action="store_false",
    )
    parser.set_defaults(skip_object_displacement_without_hand_contact=True)
    parser.add_argument(
        "--unheld-object-displacement-threshold-m",
        type=float,
        default=0.10,
        help="Exclude a sequence when an unheld interval spans this displacement.",
    )
    return parser


def write_contact_skip_manifest(
    output_dir,
    *,
    dataset,
    source_sequences,
    saved_sequences,
    skipped_sequences,
    max_hole_frames,
    min_contact_frames,
    skip_object_displacement_without_hand_contact,
    unheld_object_displacement_threshold_m,
):
    """Write the contact-cleanup/exclusion summary used by all preprocessors."""
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "dataset": str(dataset),
        "source_sequences": int(source_sequences),
        "saved_sequences": int(saved_sequences),
        "skipped_sequences_count": int(len(skipped_sequences)),
        **contact_postprocess_metadata(
            max_hole_frames=max_hole_frames,
            min_contact_frames=min_contact_frames,
            skip_object_displacement_without_hand_contact=(
                skip_object_displacement_without_hand_contact
            ),
            unheld_object_displacement_threshold_m=(
                unheld_object_displacement_threshold_m
            ),
        ),
        "sequences": list(skipped_sequences),
    }
    manifest_path = os.path.join(output_dir, "skipped_sequences.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest_path


def compute_improved_contact_labels(obj_trans, obj_rot, obj_scale, position_global_full_gt_world,
                                  obj_mesh_dir, obj_name, device, T, obj_movement_mask=None):
    """
    改进的接触标签计算函数

    算法流程：
    1. 物体移动检测：对物体位姿做 6 Hz 零相位低通后，按 ±5 帧窗口净位移/净转角判断是否在动
    2. 距离筛选：使用宽松的距离阈值排除非手部接触
    3. 速度相似度：比较左右手和物体的速度相似度，判断具体是哪只手在交互

    参数:
        obj_trans: 物体位置 [T, 3]
        obj_rot: 物体旋转矩阵 [T, 3, 3]
        obj_scale: 物体缩放 [T]
        position_global_full_gt_world: 全局关节位置 [T, J, 3]
        obj_mesh_dir: 物体网格目录
        object_name: 物体名称
        device: 计算设备
        T: 时间步数
        obj_movement_mask: 可选 [T] bool，预先在完整序列上计算的物体运动掩码。
            分块处理的调用方应传入完整序列 compute_object_movement_mask 的切片，
            避免低通/窗口判据在块边界处失真；为 None 时在当前输入上计算。

    返回:
        lhand_contact, rhand_contact, obj_contact: 接触标签 [T]
    """
    # 初始化接触标签
    lhand_contact = torch.zeros(T, dtype=torch.bool, device=device)
    rhand_contact = torch.zeros(T, dtype=torch.bool, device=device)
    obj_contact = torch.zeros(T, dtype=torch.bool, device=device)

    if T <= 1 or not obj_mesh_dir:
        return lhand_contact, rhand_contact, obj_contact

    # 1. Object-movement detection (low-pass + windowed net displacement).
    # Kept outside the try block below so that a malformed mask fails loudly
    # instead of being swallowed into all-zero labels.
    if obj_movement_mask is None:
        obj_movement_mask = compute_object_movement_mask(obj_trans, obj_rot)
    obj_movement_mask = torch.as_tensor(obj_movement_mask).to(device=device, dtype=torch.bool).reshape(-1)
    if obj_movement_mask.shape[0] != T:
        raise ValueError(
            f"obj_movement_mask length {obj_movement_mask.shape[0]} does not match T={T}"
        )

    try:

        # 如果没有检测到物体移动，直接返回
        if not obj_movement_mask.any():
            return lhand_contact, rhand_contact, obj_contact

        # 2. 加载物体网格进行距离筛选

        transformed_verts, _ = load_object_geometry(
            obj_name,
            obj_rot,
            obj_trans,
            obj_scale,
            obj_mesh_dir,
            device,
            max_vertices=_MAX_CONTACT_MESH_VERTICES,
        )  # [T, N_v, 3]

        # 提取手部关节位置
        hand_joints = select_hand_anchor_positions(position_global_full_gt_world).to(device)
        lhand_jnt = hand_joints[:, 0, :]  # SMPL-24 left metacarpal
        rhand_jnt = hand_joints[:, 1, :]  # SMPL-24 right metacarpal

        # 3. 距离筛选 - 使用更宽松的阈值
        loose_distance_threshold = 0.25  # 15cm 宽松阈值，用于排除非手部接触

        # 计算手部到物体的最小距离
        lhand2obj_dist = torch.cdist(lhand_jnt.unsqueeze(1), transformed_verts).squeeze(1)
        rhand2obj_dist = torch.cdist(rhand_jnt.unsqueeze(1), transformed_verts).squeeze(1)
        lhand2obj_dist_min = torch.min(lhand2obj_dist, dim=1)[0]  # [T]
        rhand2obj_dist_min = torch.min(rhand2obj_dist, dim=1)[0]  # [T]

        # 宽松距离掩码
        lhand_close_mask = lhand2obj_dist_min < loose_distance_threshold
        rhand_close_mask = rhand2obj_dist_min < loose_distance_threshold

        # 4. 速度相似度分析
        # 计算物体速度
        obj_velocity = torch.zeros(T, 3, device=device)
        obj_velocity[1:] = obj_trans[1:] - obj_trans[:-1]  # [T, 3]

        # 计算手部速度
        lhand_velocity = torch.zeros(T, 3, device=device)
        rhand_velocity = torch.zeros(T, 3, device=device)
        lhand_velocity[1:] = lhand_jnt[1:] - lhand_jnt[:-1]
        rhand_velocity[1:] = rhand_jnt[1:] - rhand_jnt[:-1]

        # 计算速度相似度 (使用余弦相似度和速度大小差异)
        def compute_velocity_similarity(hand_vel, obj_vel, eps=1e-8):
            """计算手部和物体速度的相似度"""
            # 余弦相似度
            hand_norm = torch.norm(hand_vel, dim=1, keepdim=True) + eps
            obj_norm = torch.norm(obj_vel, dim=1, keepdim=True) + eps
            cosine_sim = torch.sum(hand_vel * obj_vel, dim=1) / (hand_norm.squeeze() * obj_norm.squeeze())

            # 速度大小差异 (归一化)
            speed_diff = torch.abs(hand_norm.squeeze() - obj_norm.squeeze()) / (hand_norm.squeeze() + obj_norm.squeeze() + eps)

            # 综合相似度：余弦相似度高且速度大小差异小
            similarity = cosine_sim * (1 - speed_diff)
            return similarity

        lhand_vel_similarity = compute_velocity_similarity(lhand_velocity, obj_velocity)
        rhand_vel_similarity = compute_velocity_similarity(rhand_velocity, obj_velocity)

        # 5. 综合判断接触
        velocity_similarity_threshold = 0.3  # 速度相似度阈值
        min_contact_confidence = 0.4  # 最小接触置信度

        for t in range(T):
            # 物体接触：只要物体在运动（IMU有明显变化），就认为被接触
            obj_contact[t] = obj_movement_mask[t]

            if not obj_movement_mask[t]:
                continue

            # 检查是否有手部在合理距离内
            lhand_in_range = lhand_close_mask[t]
            rhand_in_range = rhand_close_mask[t]

            if not (lhand_in_range or rhand_in_range):
                continue

            # 获取当前帧的速度相似度
            lhand_sim = lhand_vel_similarity[t] if lhand_in_range else -1
            rhand_sim = rhand_vel_similarity[t] if rhand_in_range else -1

            # 判断接触逻辑
            lhand_contact_confidence = 0
            rhand_contact_confidence = 0

            if lhand_in_range:
                # 左手接触置信度：距离因子 + 速度相似度因子
                distance_factor = max(0, 1 - lhand2obj_dist_min[t] / loose_distance_threshold)
                velocity_factor = max(0, lhand_sim) if lhand_sim > velocity_similarity_threshold else 0
                lhand_contact_confidence = 0.4 * distance_factor + 0.6 * velocity_factor

            if rhand_in_range:
                # 右手接触置信度：距离因子 + 速度相似度因子
                distance_factor = max(0, 1 - rhand2obj_dist_min[t] / loose_distance_threshold)
                velocity_factor = max(0, rhand_sim) if rhand_sim > velocity_similarity_threshold else 0
                rhand_contact_confidence = 0.4 * distance_factor + 0.6 * velocity_factor

            # 根据置信度判断接触
            lhand_contact[t] = lhand_contact_confidence > min_contact_confidence
            rhand_contact[t] = rhand_contact_confidence > min_contact_confidence

    except Exception as e:
        print(f"警告: 计算改进接触标签时出错: {e}")
        import traceback
        traceback.print_exc()

    return lhand_contact, rhand_contact, obj_contact


def resolve_bimanual_contact_labels_sequence(
    lhand_contact: torch.Tensor,
    rhand_contact: torch.Tensor,
    obj_trans: torch.Tensor,
    position_global_full_gt_world: torch.Tensor,
    eps: float = _PALM_BONE_VARIATION_EPS,
):
    """Resolve simultaneous hand labels once on a complete sequence.

    For every contiguous segment where both hands are labeled as contacting,
    retain only the hand whose object-to-virtual-palm bone length has the
    smaller range over that *full* segment.  Exact ties use the smaller mean
    length and finally the left hand as a deterministic tie-breaker.

    This is intentionally a preprocessing-time heuristic.  It must not be
    reapplied to random training windows, otherwise the same source frame can
    receive different labels depending on the sampled window.
    """
    if not isinstance(lhand_contact, torch.Tensor) or not isinstance(rhand_contact, torch.Tensor):
        raise TypeError("hand contact labels must be torch.Tensor values")
    if not isinstance(obj_trans, torch.Tensor) or not isinstance(position_global_full_gt_world, torch.Tensor):
        raise TypeError("object and joint positions must be torch.Tensor values")
    if lhand_contact.ndim != 1 or rhand_contact.ndim != 1:
        raise ValueError("hand contact labels must have shape [T]")
    if lhand_contact.shape != rhand_contact.shape:
        raise ValueError("left/right hand contact labels must have the same shape")
    if obj_trans.shape[0] != lhand_contact.shape[0] or position_global_full_gt_world.shape[0] != lhand_contact.shape[0]:
        raise ValueError("sequence lengths do not match for bimanual contact resolution")

    resolved_left = lhand_contact.bool().clone()
    resolved_right = rhand_contact.bool().clone()
    both = resolved_left & resolved_right
    if not bool(both.any()):
        return resolved_left, resolved_right

    hand_anchor = select_hand_anchor_positions(position_global_full_gt_world)
    obj_trans = obj_trans.to(device=hand_anchor.device, dtype=hand_anchor.dtype)
    left_length = torch.linalg.vector_norm(obj_trans - hand_anchor[:, 0, :], dim=-1)
    right_length = torch.linalg.vector_norm(obj_trans - hand_anchor[:, 1, :], dim=-1)

    contact_indices = torch.where(both)[0]
    seg_start = int(contact_indices[0].item())
    prev = seg_start
    segments = []
    for index in contact_indices[1:].tolist():
        index = int(index)
        if index != prev + 1:
            segments.append((seg_start, prev))
            seg_start = index
        prev = index
    segments.append((seg_start, prev))

    for start, end in segments:
        left_values = left_length[start:end + 1]
        right_values = right_length[start:end + 1]
        left_values = left_values[torch.isfinite(left_values)]
        right_values = right_values[torch.isfinite(right_values)]
        if left_values.numel() == 0 or right_values.numel() == 0:
            # Keep both labels if the geometry is unusable; do not invent a
            # hand assignment from missing measurements.
            continue

        left_range = float(left_values.max() - left_values.min())
        right_range = float(right_values.max() - right_values.min())
        left_mean = float(left_values.mean())
        right_mean = float(right_values.mean())
        keep_left = (
            left_range < right_range - eps
            or (abs(left_range - right_range) <= eps and left_mean <= right_mean)
        )
        if keep_left:
            resolved_right[start:end + 1] = False
        else:
            resolved_left[start:end + 1] = False

    return resolved_left, resolved_right

def load_object_geometry(
    obj_name,
    obj_rot,
    obj_trans,
    obj_scale=None,
    obj_geo_root='./datasets/OMOMO/captured_objects',
    device='cpu',
    max_vertices=None,
    max_faces=None,
):
    """加载物体几何体并应用变换，返回变换后的顶点与面片。"""
    if obj_name is None:
        print("Warning: Object name is None, cannot load geometry.")
        return None, None

    obj_mesh_path = resolve_object_mesh_path(obj_name, obj_geo_root)
    if obj_mesh_path is None:
        print(f"警告: 未找到物体网格文件: {obj_name} under {obj_geo_root}")
        T = obj_trans.shape[0] if obj_trans is not None else 1
        return (
            torch.zeros((T, 1, 3), device=device),
            np.zeros((0, 3), dtype=np.int64),
        )

    transformed_verts, obj_mesh_faces = apply_transformation_to_obj_geometry(
        obj_mesh_path,
        obj_rot,
        obj_trans,
        scale=obj_scale,
        device=device,
        max_vertices=max_vertices,
        max_faces=max_faces,
    )  # [T, N_v, 3]
    return transformed_verts, obj_mesh_faces


def load_canonical_points_from_mesh_path(obj_mesh_path, sample_count=256, device='cpu'):
    """Load canonical object points [N,3] centered at mesh centroid."""
    key = (os.path.abspath(obj_mesh_path), int(sample_count))
    cached = _CANONICAL_OBJ_POINTS_CACHE.get(key)
    if isinstance(cached, torch.Tensor):
        return cached.to(device=device)

    obj_mesh_verts_np, obj_mesh_faces = _load_mesh_arrays(obj_mesh_path)
    centroid = obj_mesh_verts_np.mean(axis=0)
    obj_mesh_verts_np = obj_mesh_verts_np - centroid

    if obj_mesh_faces.shape[0] > 0:
        mesh = trimesh.Trimesh(vertices=obj_mesh_verts_np, faces=obj_mesh_faces, process=False)
        sampled_points_np, _ = trimesh.sample.sample_surface(mesh, int(sample_count))
    else:
        sample_count = int(sample_count)
        if obj_mesh_verts_np.shape[0] >= sample_count:
            idx = np.linspace(0, obj_mesh_verts_np.shape[0] - 1, sample_count).astype(np.int64)
            sampled_points_np = obj_mesh_verts_np[idx]
        else:
            pad = np.repeat(obj_mesh_verts_np[-1:], sample_count - obj_mesh_verts_np.shape[0], axis=0)
            sampled_points_np = np.concatenate([obj_mesh_verts_np, pad], axis=0)

    points = torch.from_numpy(sampled_points_np.astype(np.float32))
    _CANONICAL_OBJ_POINTS_CACHE[key] = points
    return points.to(device=device)


def load_object_canonical_points(
    obj_name,
    obj_geo_root='./datasets/OMOMO/captured_objects',
    sample_count=256,
    device='cpu',
):
    """Resolve mesh path and return canonical object points [N,3]."""
    if obj_name is None:
        return None

    obj_mesh_path = resolve_object_mesh_path(obj_name, obj_geo_root)
    if obj_mesh_path is None:
        return None

    try:
        return load_canonical_points_from_mesh_path(obj_mesh_path, sample_count=sample_count, device=device)
    except Exception:
        return None


_DECIMATED_MESH_CACHE = {}


def decimate_mesh_arrays(vertices, faces, max_faces, cache_key=None):
    """Quadric-decimate a display mesh, keeping a watertight face list.

    Visualization does not need the captured resolution: a BEHAVE chair is
    132857 vertices, and one transformed sequence of it is gigabytes.  Results
    are cached per (mesh, target) because a viewer reloads the same object for
    every sequence it shows.
    """
    if not max_faces or faces is None or len(faces) == 0 or len(faces) <= int(max_faces):
        return vertices, faces
    key = (cache_key, int(max_faces))
    if cache_key is not None and key in _DECIMATED_MESH_CACHE:
        return _DECIMATED_MESH_CACHE[key]
    try:
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
        )
        simplified = mesh.simplify_quadric_decimation(int(max_faces))
        out_vertices = np.asarray(simplified.vertices, dtype=np.float32)
        out_faces = np.asarray(simplified.triangles, dtype=np.int64)
    except Exception as exc:
        print(f"Mesh decimation failed, keeping the original resolution: {exc}")
        return vertices, faces
    if out_vertices.shape[0] == 0 or out_faces.shape[0] == 0:
        return vertices, faces
    if cache_key is not None:
        _DECIMATED_MESH_CACHE[key] = (out_vertices, out_faces)
    return out_vertices, out_faces


def apply_transformation_to_obj_geometry(obj_mesh_path, obj_rot, obj_trans, scale=None, device='cpu', max_vertices=None, max_faces=None):
    """
    应用变换到物体顶点 (遵循 hand_foot_dataset.py 的逻辑: Rotate -> Scale -> Translate)

    参数:
        obj_mesh_path: 物体网格路径
        obj_rot: 旋转矩阵 [T, 3, 3] (torch tensor on device)
        obj_trans: 平移向量 [T, 3] (torch tensor on device)
        scale: 缩放因子 [T] (torch tensor on device)
        device: 计算设备

    返回:
        transformed_obj_verts: 变换后的顶点 [T, Nv, 3] (torch tensor on device)
        obj_mesh_faces: 网格面片 [Nf, 3] (numpy array)
    """
    try:
        obj_mesh_verts_np, obj_mesh_faces = _load_mesh_arrays(obj_mesh_path) # Nv X 3
        obj_mesh_verts_np, obj_mesh_faces = decimate_mesh_arrays(
            obj_mesh_verts_np, obj_mesh_faces, max_faces,
            cache_key=os.path.abspath(obj_mesh_path),
        )
        centroid = obj_mesh_verts_np.mean(axis=0)
        obj_mesh_verts_np = obj_mesh_verts_np - centroid  # 适配BEHAVE TODO:不确定对其他数据集是否有效
        if max_vertices is not None and obj_mesh_verts_np.shape[0] > int(max_vertices):
            sample_idx = np.linspace(
                0,
                obj_mesh_verts_np.shape[0] - 1,
                int(max_vertices),
            ).astype(np.int64)
            obj_mesh_verts_np = obj_mesh_verts_np[sample_idx]
            obj_mesh_faces = np.zeros((0, 3), dtype=np.int64)

        obj_mesh_verts = torch.from_numpy(obj_mesh_verts_np).float().to(device) # Nv X 3
        seq_rot_mat = obj_rot.float().to(device) # T X 3 X 3
        seq_trans = obj_trans.float().to(device) # T X 3
        if scale is not None:
            seq_scale = scale.float().to(device) # T
        else:
            seq_scale = None

        T = seq_trans.shape[0]
        def _apply_seq_transform(points):
            points_seq = points[None].repeat(T, 1, 1) # T X N X 3

            # --- 遵循参考代码的顺序：Rotate -> Scale -> Translate ---

            # 1. 旋转 (Rotate): T X 3 X N
            points_rotated = torch.bmm(seq_rot_mat, points_seq.transpose(1, 2))

            # 2. 缩放 (Scale)
            if seq_scale is not None:
                scale_factor = seq_scale.unsqueeze(-1).unsqueeze(-1)
                points_scaled = scale_factor * points_rotated
            else:
                points_scaled = points_rotated

            # 3. 平移 (Translate)
            trans_vector = seq_trans.unsqueeze(-1)
            points_translated = points_scaled + trans_vector

            # 4. 转回 T X N X 3
            return points_translated.transpose(1, 2)

        transformed_obj_verts = _apply_seq_transform(obj_mesh_verts) # T X Nv X 3

    except Exception as e:
        print(f"应用变换到物体几何体失败: {e}")
        import traceback
        traceback.print_exc()
        # 返回设备上的虚拟数据
        T = obj_trans.shape[0] if obj_trans is not None else 1
        transformed_obj_verts = torch.zeros((T, 1, 3), device=device)
        obj_mesh_faces = np.zeros((0, 3), dtype=np.int64)

    return transformed_obj_verts, obj_mesh_faces

def compute_foot_contact_labels(position_global_full_gt_world, foot_velocity_threshold=0.008):
    """
    计算左脚和右脚的地面接触标签

    参数:
        position_global_full_gt_world: 全局关节位置 [T, 22, 3]
        foot_velocity_threshold: 脚部速度阈值，低于此值认为与地面接触

    返回:
        lfoot_contact: 左脚接触标签 [T]
        rfoot_contact: 右脚接触标签 [T]
    """
    T = position_global_full_gt_world.shape[0]
    device = position_global_full_gt_world.device

    # 提取左脚和右脚的全局位置（关节索引7和8）
    lfoot_pos = position_global_full_gt_world[:, 7, :]  # [T, 3] 左脚踝
    rfoot_pos = position_global_full_gt_world[:, 8, :]  # [T, 3] 右脚踝

    # 计算脚部速度（相邻帧差分）
    if T > 1:
        lfoot_vel = torch.zeros_like(lfoot_pos)
        rfoot_vel = torch.zeros_like(rfoot_pos)

        # 计算速度：当前帧 - 前一帧
        lfoot_vel[1:] = lfoot_pos[1:] - lfoot_pos[:-1]
        rfoot_vel[1:] = rfoot_pos[1:] - rfoot_pos[:-1]

        # 计算速度的L2范数
        lfoot_speed = torch.norm(lfoot_vel, dim=1)  # [T]
        rfoot_speed = torch.norm(rfoot_vel, dim=1)  # [T]

        # 根据速度阈值判断接触（速度小于阈值则接触）
        lfoot_contact = (lfoot_speed < foot_velocity_threshold).float()
        rfoot_contact = (rfoot_speed < foot_velocity_threshold).float()

        # 第一帧默认接触地面
        lfoot_contact[0] = 1.0
        rfoot_contact[0] = 1.0
    else:
        # 只有一帧时，默认接触地面
        lfoot_contact = torch.ones(T, device=device)
        rfoot_contact = torch.ones(T, device=device)

    return lfoot_contact, rfoot_contact

def process_sequence(
    seq_data,
    seq_key,
    save_dir,
    bm,
    device='cuda',
    bps_dir=None,
    bps_points=None,
    obj_mesh_dir=None,
    obj_points_count: int = 256,
    resolve_bimanual_contact_conflicts: bool = False,
    contact_max_hole_frames: int = 30,
    contact_min_duration_frames: int = 30,
    skip_object_displacement_without_hand_contact: bool = False,
    unheld_object_displacement_threshold_m: float = 0.10,
    skipped_sequences=None,
    skip_record_context=None,
    output_name=None,
):
    import pytorch3d.transforms as transforms
    """处理单个序列并保存为pt文件"""

    seq_name = str(seq_data['seq_name'])
    # OMOMO is represented in one canonical space: SMPL-X neutral with zero
    # betas. Source gender/betas are deliberately neither read nor propagated.
    bm, body_model_type = _resolve_omomo_canonical_body_model(bm, seq_data)
    canonical_num_betas = _body_model_num_betas(bm)

    root_orient_np = np.asarray(seq_data['root_orient'], dtype=np.float32).reshape(-1, 3)
    pose_body_np = np.asarray(seq_data['pose_body'], dtype=np.float32).reshape(-1, 63)
    trans_np = np.asarray(seq_data['trans'], dtype=np.float32).reshape(-1, 3)
    sequence_length = root_orient_np.shape[0]

    zero_root_orient = torch.zeros(sequence_length, 3, device=device)
    zero_pose_body = torch.zeros(sequence_length, 63, device=device)
    zero_trans = torch.zeros(sequence_length, 3, device=device)
    canonical_rest_pose = _forward_canonical_body_model(
        bm,
        zero_root_orient,
        zero_pose_body,
        zero_trans,
    )
    Jtr_0 = np.asarray(
        canonical_rest_pose.Jtr[:, 0, :].cpu(), dtype=np.float32
    )

    if output_name is None:
        output_name = f"{seq_key}.pt"
    output_name = str(output_name)
    if not output_name.endswith(".pt"):
        output_name += ".pt"

    seq_key_str = str(seq_key)
    dataset_name_str = str(seq_data.get('dataset', ''))

    names_to_check = [seq_name.lower(), seq_key_str.lower(), dataset_name_str.lower()]

    rot_x90_np = None
    rot_x90_torch = None
    rot_x90_np = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, -1.0, 0.0]], dtype=np.float32)
    # rot_x90_np = np.array([[1.0, 0.0, 0.0],
    #                         [0.0, 1.0, 0.0],
    #                         [0.0, 0.0, 1.0]], dtype=np.float32)
    rot_x90_torch = torch.from_numpy(rot_x90_np).float()

    root_rot_mat = transforms.axis_angle_to_matrix(torch.from_numpy(root_orient_np).float())
    rot_x90_repeat = rot_x90_torch.unsqueeze(0).repeat(root_rot_mat.shape[0], 1, 1)
    root_rot_mat = torch.matmul(rot_x90_repeat, root_rot_mat)
    root_orient_np = transforms.matrix_to_axis_angle(root_rot_mat).detach().cpu().numpy().astype(np.float32)

    # trans_np = trans_np @ rot_x90_np.T
    trans_np = (trans_np + Jtr_0) @ rot_x90_np.T - Jtr_0

    bdata_poses = np.concatenate([root_orient_np, pose_body_np], axis=1)

    # 构建body参数字典
    root_orient_torch = torch.from_numpy(root_orient_np).to(device).float()
    pose_body_torch = torch.from_numpy(pose_body_np).to(device).float()
    trans_torch = torch.from_numpy(trans_np).to(device).float()

    # The same neutral zero-beta model used for Jtr_0 produces every joint.
    body_pose_world = _forward_canonical_body_model(
        bm,
        root_orient_torch,
        pose_body_torch,
        trans_torch,
    )

    pose_aa = torch.tensor(bdata_poses, device=device, dtype=torch.float32).reshape(
        bdata_poses.shape[0], -1, 3
    )
    local_rot_mat = ensure_smpl24_local_rotations(transforms.axis_angle_to_matrix(pose_aa))
    pose_aa = torch.cat(
        (pose_aa, torch.zeros(pose_aa.shape[0], 2, 3, device=device)),
        dim=1,
    )
    rotation_global_matrot_reshaped = local_rotations_to_global(local_rot_mat)

    # 提取头部全局旋转
    # head_rotation_global_matrot = rotation_global_matrot[:, [15], :, :]

    # Body-22 plus the two palm anchors, resolved from the native joint layout.
    # OMOMO's own `rest_offsets[22:24]` used to be applied here as a special
    # case; with the correct (SMPL-X) body model those offsets are reproduced by
    # the native middle1 joints to 0.09 mm, so the special case is redundant.
    position_global_full_gt_world = build_smpl24_joint_positions(
        body_pose_world.Jtr
    ).cpu()

    # 计算足部接触标签
    lfoot_contact, rfoot_contact = compute_foot_contact_labels(
        position_global_full_gt_world.to(device)
    )

    # 计算头部全局变换矩阵
    # position_head_world = position_global_full_gt_world[:, 15, :].to(device)
    # head_global_trans = torch.eye(4, device=device).repeat(position_head_world.shape[0], 1, 1)
    # head_global_trans[:, :3, :3] = head_rotation_global_matrot.squeeze()
    # head_global_trans[:, :3, 3] = position_head_world



    # -------------------------------------------------

    # 提取物体相关信息(如果存在)
    obj_data = {}
    transformed_verts = None # 初始化以备后用
    if 'obj_scale' in seq_data:
        # 将物体数据转移到设备
        obj_scale = torch.tensor(seq_data['obj_scale'], device=device).float()

        # --- 更健壮地加载和处理 obj_trans ---
        obj_rot = torch.tensor(seq_data['obj_rot'], device=device).float()
        # OMOMO's obj_com_pos is the mesh-centered trajectory used for contact
        # and evaluation. Raw obj_trans is the object frame origin and can be
        # tens of centimeters away; the processed output still uses obj_trans
        # as the canonical field name, without duplicating obj_com_pos.
        obj_trans_source = seq_data.get("obj_com_pos")
        if obj_trans_source is None:
            obj_trans_source = seq_data.get("obj_trans")
        if obj_trans_source is None:
            raise KeyError("OMOMO sequence is missing obj_trans")
        obj_trans = torch.as_tensor(obj_trans_source, device=device).float().squeeze(-1)
        T = obj_trans.shape[0] # 获取时间步长

        if rot_x90_torch is not None:
            rot_x90_device = rot_x90_torch.to(device)
            rot_repeat = rot_x90_device.unsqueeze(0).repeat(T, 1, 1)
            obj_rot = torch.matmul(rot_repeat, obj_rot)
            obj_trans = torch.matmul(obj_trans, rot_x90_device.t())

        # 提取物体名称
        object_name = seq_name.split("_")[1] if "_" in seq_name else "unknown"
        has_object = bool(seq_data.get("has_object", True))

        # --- 改进的接触检测算法 ---
        lhand_contact_raw, rhand_contact_raw, obj_contact_raw = compute_improved_contact_labels(
            obj_trans, obj_rot, obj_scale, position_global_full_gt_world,
            obj_mesh_dir, object_name, device, T
        )
        lhand_contact = postprocess_contact_labels(
            lhand_contact_raw,
            max_hole_frames=contact_max_hole_frames,
            min_contact_frames=contact_min_duration_frames,
        )
        rhand_contact = postprocess_contact_labels(
            rhand_contact_raw,
            max_hole_frames=contact_max_hole_frames,
            min_contact_frames=contact_min_duration_frames,
        )
        obj_contact = remove_short_contact_runs(obj_contact_raw, OBJ_MOVE_MIN_RUN_FRAMES)
        if resolve_bimanual_contact_conflicts:
            lhand_contact, rhand_contact = resolve_bimanual_contact_labels_sequence(
                lhand_contact,
                rhand_contact,
                obj_trans,
                position_global_full_gt_world,
            )
        # --- 接触检测结束 ---

        unheld_motion_interval = None
        if has_object and skip_object_displacement_without_hand_contact:
            unheld_motion_interval = find_unheld_object_displacement_interval(
                obj_trans,
                lhand_contact,
                rhand_contact,
                min_displacement_m=unheld_object_displacement_threshold_m,
            )
        if unheld_motion_interval is not None:
            skip_record = {
                "output_file": output_name,
                "seq_key": str(seq_key),
                "seq_name": seq_name,
                "obj_name": object_name,
                "reason": "object_displacement_without_hand_contact",
                "sequence_frames": int(T),
                "obj_motion_frames": int(obj_contact.sum().item()),
                "lhand_contact_frames": int(lhand_contact.sum().item()),
                "rhand_contact_frames": int(rhand_contact.sum().item()),
                "lhand_contact_raw_frames": int(lhand_contact_raw.sum().item()),
                "rhand_contact_raw_frames": int(rhand_contact_raw.sum().item()),
            }
            skip_record.update(unheld_motion_interval)
            if skip_record_context:
                skip_record.update(dict(skip_record_context))
            if skipped_sequences is not None:
                skipped_sequences.append(skip_record)
            return 0

        # 预存canonical点云（可选）
        canonical_points = None
        if has_object and obj_mesh_dir:
            canonical_points = load_object_canonical_points(
                object_name,
                obj_geo_root=obj_mesh_dir,
                sample_count=int(max(1, obj_points_count)),
                device="cpu",
            )

        # 构建物体数据字典
        obj_data = {
            "obj_name": object_name,
            "has_object": has_object,
            "obj_scale": obj_scale.cpu(),
            "obj_trans": obj_trans.cpu(),
            "obj_rot": obj_rot.cpu(),
            "lhand_contact": lhand_contact.cpu(), # 基于距离
            "rhand_contact": rhand_contact.cpu(), # 基于距离
            "lhand_contact_raw": lhand_contact_raw.cpu(),
            "rhand_contact_raw": rhand_contact_raw.cpu(),
            "obj_contact": obj_contact.cpu(),      # 基于运动
            "obj_contact_raw": obj_contact_raw.cpu(),
            "contact_max_hole_frames": int(contact_max_hole_frames),
            "contact_min_duration_frames": int(contact_min_duration_frames),
            "unheld_object_displacement_threshold_m": float(
                unheld_object_displacement_threshold_m
            ),
        }
        if isinstance(canonical_points, torch.Tensor) and canonical_points.dim() == 2 and canonical_points.shape[-1] == 3:
            obj_data["obj_points_canonical"] = canonical_points.half().cpu()

    # 组装输出数据
    canonical_metadata = canonical_skeleton_metadata(body_model_type)
    data = {
        "seq_name": seq_name,
        **canonical_metadata,
        "canonical_body_model_type": body_model_type,
        "canonical_num_betas": canonical_num_betas,
        "canonical_joint_space": "neutral_zero_betas",
        # "head_global_trans": head_global_trans.cpu(),
        "pose": pose_aa.reshape(pose_aa.shape[0], -1).cpu(),   # [T, 72]
        "position_global_full_gt_world": position_global_full_gt_world.float(), # [T, 24, 3]
        "trans": trans_torch.cpu(),
        "rotation_global": rotation_global_matrot_reshaped.cpu(), # [T, 24, 3, 3]
        "lfoot_contact": lfoot_contact.cpu(),  # 左脚接触标签
        "rfoot_contact": rfoot_contact.cpu(),   # 右脚接触标签
    }

    data.update(
        precompute_sequence_features(
            position_global_full_gt_world,
            rotation_global_matrot_reshaped,
            trans_torch,
            obj_trans=obj_data.get("obj_trans") if obj_data else None,
            obj_rot=obj_data.get("obj_rot") if obj_data else None,
            # Always synthesize human IMU from the canonical joints above;
            # source IMU may encode subject-specific shape.
            human_imu=None,
            obj_imu=seq_data.get("obj_imu_real"),
            lhand_contact=obj_data.get("lhand_contact") if obj_data else None,
            rhand_contact=obj_data.get("rhand_contact") if obj_data else None,
            obj_contact=obj_data.get("obj_contact") if obj_data else None,
            has_object=bool(obj_data.get("has_object", False)) if obj_data else False,
        )
    )

    # 添加物体数据(如果存在)
    if obj_data:
        data.update(obj_data)

    # 保存处理后的数据
    torch.save(data, os.path.join(save_dir, output_name))
    return 1

def _process_unified_omomo(
    data_dict_train,
    data_dict_test,
    save_dir,
    bm,
    device,
    obj_mesh_dir,
    obj_points_count,
    resolve_bimanual_contact_conflicts,
    contact_max_hole_frames,
    contact_min_duration_frames,
    skip_object_displacement_without_hand_contact,
    unheld_object_displacement_threshold_m,
):
    """Process train then test into one numerically ordered directory.

    The source pickles both use keys ``0..N-1``.  Keeping those keys would
    overwrite the test samples, so the test offset is explicitly added after
    all training samples.  This produces the numbering expected by the
    existing OMOMO ``split_info.json`` (0..5881 for train followed by test).
    """
    os.makedirs(save_dir, exist_ok=True)
    total = 0
    saved_count = 0
    skipped_sequences = []
    for seq_key in tqdm(data_dict_train, desc="处理统一训练序列"):
        saved_count += process_sequence(
            data_dict_train[seq_key],
            seq_key,
            save_dir,
            bm,
            device=device,
            obj_mesh_dir=obj_mesh_dir,
            obj_points_count=obj_points_count,
            resolve_bimanual_contact_conflicts=resolve_bimanual_contact_conflicts,
            contact_max_hole_frames=contact_max_hole_frames,
            contact_min_duration_frames=contact_min_duration_frames,
            skip_object_displacement_without_hand_contact=skip_object_displacement_without_hand_contact,
            unheld_object_displacement_threshold_m=unheld_object_displacement_threshold_m,
            skipped_sequences=skipped_sequences,
            skip_record_context={"source_split": "train", "unified_index": total},
            output_name=f"{total}.pt",
        )
        total += 1

    train_source_count = total
    for seq_key in tqdm(data_dict_test, desc="处理统一测试序列"):
        saved_count += process_sequence(
            data_dict_test[seq_key],
            seq_key,
            save_dir,
            bm,
            device=device,
            obj_mesh_dir=obj_mesh_dir,
            obj_points_count=obj_points_count,
            resolve_bimanual_contact_conflicts=resolve_bimanual_contact_conflicts,
            contact_max_hole_frames=contact_max_hole_frames,
            contact_min_duration_frames=contact_min_duration_frames,
            skip_object_displacement_without_hand_contact=skip_object_displacement_without_hand_contact,
            unheld_object_displacement_threshold_m=unheld_object_displacement_threshold_m,
            skipped_sequences=skipped_sequences,
            skip_record_context={"source_split": "test", "unified_index": total},
            output_name=f"{total}.pt",
        )
        total += 1

    skip_manifest = {
        "source_train_sequences": int(train_source_count),
        "source_test_sequences": int(total - train_source_count),
        "source_total_sequences": int(total),
        "saved_sequences": int(saved_count),
        "skipped_sequences_count": int(len(skipped_sequences)),
        "contact_max_hole_frames": int(contact_max_hole_frames),
        "contact_min_duration_frames": int(contact_min_duration_frames),
        "unheld_object_displacement_threshold_m": float(
            unheld_object_displacement_threshold_m
        ),
        "skip_object_displacement_without_hand_contact": bool(
            skip_object_displacement_without_hand_contact
        ),
        "resolve_bimanual_contact_conflicts": bool(resolve_bimanual_contact_conflicts),
        "sequences": skipped_sequences,
    }
    skip_manifest_path = os.path.join(save_dir, "skipped_sequences.json")
    with open(skip_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(skip_manifest, handle, ensure_ascii=False, indent=2)

    return {
        "train_source_count": int(train_source_count),
        "source_total_count": int(total),
        "saved_count": int(saved_count),
        "skipped_sequences": skipped_sequences,
        "skip_manifest_path": skip_manifest_path,
    }


def split_unified_by_json(
    input_dir,
    split_info_path,
    train_dir,
    test_dir,
    copy_files=True,
    allowed_missing_files=None,
    output_split_info_path=None,
    enforce_numeric_names=True,
    allow_unassigned=False,
):
    """Split a flat directory of processed sequences according to an existing JSON.

    ``enforce_numeric_names`` keeps the OMOMO guarantee that the unified directory
    is exactly ``0.pt..N.pt``.  Datasets that keep their source sequence names
    (PAHOI) pass False and are checked against the JSON file list instead.

    ``allow_unassigned`` relaxes the two coverage checks for the case where the
    reference JSON only labels a subset of the corpus -- e.g. re-running with a
    corrected body model recovers sequences that the previous run's contact
    rules had dropped.  Those files stay in ``input_dir`` and are listed under
    ``unassigned_files`` in the output JSON rather than being force-assigned to
    a split, so the reference split's membership is reproduced exactly.
    """
    with open(split_info_path, "r", encoding="utf-8") as handle:
        split_info = json.load(handle)
    train_files = list(split_info.get("train_files", []))
    test_files = list(split_info.get("test_files", []))
    expected_total = int(split_info.get("total_files", len(train_files) + len(test_files)))
    all_names = train_files + test_files
    if len(all_names) != expected_total or len(set(all_names)) != expected_total:
        raise ValueError("split_info.json 的 train/test 文件列表不完整或存在重复")
    expected_names = set(all_names)
    if enforce_numeric_names:
        for name in all_names:
            # the sort keys below index by number, so every name must be numeric
            int(os.path.splitext(name)[0])
        if not allow_unassigned and expected_names != {
            f"{index}.pt" for index in range(expected_total)
        }:
            # A full reference split must enumerate the unified directory exactly.
            # With allow_unassigned the JSON is a subset by construction, so the
            # contiguity requirement does not apply.
            raise ValueError("split_info.json 文件名必须覆盖统一目录的 0.pt..N.pt")

    present_names = {
        name for name in os.listdir(input_dir)
        if name.endswith(".pt") and os.path.isfile(os.path.join(input_dir, name))
    }
    unexpected_names = present_names - expected_names
    if unexpected_names and not allow_unassigned:
        raise ValueError(f"统一目录包含 split_info.json 未定义的文件: {sorted(unexpected_names)}")

    missing_names = expected_names - present_names
    allowed_missing_files = set(allowed_missing_files or [])
    if allow_unassigned:
        # Only the direction that signals a real bug is fatal: a file the JSON
        # asks for that was neither produced nor deliberately skipped.
        unexpected_missing = sorted(missing_names - allowed_missing_files)
        if unexpected_missing:
            raise ValueError(
                f"统一目录缺失 split_info.json 要求的文件且未被标记跳过: {unexpected_missing}"
            )
    elif missing_names != allowed_missing_files:
        unexpected_missing = sorted(missing_names - allowed_missing_files)
        unexpectedly_present = sorted(allowed_missing_files - missing_names)
        raise ValueError(
            "统一目录缺失文件与已标记跳过列表不一致: "
            f"未授权缺失={unexpected_missing}, 标记但仍存在={unexpectedly_present}"
        )

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    operation = shutil.copy2 if copy_files else shutil.move
    kept_train_files = []
    kept_test_files = []
    for file_name, target_dir in ((name, train_dir) for name in train_files):
        source = os.path.join(input_dir, file_name)
        if not os.path.isfile(source):
            continue
        operation(source, os.path.join(target_dir, file_name))
        kept_train_files.append(file_name)
    for file_name, target_dir in ((name, test_dir) for name in test_files):
        source = os.path.join(input_dir, file_name)
        if not os.path.isfile(source):
            continue
        operation(source, os.path.join(target_dir, file_name))
        kept_test_files.append(file_name)

    kept_total = len(kept_train_files) + len(kept_test_files)
    filtered_split_info = {
        "train_ratio": len(kept_train_files) / kept_total if kept_total else 0.0,
        "test_ratio": len(kept_test_files) / kept_total if kept_total else 0.0,
        "total_files": kept_total,
        "source_total_files": expected_total,
        "skipped_files": sorted(
            missing_names,
            key=(
                (lambda name: int(os.path.splitext(name)[0]))
                if enforce_numeric_names
                else (lambda name: name)
            ),
        ),
        "train_files": kept_train_files,
        "test_files": kept_test_files,
        "source_split_info": os.path.abspath(split_info_path),
    }
    if allow_unassigned:
        name_key = (
            (lambda name: int(os.path.splitext(name)[0]))
            if enforce_numeric_names
            else (lambda name: name)
        )
        # Produced but not labelled by the reference split; left in input_dir.
        filtered_split_info["unassigned_files"] = sorted(unexpected_names, key=name_key)
        filtered_split_info["unassigned_count"] = len(unexpected_names)
    if output_split_info_path:
        output_split_dir = os.path.dirname(os.path.abspath(output_split_info_path))
        os.makedirs(output_split_dir, exist_ok=True)
        with open(output_split_info_path, "w", encoding="utf-8") as handle:
            json.dump(filtered_split_info, handle, ensure_ascii=False, indent=2)

    print(
        f"按 {split_info_path} 完成分割：训练 {len(kept_train_files)}，"
        f"测试 {len(kept_test_files)}，跳过 {len(missing_names)}"
        + (f"，未分配（参考 split 未覆盖，保留在统一目录）{len(unexpected_names)}"
           if allow_unassigned and unexpected_names else "")
    )
    return filtered_split_info


# def preprocess_amass(args, bm):

#     import pytorch3d.transforms as transforms
#     """
#     处理AMASS数据集，提取姿态、平移和IMU数据

#     参数:
#         args: 命令行参数，包含数据路径和保存目录
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"处理AMASS数据集，使用设备: {device}")

#     # 加载SMPL模型
#     bm_fname_male = os.path.join(args.support_dir, f"smplh/male/model.npz")
#     num_betas = 16
#     body_model = BodyModel(
#         bm_fname=bm_fname_male,
#         num_betas=num_betas,
#         model_type='smplh'
#     ).to(device)

#     # 创建AMASS独立的保存目录
#     # 将普通处理目录转换为AMASS处理目录
#     if "processed_data" in args.save_dir_train:
#         # 替换 processed_data 为 processed_amass_data
#         amass_base_dir = args.save_dir_train.replace("processed_data", "processed_amass_data").replace("/train", "")
#     else:
#         # 如果目录名不包含 processed_data，则在目录前添加 amass_ 前缀
#         dir_parts = args.save_dir_train.split('/')
#         if dir_parts[-1] == "train":
#             dir_parts = dir_parts[:-1]  # 移除train部分
#         dir_parts[-1] = f"amass_{dir_parts[-1]}"
#         amass_base_dir = '/'.join(dir_parts)

#     # 创建训练集、测试集和debug目录
#     amass_train_dir = os.path.join(amass_base_dir, "train")
#     amass_test_dir = os.path.join(amass_base_dir, "test")
#     amass_debug_dir = os.path.join(amass_base_dir, "debug")

#     os.makedirs(amass_train_dir, exist_ok=True)
#     os.makedirs(amass_test_dir, exist_ok=True)
#     os.makedirs(amass_debug_dir, exist_ok=True)

#     print(f"AMASS训练数据将保存到: {amass_train_dir}")
#     print(f"AMASS测试数据将保存到: {amass_test_dir}")
#     print(f"AMASS调试数据将保存到: {amass_debug_dir}")

#     # 定义AMASS数据集路径
#     amass_dirs = []
#     if hasattr(args, "amass_dirs") and args.amass_dirs:
#         amass_dirs = args.amass_dirs
#     else:
#         amass_dirs = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU',
#               'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
#               'BioMotionLab_NTroje', 'BMLhandball', 'MPI_Limits', 'DFaust67']

#     # 创建单独的聚合数据文件
#     sequence_index = 0
#     train_count = 0
#     test_count = 0
#     train_test_split_ratio = 0.9  # 90%训练，10%测试
#     amass_summary = {'datasets': [], 'total_sequences': 0, 'total_frames': 0, 'train_sequences': 0, 'test_sequences': 0}

#     # 遍历所有数据集目录
#     for ds_name in amass_dirs:
#         print(f'读取AMASS子数据集: {ds_name}')
#         ds_path = os.path.join(args.amass_dir, ds_name)
#         ds_summary = {'name': ds_name, 'sequences': 0, 'frames': 0}

#         if not os.path.exists(ds_path):
#             print(f"警告: 找不到数据集 {ds_path}")
#             continue

#         # 处理每个数据集中的每个文件
#         for npz_fname in tqdm(glob.glob(os.path.join(ds_path, '*/*_poses.npz'))):
#             try:
#                 # 清理GPU缓存
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()

#                 # 加载单个文件
#                 cdata = np.load(npz_fname)

#                 # 根据帧率调整采样
#                 framerate = int(cdata['mocap_framerate'])
#                 if framerate == 120:
#                     step = 2
#                 elif framerate == 60 or framerate == 59:
#                     step = 1
#                 else:
#                     continue  # 跳过其他帧率的数据

#                 # 获取姿势、平移和体形参数
#                 poses_data = cdata['poses'][::step].astype(np.float32)
#                 trans_data = cdata['trans'][::step].astype(np.float32)

#                 # 检查序列长度
#                 seq_len = poses_data.shape[0]
#                 if seq_len <= 120:
#                     print(f"\t丢弃太短的序列: {npz_fname}，长度为 {seq_len}")
#                     continue

#                 # 构建序列数据字典，适配 process_sequence 的格式
#                 seq_name = f"amass_{ds_name}_{sequence_index}"

#                 # 将poses_data分解为root_orient和pose_body
#                 root_orient = poses_data[:, :3]  # 根关节方向
#                 pose_body = poses_data[:, 3:66]  # 身体姿态 (前21个关节，每个3维)
#                 pose_mat = transforms.axis_angle_to_matrix(poses_data.reshape(seq_len, -1, 3))
#                 pose_6d = transforms.matrix_to_rotation_6d(pose_mat)

#                 smpl_input = {
#                 "root_orient": root_orient,
#                 "pose_body": pose_body,
#                 "trans": trans_data
#                 }
#                 # 使用SMPL模型获取全局姿态
#                 body_pose_world = bm(**smpl_input)
#                 position_global_full_gt_world = body_pose_world.Jtr[:, :22, :].cpu()
#                 # 只使用前22个关节的层次结构
#                 kintree_table = bm.kintree_table[0].long()[:22]  # 只取前22个关节的父子关系
#                 kintree_table[0] = -1

#                 # 计算全局旋转矩阵
#                 rotation_global_matrot = local2global_pose(
#                     pose_mat.reshape(seq_len, -1, 9), kintree_table
#                 )

#                 # 重塑全局旋转矩阵为标准形状 [T, J, 3, 3]
#                 rotation_global_matrot_reshaped = rotation_global_matrot.reshape(rotation_global_matrot.shape[0], -1, 3, 3)

#                 # 构建序列数据字典，添加物体数据（填充0值）以保持与其他数据集的一致性
#                 seq_data = {
#                     'seq_name': seq_name,
#                     'gender': 'neutral',
#                     # AMASS 不含物体交互，后续数据加载时需明确标记
#                     'has_object': False,
#                     'rotation_local_full_gt_list': pose_6d.reshape(seq_len, -1),
#                     'position_global_full_gt_world': position_global_full_gt_world.cpu().numpy(),
#                     'trans': trans_data,
#                     'rotation_global': rotation_global_matrot_reshaped.cpu(),
#                     # 物体相关字段占位，保持兼容但配合 has_object=False
#                     'obj_scale': np.ones(seq_len, dtype=np.float32),          # [T]
#                     'obj_trans': np.zeros((seq_len, 3), dtype=np.float32),     # [T, 3]
#                     'obj_rot': np.tile(np.eye(3, dtype=np.float32), (seq_len, 1, 1)),  # [T, 3, 3]
#                     'obj_com_pos': np.zeros((seq_len, 3), dtype=np.float32),   # [T, 3]
#                 }

#                 # 决定分配到训练集还是测试集
#                 # 使用简单的取模方式：每10个序列中前9个分配给训练集，最后1个分配给测试集
#                 is_train = (sequence_index % 10) < (train_test_split_ratio * 10)
#                 target_dir = amass_train_dir if is_train else amass_test_dir
#                 seq_prefix = "train" if is_train else "test"

#                 # 使用 process_sequence 函数处理序列
#                 # 注意：传递空的obj_mesh_dir，因为AMASS数据已经包含了填充的物体数据
#                 success = process_sequence(
#                     seq_data=seq_data,
#                     seq_key=f"{seq_prefix}_{sequence_index}",
#                     save_dir=target_dir,
#                     bm=body_model,
#                     device=device,
#                     bps_dir=None,  # AMASS不使用BPS
#                     bps_points=None,  # AMASS不使用BPS
#                     obj_mesh_dir=""  # 传递空字符串，跳过接触检测
#                 )

#                 if success:
#                     # 更新统计信息
#                     sequence_index += 1
#                     ds_summary['sequences'] += 1
#                     ds_summary['frames'] += seq_len
#                     amass_summary['total_sequences'] += 1
#                     amass_summary['total_frames'] += seq_len

#                     if is_train:
#                         train_count += 1
#                         amass_summary['train_sequences'] += 1
#                     else:
#                         test_count += 1
#                         amass_summary['test_sequences'] += 1

#             except Exception as e:
#                 print(f"处理文件时出错 {npz_fname}: {e}")
#                 import traceback
#                 traceback.print_exc()

#         # 添加数据集摘要
#         amass_summary['datasets'].append(ds_summary)
#         print(f"完成处理数据集 {ds_name}: {ds_summary['sequences']} 个序列, {ds_summary['frames']} 帧")

#     print(f"AMASS数据处理完成，共处理了 {amass_summary['total_sequences']} 个序列")
#     print(f"  训练集: {amass_summary['train_sequences']} 个序列")
#     print(f"  测试集: {amass_summary['test_sequences']} 个序列")
#     print(f"预处理后的AMASS数据集保存在:")
#     print(f"  训练集: {amass_train_dir}")
#     print(f"  测试集: {amass_test_dir}")

#     # 从训练集中提取前10个文件到debug目录
#     print("\n开始创建debug数据集...")
#     import shutil

#     train_files = [f for f in os.listdir(amass_train_dir) if f.endswith('.pt')]
#     train_files.sort()  # 按文件名排序

#     debug_count = min(10, len(train_files))  # 最多10个文件
#     for i in range(debug_count):
#         src_file = os.path.join(amass_train_dir, train_files[i])
#         dst_file = os.path.join(amass_debug_dir, train_files[i])
#         shutil.copy2(src_file, dst_file)
#         print(f"  复制调试文件: {train_files[i]}")

#     print(f"调试数据集创建完成，共 {debug_count} 个文件保存在: {amass_debug_dir}")

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # OMOMO family detection remains auditable, but every sequence is evaluated
    # with the verified SMPL-X neutral template and explicit zero betas.
    print("准备 canonical SMPL-X neutral 身体模型（zero betas）...")
    body_models = BodyModelRegistry(
        args.support_dir,
        device=device,
        num_betas=16,
        default_type=args.body_model_type,
        default_gender=CANONICAL_BODY_MODEL_GENDER,
    )
    if CANONICAL_BODY_MODEL_GENDER not in body_models.available_genders(
        _OMOMO_CANONICAL_BODY_MODEL_TYPE
    ):
        raise FileNotFoundError(
            f"SMPL-X neutral model not found under {args.support_dir}; "
            "gender fallback is disabled for canonical OMOMO preprocessing."
        )

    # 处理AMASS数据集
    # if args.process_amass:
    #     print("开始处理AMASS数据集...")
    #     preprocess_amass(args, bm_male)
    #     print("AMASS数据集处理完成!")
    #     return


    # # 创建BPS目录
    # bps_dir = os.path.join(args.save_dir, "bps_features")
    # os.makedirs(bps_dir, exist_ok=True)
    # 准备BPS基础点云
    # print("准备BPS基础点云...")
    # bps_points = prep_bps_data(n_bps_points=args.n_bps_points, radius=args.bps_radius, device=device)

    # 加载数据集
    print(f"正在加载训练数据集：{args.data_path_train}")
    data_dict_train = joblib.load(args.data_path_train)
    print(f"训练数据集加载完成，共有{len(data_dict_train)}个序列")

    print(f"正在加载测试数据集：{args.data_path_test}")
    data_dict_test = joblib.load(args.data_path_test)
    print(f"测试数据集加载完成，共有{len(data_dict_test)}个序列")

    # OMOMO must first be numbered as one concatenated dataset.  The source
    # train/test pickles both start at key 0, so processing them directly into
    # separate directories would make the numeric IDs incompatible with the
    # checked-in split_info.json.
    if args.save_dir_unified:
        process_summary = _process_unified_omomo(
            data_dict_train,
            data_dict_test,
            args.save_dir_unified,
            body_models,
            device,
            args.obj_mesh_dir,
            args.obj_points_count,
            args.resolve_bimanual_contact_conflicts,
            args.contact_max_hole_frames,
            args.contact_min_duration_frames,
            args.skip_object_displacement_without_hand_contact,
            args.unheld_object_displacement_threshold_m,
        )
        print(
            "统一序列处理完成："
            f"源训练 {process_summary['train_source_count']}，"
            f"源总计 {process_summary['source_total_count']}，"
            f"保存 {process_summary['saved_count']}，"
            f"跳过 {len(process_summary['skipped_sequences'])}，"
            f"目录：{args.save_dir_unified}"
        )
        if args.split_info:
            output_split_info_path = args.output_split_info
            if output_split_info_path is None:
                output_root = os.path.commonpath(
                    [os.path.abspath(args.save_dir_train), os.path.abspath(args.save_dir_test)]
                )
                output_split_info_path = os.path.join(output_root, "split_info.json")
            split_unified_by_json(
                args.save_dir_unified,
                args.split_info,
                args.save_dir_train,
                args.save_dir_test,
                copy_files=not args.move_after_split,
                allowed_missing_files={
                    record["output_file"] for record in process_summary["skipped_sequences"]
                },
                output_split_info_path=output_split_info_path,
                allow_unassigned=args.allow_unassigned,
            )
            output_root = os.path.dirname(os.path.abspath(output_split_info_path))
            shutil.copy2(
                process_summary["skip_manifest_path"],
                os.path.join(output_root, "skipped_sequences.json"),
            )
        print("所有数据处理完成!")
        return

    # 创建旧式分目录输出（保留兼容性）
    os.makedirs(args.save_dir_train, exist_ok=True)
    os.makedirs(args.save_dir_test, exist_ok=True)

    # 处理所有序列
    print("开始处理训练序列...")

    for seq_key in tqdm(data_dict_train, desc="处理训练序列"):
        process_sequence(
            data_dict_train[seq_key],
            seq_key,
            args.save_dir_train,
            body_models,
            device=device,
            obj_mesh_dir=args.obj_mesh_dir,
            obj_points_count=args.obj_points_count,
            resolve_bimanual_contact_conflicts=args.resolve_bimanual_contact_conflicts,
            contact_max_hole_frames=args.contact_max_hole_frames,
            contact_min_duration_frames=args.contact_min_duration_frames,
            skip_object_displacement_without_hand_contact=(
                args.skip_object_displacement_without_hand_contact
            ),
            unheld_object_displacement_threshold_m=(
                args.unheld_object_displacement_threshold_m
            ),
        )

    print(f"训练序列处理完成，结果保存在：{args.save_dir_train}")

    print("开始处理测试序列...")
    for seq_key in tqdm(data_dict_test, desc="处理测试序列"):
        process_sequence(
            data_dict_test[seq_key],
            seq_key,
            args.save_dir_test,
            body_models,
            device=device,
            obj_mesh_dir=args.obj_mesh_dir,
            obj_points_count=args.obj_points_count,
            resolve_bimanual_contact_conflicts=args.resolve_bimanual_contact_conflicts,
            contact_max_hole_frames=args.contact_max_hole_frames,
            contact_min_duration_frames=args.contact_min_duration_frames,
            skip_object_displacement_without_hand_contact=(
                args.skip_object_displacement_without_hand_contact
            ),
            unheld_object_displacement_threshold_m=(
                args.unheld_object_displacement_threshold_m
            ),
        )

    print(f"测试序列处理完成，结果保存在：{args.save_dir_test}")
    print("所有数据处理完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理人体动作数据集")
    parser.add_argument("--data_path_train", type=str, default="datasets/OMOMO/train_diffusion_manip_seq_joints24.p",
                        help="输入数据集路径(.p文件)")
    parser.add_argument("--data_path_test", type=str, default="datasets/OMOMO/test_diffusion_manip_seq_joints24.p",
                        help="输入数据集路径(.p文件)")
    parser.add_argument("--save_dir_train", type=str, default="process/processed_split_data_OMOMO_bps/train",
                        help="输出数据保存目录")
    parser.add_argument("--save_dir_test", type=str, default="process/processed_split_data_OMOMO_bps/test",
                        help="输出数据保存目录")
    parser.add_argument(
        "--save_dir_unified",
        type=str,
        default=None,
        help="先将 train/test 输入按 train 后 test 的顺序编号并保存到同一目录（OMOMO 推荐）。",
    )
    parser.add_argument(
        "--split_info",
        type=str,
        default=None,
        help="统一处理后按已有 split_info.json 的文件名列表分割到 save_dir_train/save_dir_test。",
    )
    parser.add_argument(
        "--move-after-split",
        action="store_true",
        help="按 split_info 分割时移动文件；默认复制，保留统一目录作为可审计副本。",
    )
    parser.add_argument("--support_dir", type=str, default="datasets/smpl_models",
                        help="SMPL模型目录")
    parser.add_argument(
        "--body-model-type",
        type=str,
        default="smplx",
        choices=("smplh", "smplx", "smpl"),
        help=(
            "序列没有可甄别线索时使用的模型族。OMOMO 的 canonical family "
            "必须为 smplx；所有实例固定为 neutral、zero betas。"
        ),
    )
    parser.add_argument("--obj_mesh_dir", type=str, default="datasets/OMOMO/captured_objects",
                        help="物体网格目录")
    parser.add_argument("--obj_points_count", type=int, default=256,
                        help="canonical物体点云点数（保存到pt）")
    parser.add_argument(
        "--contact-max-hole-frames",
        type=int,
        default=30,
        help="填补两侧均为接触的内部无接触孔洞的最大帧数（默认：30）。",
    )
    parser.add_argument(
        "--contact-min-duration-frames",
        type=int,
        default=30,
        help="保留接触段所需的最小连续帧数（默认：30，恰好30帧会保留）。",
    )
    skip_motion_group = parser.add_mutually_exclusive_group()
    skip_motion_group.add_argument(
        "--skip-object-displacement-without-hand-contact",
        dest="skip_object_displacement_without_hand_contact",
        action="store_true",
        help="跳过任一无手接触区间内物体位移达到阈值的序列。",
    )
    skip_motion_group.add_argument(
        "--keep-object-displacement-without-hand-contact",
        dest="skip_object_displacement_without_hand_contact",
        action="store_false",
        help="保留无手接触区间内物体发生大位移的序列。",
    )
    parser.set_defaults(skip_object_displacement_without_hand_contact=True)
    parser.add_argument(
        "--unheld-object-displacement-threshold-m",
        type=float,
        default=0.10,
        help="无手接触连续区间内触发整序列排除的物体最大位移，单位米（默认：0.10）。",
    )
    parser.add_argument(
        "--allow-unassigned",
        action="store_true",
        help=(
            "参考 split_info 只覆盖部分语料时，允许本次新产出的未标注序列保留在统一目录，"
            "并记入输出 split_info 的 unassigned_files，而不是报错。"
        ),
    )
    parser.add_argument(
        "--output-split-info",
        type=str,
        default=None,
        help="过滤后 split_info.json 的输出路径；默认写入 train/test 的共同父目录。",
    )
    resolve_group = parser.add_mutually_exclusive_group()
    resolve_group.add_argument(
        "--resolve-bimanual-contact-conflicts",
        dest="resolve_bimanual_contact_conflicts",
        action="store_true",
        help="在完整序列预处理阶段，双手同时接触时只保留骨长变化更小的一只手。",
    )
    resolve_group.add_argument(
        "--no-resolve-bimanual-contact-conflicts",
        dest="resolve_bimanual_contact_conflicts",
        action="store_false",
        help="保留原始双手接触标签，不做序列级裁决。",
    )
    parser.set_defaults(resolve_bimanual_contact_conflicts=False)
    parser.add_argument("--process_amass", action="store_true",
                        help="是否处理AMASS数据集")
    parser.add_argument("--amass_dir", type=str, default="/mnt/d/a_WORK/Projects/PhD/datasets/AMASS_SMPL_H",
                        help="AMASS数据集路径")
    parser.add_argument("--amass_dirs", type=str, nargs="+",
                        help="AMASS子数据集列表，例如CMU BMLmovi")

    args = parser.parse_args()
    main(args)
