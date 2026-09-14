"""Shared helpers for human pose modules."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, Optional, Sequence

import torch

from configs import _IGNORED_INDICES, _REDUCED_INDICES, _SENSOR_ROT_INDICES
from utils.rotation_conversions import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix


SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
WRIST_JOINT_INDICES = (20, 21)
FOREARM_PARENT_JOINT_INDICES = (18, 19)
VIRTUAL_PALM_JOINT_INDICES = (22, 23)

# Palm anchors, per native joint layout.  For SMPL-H/SMPL-X these are the
# *middle-finger* MCP joints: they sit on the wrist's central axis (rest-pose z
# offset ~2 mm) and are the closest single native joint to the palm centre.  The
# index-finger MCP used previously (SMPL-H 22 / SMPL-X 25) is 2.2 cm off towards
# the thumb and 4.2 cm from SMPL's own hand joint.  OMOMO's per-subject
# ``rest_offsets[22:24]`` are exactly the SMPL-X middle1 joints (verified across
# all 17 subjects, 0.06-3.7 mm), so this choice also makes the dataset's own
# skeleton definition reproducible without a special case.
SMPL_PALM_JOINT_INDICES = (22, 23)   # plain SMPL: joints 22/23 already are the hands
SMPLH_PALM_JOINT_INDICES = (25, 40)  # SMPL-H: L/R middle1
SMPLX_PALM_JOINT_INDICES = (28, 43)  # SMPL-X: L/R middle1 (jaw/eyes occupy 22..24)
SMPL_NATIVE_JOINT_COUNT = 24
SMPLH_NATIVE_JOINT_COUNT = 52
SMPLX_NATIVE_JOINT_COUNT = 55
CANONICAL_BODY_MODEL_GENDER = 'neutral'
CANONICAL_BETAS = 'zero'
CANONICAL_SKELETON_VERSION = 1
# Native joint layouts are not interchangeable: SMPL-X inserts jaw/eye joints at
# 22..24 before the hand chains, so reusing the SMPL-H indices on SMPL-X output
# silently returns two left-hand joints instead of the left and right palms.
PALM_JOINT_INDICES_BY_NATIVE_COUNT = {
    SMPL_NATIVE_JOINT_COUNT: SMPL_PALM_JOINT_INDICES,
    SMPLH_NATIVE_JOINT_COUNT: SMPLH_PALM_JOINT_INDICES,
    SMPLX_NATIVE_JOINT_COUNT: SMPLX_PALM_JOINT_INDICES,
}
BODY_MODEL_TYPE_BY_NATIVE_COUNT = {
    SMPL_NATIVE_JOINT_COUNT: "smpl",
    SMPLH_NATIVE_JOINT_COUNT: "smplh",
    SMPLX_NATIVE_JOINT_COUNT: "smplx",
}
NATIVE_JOINT_COUNT_BY_BODY_MODEL_TYPE = {
    value: key for key, value in BODY_MODEL_TYPE_BY_NATIVE_COUNT.items()
}
# Width of a flattened axis-angle ``poses`` array, used to identify the family a
# raw capture was fitted with: 24*3, 52*3 and 55*3 respectively.
POSE_WIDTH_BY_BODY_MODEL_TYPE = {"smpl": 72, "smplh": 156, "smplx": 165}
BODY_MODEL_TYPE_BY_POSE_WIDTH = {
    value: key for key, value in POSE_WIDTH_BY_BODY_MODEL_TYPE.items()
}
SUPPORTED_BODY_MODEL_TYPES = ("smplx", "smplh", "smpl")
SMPL24_JOINT_COUNT = 24
# Estimated from zero-pose SMPL-H middle metacarpal offsets. The virtual palm
# anchor intentionally follows the forearm extension instead of native SMPL-H
# hand-joint ordering, where Jtr[22:24] are not left/right palm roots.
PALM_EXTENSION_RATIOS = (0.43, 0.42)
_NO_SHAPE_ARG = object()


def compute_virtual_palm_positions(
    joints: torch.Tensor,
    *,
    forearm_parent_indices: Sequence[int] = FOREARM_PARENT_JOINT_INDICES,
    wrist_indices: Sequence[int] = WRIST_JOINT_INDICES,
    extension_ratios: Sequence[float] = PALM_EXTENSION_RATIOS,
) -> torch.Tensor:
    """Compute left/right virtual palm anchors from elbow and wrist positions."""
    if not isinstance(joints, torch.Tensor):
        raise TypeError("joints must be a torch.Tensor")
    if joints.shape[-1] != 3:
        raise ValueError(f"joints must end with xyz coordinates, got {joints.shape}")

    joint_count = int(joints.shape[-2])
    max_wrist_idx = max(int(idx) for idx in wrist_indices)
    if joint_count <= max_wrist_idx:
        shape = (*joints.shape[:-2], 2, 3)
        return torch.zeros(shape, device=joints.device, dtype=joints.dtype)

    wrists = torch.stack(
        [joints[..., int(wrist_indices[0]), :], joints[..., int(wrist_indices[1]), :]],
        dim=-2,
    )

    max_parent_idx = max(int(idx) for idx in forearm_parent_indices)
    if joint_count <= max_parent_idx:
        return wrists

    forearms = torch.stack(
        [
            joints[..., int(wrist_indices[0]), :] - joints[..., int(forearm_parent_indices[0]), :],
            joints[..., int(wrist_indices[1]), :] - joints[..., int(forearm_parent_indices[1]), :],
        ],
        dim=-2,
    )
    ratios = torch.as_tensor(extension_ratios, device=joints.device, dtype=joints.dtype)
    ratio_shape = (1,) * (forearms.dim() - 2) + (2, 1)
    return wrists + forearms * ratios.view(ratio_shape)


def append_virtual_palm_joints(
    joints: torch.Tensor,
    *,
    num_joints: int = 24,
    palm_indices: Sequence[int] = VIRTUAL_PALM_JOINT_INDICES,
) -> torch.Tensor:
    """Return a custom 24-joint tensor whose 22/23 entries are virtual palms."""
    if joints.shape[-1] != 3:
        raise ValueError(f"joints must end with xyz coordinates, got {joints.shape}")

    joint_count = int(joints.shape[-2])
    if joint_count >= num_joints:
        return joints[..., :num_joints, :].clone()
    else:
        pad_shape = (*joints.shape[:-2], num_joints - joint_count, 3)
        pad = torch.zeros(pad_shape, device=joints.device, dtype=joints.dtype)
        out = torch.cat((joints, pad), dim=-2)

    palms = compute_virtual_palm_positions(joints)
    left_idx, right_idx = (int(palm_indices[0]), int(palm_indices[1]))
    if left_idx < num_joints:
        out[..., left_idx, :] = palms[..., 0, :]
    if right_idx < num_joints:
        out[..., right_idx, :] = palms[..., 1, :]
    return out


def append_virtual_palm_rotations(
    rotations: torch.Tensor,
    *,
    num_joints: int = 24,
    wrist_indices: Sequence[int] = WRIST_JOINT_INDICES,
    palm_indices: Sequence[int] = VIRTUAL_PALM_JOINT_INDICES,
) -> torch.Tensor:
    """Append virtual palm global rotations by inheriting wrist rotations."""
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"rotations must end with [3,3], got {rotations.shape}")

    joint_count = int(rotations.shape[-3])
    if joint_count >= num_joints:
        return rotations[..., :num_joints, :, :].clone()
    else:
        eye = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
        view_shape = (1,) * (rotations.dim() - 3) + (1, 3, 3)
        pad_shape = (*rotations.shape[:-3], num_joints - joint_count, 3, 3)
        pad = eye.view(view_shape).expand(pad_shape)
        out = torch.cat((rotations, pad), dim=-3)

    left_wrist, right_wrist = (int(wrist_indices[0]), int(wrist_indices[1]))
    left_palm, right_palm = (int(palm_indices[0]), int(palm_indices[1]))
    if joint_count > left_wrist and left_palm < num_joints:
        out[..., left_palm, :, :] = rotations[..., left_wrist, :, :]
    if joint_count > right_wrist and right_palm < num_joints:
        out[..., right_palm, :, :] = rotations[..., right_wrist, :, :]
    return out


def select_hand_anchor_positions(joints: torch.Tensor) -> torch.Tensor:
    """Select the SMPL-24 left/right palm positions used by VC/OT."""
    if not isinstance(joints, torch.Tensor):
        raise TypeError("joints must be a torch.Tensor")
    if joints.shape[-1] != 3:
        raise ValueError(f"joints must end with xyz coordinates, got {joints.shape}")
    if int(joints.shape[-2]) < SMPL24_JOINT_COUNT:
        raise ValueError(
            "SMPL-24 palm positions require joints 22/23; "
            f"got only {int(joints.shape[-2])} joints"
        )
    return joints[..., list(VIRTUAL_PALM_JOINT_INDICES), :]


def build_smpl24_joint_positions(joints: torch.Tensor) -> torch.Tensor:
    """Map native SMPL/SMPL-H/SMPL-X joints to body-22 plus left/right palms.

    The palm indices are resolved from the native joint count, so the caller
    never has to know which family produced ``joints``.  A 24-joint (plain SMPL)
    input passes through unchanged, because its joints 22/23 already are the
    hands.
    """
    if not isinstance(joints, torch.Tensor):
        raise TypeError("joints must be a torch.Tensor")
    if joints.shape[-1] != 3:
        raise ValueError(f"joints must end with xyz coordinates, got {joints.shape}")
    native_joint_count = int(joints.shape[-2])
    palm_indices = PALM_JOINT_INDICES_BY_NATIVE_COUNT.get(native_joint_count)
    if palm_indices is None:
        raise ValueError(
            "native joints must come from SMPL "
            f"({SMPL_NATIVE_JOINT_COUNT}), SMPL-H ({SMPLH_NATIVE_JOINT_COUNT}) or "
            f"SMPL-X ({SMPLX_NATIVE_JOINT_COUNT}) so the palm indices are "
            f"unambiguous, got {native_joint_count}"
        )
    palms = joints[..., list(palm_indices), :]
    return torch.cat((joints[..., :22, :], palms), dim=-2)


def ensure_smpl24_local_rotations(rotations: torch.Tensor) -> torch.Tensor:
    """Pad 22-joint local rotations with identity metacarpal rotations."""
    if not isinstance(rotations, torch.Tensor):
        raise TypeError("rotations must be a torch.Tensor")
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"rotations must end with [3,3], got {rotations.shape}")
    joint_count = int(rotations.shape[-3])
    if joint_count >= SMPL24_JOINT_COUNT:
        return rotations[..., :SMPL24_JOINT_COUNT, :, :]
    if joint_count != 22:
        raise ValueError(f"expected 22 or 24 local rotations, got {joint_count}")

    identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
    pad_shape = (*rotations.shape[:-3], 2, 3, 3)
    identity = identity.view((1,) * (rotations.dim() - 3) + (1, 3, 3))
    identity = identity.expand(pad_shape)
    return torch.cat((rotations, identity), dim=-3)


def local_rotations_to_global(
    local_rotations: torch.Tensor,
    parents: Sequence[int] = SMPL_PARENTS,
) -> torch.Tensor:
    """Convert local SMPL rotations to global rotations for arbitrary batch dims."""
    if local_rotations.shape[-2:] != (3, 3):
        raise ValueError(f"local_rotations must end with [3,3], got {local_rotations.shape}")
    rotations = local_rotations.reshape(-1, local_rotations.shape[-3], 3, 3)
    if rotations.shape[1] > len(parents):
        raise ValueError(
            f"rotation joint count {rotations.shape[1]} exceeds parent table size {len(parents)}"
        )
    global_rotations = torch.empty_like(rotations)
    global_rotations[:, 0] = rotations[:, 0]
    for joint_idx in range(1, rotations.shape[1]):
        parent_idx = int(parents[joint_idx])
        global_rotations[:, joint_idx] = torch.matmul(
            global_rotations[:, parent_idx], rotations[:, joint_idx]
        )
    return global_rotations.reshape(*local_rotations.shape[:-3], rotations.shape[1], 3, 3)


def select_wrist_positions(
    joints: torch.Tensor,
    *,
    wrist_indices: Sequence[int] = WRIST_JOINT_INDICES,
) -> torch.Tensor:
    """Select the left/right wrist positions used for wrist-mounted IMU supervision."""
    if not isinstance(joints, torch.Tensor):
        raise TypeError("joints must be a torch.Tensor")
    if joints.shape[-1] != 3:
        raise ValueError(f"joints must end with xyz coordinates, got {joints.shape}")

    joint_count = int(joints.shape[-2])
    max_wrist_idx = max(int(idx) for idx in wrist_indices)
    shape = (*joints.shape[:-2], 2, 3)
    if joint_count <= max_wrist_idx:
        return torch.zeros(shape, device=joints.device, dtype=joints.dtype)
    return torch.stack(
        [joints[..., int(wrist_indices[0]), :], joints[..., int(wrist_indices[1]), :]],
        dim=-2,
    )


def load_body_model(body_model_path: str, num_betas: int = 16):
    """Load a frozen BodyModel instance.

    ``num_betas`` is clamped to what the file actually ships, so pointing this at
    a plain SMPL model (10 shape dirs) does not blow up inside ``lbs``.
    """
    from human_body_prior.body_model.body_model import BodyModel

    body_model = BodyModel(
        bm_fname=body_model_path,
        num_betas=clamp_num_betas(body_model_path, num_betas),
    )
    body_model.eval()
    for param in body_model.parameters():
        param.requires_grad_(False)
    return body_model


# --- body model family resolution -----------------------------------------
# ``support_dir`` layouts differ per family: AMASS ships SMPL-H as
# ``smplh/<gender>/model.npz`` while the SMPL-X release ships
# ``smplx/SMPLX_<GENDER>.npz``.  Try both spellings so a support directory
# assembled either way resolves.
_BODY_MODEL_PATH_TEMPLATES = {
    "smplh": ("smplh/{gender}/model.npz", "smplh/SMPLH_{GENDER}.npz"),
    "smplx": ("smplx/SMPLX_{GENDER}.npz", "smplx/{gender}/model.npz"),
    "smpl": ("smpl/{gender}/model.npz", "smpl/SMPL_{GENDER}.npz"),
}
_BODY_MODEL_TYPE_ALIASES = {
    "smpl": "smpl",
    "smplh": "smplh",
    "smpl+h": "smplh",
    "smpl_h": "smplh",
    "smplh16": "smplh",
    "smplx": "smplx",
    "smpl-x": "smplx",
    "smpl_x": "smplx",
    "smplx16": "smplx",
}
_GENDER_ALIASES = {"m": "male", "f": "female", "n": "neutral", "": "neutral"}


def normalize_body_model_type(model_type) -> Optional[str]:
    """Map a free-form model-type string onto smpl/smplh/smplx."""
    if model_type is None:
        return None
    text = str(model_type).strip().strip("'\"").lower().replace(" ", "")
    return _BODY_MODEL_TYPE_ALIASES.get(text)


def normalize_gender(gender) -> str:
    if gender is None:
        return "neutral"
    text = str(gender).strip().strip("'\"").lower()
    text = _GENDER_ALIASES.get(text, text)
    return text if text in ("male", "female", "neutral") else "neutral"


def resolve_body_model_path(
    support_dir: str,
    model_type: str,
    gender: str = "neutral",
) -> Optional[str]:
    """Return the on-disk path for a (family, gender) pair, or None if absent."""
    import os

    resolved_type = normalize_body_model_type(model_type)
    if resolved_type is None:
        raise ValueError(f"unsupported body model type: {model_type!r}")
    gender = normalize_gender(gender)
    for template in _BODY_MODEL_PATH_TEMPLATES[resolved_type]:
        candidate = os.path.join(
            support_dir, template.format(gender=gender, GENDER=gender.upper())
        )
        if os.path.isfile(candidate):
            return candidate
    return None


def body_model_type_of(body_model) -> Optional[str]:
    """Read the family off a loaded BodyModel (it infers it from posedirs)."""
    return normalize_body_model_type(getattr(body_model, "model_type", None))


def clamp_num_betas(body_model_path: str, num_betas: int) -> int:
    """Clamp a requested beta count to what the model file provides.

    SMPL-X packs expression dirs after the first 300 shape dirs, so only the
    plain SMPL case (10) actually constrains anything in practice.
    """
    import numpy as np

    try:
        with np.load(body_model_path, allow_pickle=True) as data:
            available = int(data["shapedirs"].shape[-1])
    except Exception:
        return int(num_betas)
    if available > 300:  # SMPL-X: 300 shape + expression dirs
        available = 300
    return int(min(int(num_betas), available))


def infer_body_model_type_from_pose_width(pose_width) -> Optional[str]:
    """Identify a family from the width of a flattened axis-angle pose array."""
    if pose_width is None:
        return None
    return BODY_MODEL_TYPE_BY_POSE_WIDTH.get(int(pose_width))


def is_neutral_body_model_path(path) -> bool:
    """Return whether a standard SMPL-family path names a neutral template."""
    normalized = str(path).replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    return "/neutral/" in normalized or "_neutral." in name


def canonical_skeleton_metadata(model_type: str) -> dict:
    resolved = normalize_body_model_type(model_type)
    if resolved is None:
        raise ValueError(f"Unsupported canonical body-model family: {model_type!r}")
    return {
        "body_model_type": resolved,
        "gender": CANONICAL_BODY_MODEL_GENDER,
        "canonical_body_model_gender": CANONICAL_BODY_MODEL_GENDER,
        "canonical_betas": CANONICAL_BETAS,
        "canonical_skeleton_version": CANONICAL_SKELETON_VERSION,
    }

def load_body_models_from_cfg(cfg, num_betas: int = 16):
    """Load every body model a config declares, keyed by family.

    Accepts either the legacy single ``body_model_path`` or a
    ``body_model_paths: {smplh: ..., smplx: ...}`` mapping (or both).  Returning
    a dict lets the caller run FK in whichever family produced a given sample:
    OMOMO/PAHOI ground truth is SMPL-X while AMASS/BEHAVE/HODome/IMHD is SMPL-H,
    and the two skeletons differ by ~3.7 cm per joint even at zero betas.

    Returns ``(models, default_type)``.  ``default_type`` is the family used for
    samples that do not declare one, and is the legacy path's family when given.
    """
    models = {}
    default_type = None

    if isinstance(cfg, Mapping):
        canonical_contract = bool(cfg.get("require_canonical_skeleton", False))
        canonical_contract = canonical_contract or "canonical_body_model_gender" in cfg
        canonical_contract = canonical_contract or "canonical_betas" in cfg
        canonical_gender_value = cfg.get(
            "canonical_body_model_gender", CANONICAL_BODY_MODEL_GENDER
        )
        canonical_betas_value = cfg.get("canonical_betas", CANONICAL_BETAS)
        single_path = cfg.get("body_model_path")
        mapping = cfg.get("body_model_paths") or {}
    else:
        canonical_contract = bool(getattr(cfg, "require_canonical_skeleton", False))
        canonical_contract = canonical_contract or hasattr(cfg, "canonical_body_model_gender")
        canonical_contract = canonical_contract or hasattr(cfg, "canonical_betas")
        canonical_gender_value = getattr(
            cfg, "canonical_body_model_gender", CANONICAL_BODY_MODEL_GENDER
        )
        canonical_betas_value = getattr(cfg, "canonical_betas", CANONICAL_BETAS)
        single_path = getattr(cfg, "body_model_path", None)
        mapping = getattr(cfg, "body_model_paths", None) or {}
    canonical_gender = str(canonical_gender_value).strip().lower()
    canonical_betas = str(canonical_betas_value).strip().lower()
    if canonical_contract and canonical_gender != CANONICAL_BODY_MODEL_GENDER:
        raise ValueError(
            "Training/inference body models must use the neutral canonical template; "
            f"got canonical_body_model_gender={canonical_gender!r}."
        )
    if canonical_contract and canonical_betas != CANONICAL_BETAS:
        raise ValueError(
            "Training/inference body models must use zero canonical betas; "
            f"got canonical_betas={canonical_betas!r}."
        )

    if not isinstance(mapping, dict):
        raise ValueError(
            f"body_model_paths must be a mapping of family -> path, got {type(mapping).__name__}"
        )

    for raw_type, path in mapping.items():
        model_type = normalize_body_model_type(raw_type)
        if model_type is None:
            raise ValueError(f"body_model_paths has an unsupported family: {raw_type!r}")
        if not path:
            continue
        if canonical_contract and not is_neutral_body_model_path(path):
            raise ValueError(
                f"body_model_paths[{raw_type!r}] must point to a neutral model: {path}"
            )
        model = load_body_model(str(path), num_betas=num_betas)
        found = body_model_type_of(model)
        if found is not None and found != model_type:
            raise ValueError(
                f"body_model_paths['{raw_type}'] points at a {found} model: {path}"
            )
        models[model_type] = model

    if single_path:
        if canonical_contract and not is_neutral_body_model_path(single_path):
            raise ValueError(f"body_model_path must point to a neutral model: {single_path}")
        model = load_body_model(str(single_path), num_betas=num_betas)
        default_type = body_model_type_of(model)
        if default_type is None:
            raise ValueError(f"Could not identify the family of {single_path}")
        # An explicit mapping entry wins, so the two never diverge.
        models.setdefault(default_type, model)

    if not models:
        raise ValueError("Set body_model_path or body_model_paths in the config")
    if default_type is None:
        # Prefer SMPL-X when no legacy path pins the default: it is the family of
        # the datasets this repo trains on by default.
        for candidate in SUPPORTED_BODY_MODEL_TYPES:
            if candidate in models:
                default_type = candidate
                break
    return models, default_type


# Fields a capture may use to declare which body model it was fitted with.
_MODEL_TYPE_FIELDS = ("surface_model_type", "body_model_type", "model_type", "model", "smpl_type")
# Fields holding a flattened axis-angle pose, widest layout first.
_POSE_FIELDS = ("poses", "pose", "full_pose")


def _scalar_to_str(value) -> Optional[str]:
    """Unwrap numpy 0-d arrays / bytes into a plain string."""
    import numpy as np

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if isinstance(value, str):
        return value
    return None


class BodyModelRegistry:
    """Lazily loads and caches BodyModels keyed by (family, gender).

    Datasets in this repo are a mix of SMPL-H (AMASS, BEHAVE, HODome, IMHD) and
    SMPL-X (OMOMO, PAHOI) fits.  Running one family's parameters through the
    other silently displaces the whole skeleton -- the root joint alone differs
    by 12.7 cm between SMPL-H and SMPL-X -- so every call site resolves the
    family from the sequence instead of hard-coding a path.
    """

    def __init__(
        self,
        support_dir: str,
        device=None,
        num_betas: int = 16,
        default_type: str = "smplh",
        default_gender: str = "neutral",
    ) -> None:
        self.support_dir = str(support_dir)
        self.device = device
        self.num_betas = int(num_betas)
        self.default_type = normalize_body_model_type(default_type) or "smplh"
        self.default_gender = normalize_gender(default_gender)
        self._models = {}
        self._rest_joints = {}

    def available_genders(self, model_type: str):
        return tuple(
            gender
            for gender in ("male", "female", "neutral")
            if resolve_body_model_path(self.support_dir, model_type, gender) is not None
        )

    def has(self, model_type) -> bool:
        resolved = normalize_body_model_type(model_type)
        return bool(resolved) and bool(self.available_genders(resolved))

    def get(self, model_type=None, gender=None):
        """Return a cached BodyModel, falling back across gender then family."""
        from human_body_prior.body_model.body_model import BodyModel

        resolved_type = normalize_body_model_type(model_type) or self.default_type
        gender = normalize_gender(gender)
        key = (resolved_type, gender)
        if key in self._models:
            return self._models[key]

        path = resolve_body_model_path(self.support_dir, resolved_type, gender)
        if path is None:
            for fallback_gender in ("neutral", "male", "female"):
                path = resolve_body_model_path(self.support_dir, resolved_type, fallback_gender)
                if path is not None:
                    break
        if path is None:
            raise FileNotFoundError(
                f"No {resolved_type} body model found under {self.support_dir}; "
                f"expected one of {_BODY_MODEL_PATH_TEMPLATES[resolved_type]}"
            )

        model = BodyModel(bm_fname=path, num_betas=clamp_num_betas(path, self.num_betas))
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        if self.device is not None:
            model = model.to(self.device)
        self._models[key] = model
        return model

    def get_canonical(self, model_type=None):
        """Return the requested family's neutral model without gender fallback."""
        resolved_type = normalize_body_model_type(model_type) or self.default_type
        neutral_path = resolve_body_model_path(
            self.support_dir, resolved_type, CANONICAL_BODY_MODEL_GENDER
        )
        if neutral_path is None:
            raise FileNotFoundError(
                f"Canonical neutral {resolved_type} model not found under {self.support_dir}"
            )
        return self.get(resolved_type, CANONICAL_BODY_MODEL_GENDER)

    def rest_joints(self, model_type, gender, betas=None) -> torch.Tensor:
        """Shaped rest-pose joints ``[N, 3]`` for one (family, gender, betas)."""
        import numpy as np

        model = self.get(model_type, gender)
        betas_np = None
        if betas is not None:
            betas_np = np.asarray(betas, dtype=np.float32).reshape(-1)[: model.num_betas]
            if betas_np.shape[0] < model.num_betas:
                betas_np = np.pad(betas_np, (0, model.num_betas - betas_np.shape[0]))
        key = (
            normalize_body_model_type(model_type) or self.default_type,
            normalize_gender(gender),
            None if betas_np is None else betas_np.tobytes(),
        )
        if key in self._rest_joints:
            return self._rest_joints[key]

        device = next(model.buffers()).device
        kwargs = {}
        if betas_np is not None:
            kwargs["betas"] = torch.from_numpy(betas_np).to(device).unsqueeze(0)
        with torch.no_grad():
            joints = model(**kwargs).Jtr[0].detach().cpu()
        self._rest_joints[key] = joints
        return joints

    def match_type_by_rest_offsets(
        self,
        rest_offsets,
        betas=None,
        gender=None,
        candidates: Sequence[str] = SUPPORTED_BODY_MODEL_TYPES,
        tolerance_m: float = 0.01,
    ) -> Optional[str]:
        """Pick the family whose shaped rest skeleton reproduces ``rest_offsets``.

        ``rest_offsets`` are parent-relative SMPL-24 bone vectors as shipped by
        OMOMO.  Only body joints 1..21 are compared, because 22/23 are the
        dataset's own palm convention rather than a native joint index.
        """
        import numpy as np

        offsets = np.asarray(rest_offsets, dtype=np.float64)
        if offsets.ndim != 2 or offsets.shape[0] < 22 or offsets.shape[1] != 3:
            return None

        best_type, best_error = None, float("inf")
        for candidate in candidates:
            if not self.has(candidate):
                continue
            joints = self.rest_joints(candidate, gender, betas).numpy().astype(np.float64)
            if joints.shape[0] < 22:
                continue
            error = max(
                float(np.linalg.norm(offsets[j] - (joints[j] - joints[SMPL_PARENTS[j]])))
                for j in range(1, 22)
            )
            if error < best_error:
                best_type, best_error = candidate, error
        if best_type is None or best_error > float(tolerance_m):
            return None
        return best_type

    def infer_type(self, seq_data, gender=None) -> str:
        """Resolve the family a sequence was fitted with.

        Order: an explicit declaration, then the width of the raw pose array,
        then a numeric match against ``rest_offsets``, then the default.
        """
        import numpy as np

        if not hasattr(seq_data, "get"):
            return self.default_type

        for field in _MODEL_TYPE_FIELDS:
            declared = normalize_body_model_type(_scalar_to_str(seq_data.get(field)))
            if declared is not None:
                return declared

        for field in _POSE_FIELDS:
            poses = seq_data.get(field)
            if poses is None:
                continue
            poses = np.asarray(poses)
            if poses.ndim == 0:
                continue
            inferred = infer_body_model_type_from_pose_width(poses.shape[-1])
            if inferred is not None:
                return inferred

        rest_offsets = seq_data.get("rest_offsets")
        if rest_offsets is not None:
            matched = self.match_type_by_rest_offsets(
                rest_offsets,
                betas=seq_data.get("betas"),
                gender=gender if gender is not None else seq_data.get("gender"),
            )
            if matched is not None:
                return matched

        return self.default_type

    def for_sequence(self, seq_data, gender=None):
        """Return ``(body_model, model_type)`` for one sequence."""
        if gender is None and hasattr(seq_data, "get"):
            gender = _scalar_to_str(seq_data.get("gender")) or seq_data.get("gender")
        model_type = self.infer_type(seq_data, gender=gender)
        return self.get(model_type, gender), model_type


class GenderedBodyModelLoader:
    """Per-dataset loader that resolves both gender and model family.

    Keeps the ``loader.get(gender)`` shape the dataset preprocessors already
    use, but the family is looked up from the sequence via :meth:`resolve`
    rather than hard-coded to SMPL-H.
    """

    def __init__(
        self,
        support_dir: str,
        device=None,
        num_betas: int = 16,
        default_type: str = "smplh",
    ) -> None:
        self.registry = BodyModelRegistry(
            support_dir, device=device, num_betas=num_betas, default_type=default_type
        )
        if not self.registry.has(self.registry.default_type):
            raise FileNotFoundError(
                f"No {self.registry.default_type} body model found under {support_dir}"
            )
        self._model_type = self.registry.default_type

    @property
    def num_betas(self) -> int:
        return self.registry.num_betas

    @property
    def model_type(self) -> str:
        return self._model_type

    def get(self, gender=None, model_type=None):
        """Return the BodyModel for ``gender`` in the currently active family."""
        return self.registry.get(model_type or self._model_type, gender)

    def resolve(self, seq_data=None, gender=None, model_type=None):
        """Resolve ``(body_model, model_type)`` for a sequence and remember it.

        ``model_type`` forces a family; otherwise it is inferred from the
        sequence (declared field, pose width, or rest_offsets match).
        """
        resolved = normalize_body_model_type(model_type)
        if resolved is None and seq_data is not None:
            resolved = self.registry.infer_type(seq_data, gender=gender)
        if resolved is None:
            resolved = self._model_type
        if not self.registry.has(resolved):
            raise FileNotFoundError(
                f"Sequence needs a {resolved} body model, but none was found under "
                f"{self.registry.support_dir}"
            )
        self._model_type = resolved
        return self.registry.get(resolved, gender), resolved


def ensure_body_model_device(module, device: torch.device):
    """Move module.body_model to device once and track the current device."""
    body_model = getattr(module, "body_model", None)
    if body_model is not None and getattr(module, "body_model_device", None) != device:
        module.body_model = body_model.to(device)
        module.body_model_device = device


def _body_model_batch_size(*args) -> int:
    for arg in args:
        if arg is not None:
            return int(arg.shape[0])
    return 1


def _expand_body_model_buffer(body_model, name: str, batch_size: int) -> torch.Tensor:
    value = getattr(body_model, name)
    return value.expand(batch_size, *value.shape[1:])


def _body_model_shape_components(
    body_model,
    *,
    batch_size: int,
    betas: Optional[torch.Tensor],
    dmpls: Optional[torch.Tensor],
    expression: Optional[torch.Tensor],
):
    model_type = getattr(body_model, "model_type", "")
    use_dmpl = bool(getattr(body_model, "use_dmpl", False))
    use_expression = bool(getattr(body_model, "use_expression", False)) or (
        model_type in {"smplx", "flame"} and hasattr(body_model, "exprdirs")
    )

    if use_dmpl:
        if betas is None:
            betas = _expand_body_model_buffer(body_model, "init_betas", batch_size)
        if dmpls is None:
            dmpls = _expand_body_model_buffer(body_model, "init_dmpls", batch_size)
        return torch.cat([betas, dmpls], dim=-1), torch.cat([body_model.shapedirs, body_model.dmpldirs], dim=-1)

    if use_expression:
        if betas is None:
            betas = _expand_body_model_buffer(body_model, "init_betas", batch_size)
        if expression is None:
            expression = _expand_body_model_buffer(body_model, "init_expression", batch_size)
        return torch.cat([betas, expression], dim=-1), torch.cat([body_model.shapedirs, body_model.exprdirs], dim=-1)

    if betas is None:
        return _NO_SHAPE_ARG, body_model.shapedirs
    return betas, body_model.shapedirs


def _body_model_full_pose(
    body_model,
    *,
    batch_size: int,
    root_orient: Optional[torch.Tensor],
    pose_body: Optional[torch.Tensor],
    pose_hand: Optional[torch.Tensor],
    pose_jaw: Optional[torch.Tensor],
    pose_eye: Optional[torch.Tensor],
) -> torch.Tensor:
    model_type = getattr(body_model, "model_type", "")
    if root_orient is None:
        root_orient = _expand_body_model_buffer(body_model, "init_root_orient", batch_size)

    if model_type in {"smpl", "smplh"}:
        if pose_body is None:
            pose_body = _expand_body_model_buffer(body_model, "init_pose_body", batch_size)
        if pose_hand is None:
            pose_hand = _expand_body_model_buffer(body_model, "init_pose_hand", batch_size)
        return torch.cat([root_orient, pose_body, pose_hand], dim=-1)

    if model_type == "smplx":
        if pose_body is None:
            pose_body = _expand_body_model_buffer(body_model, "init_pose_body", batch_size)
        if pose_hand is None:
            pose_hand = _expand_body_model_buffer(body_model, "init_pose_hand", batch_size)
        if pose_jaw is None:
            pose_jaw = _expand_body_model_buffer(body_model, "init_pose_jaw", batch_size)
        if pose_eye is None:
            pose_eye = _expand_body_model_buffer(body_model, "init_pose_eye", batch_size)
        return torch.cat([root_orient, pose_body, pose_jaw, pose_eye, pose_hand], dim=-1)

    if model_type == "flame":
        if pose_body is None:
            pose_body = _expand_body_model_buffer(body_model, "init_pose_body", batch_size)
        if pose_jaw is None:
            pose_jaw = _expand_body_model_buffer(body_model, "init_pose_jaw", batch_size)
        if pose_eye is None:
            pose_eye = _expand_body_model_buffer(body_model, "init_pose_eye", batch_size)
        return torch.cat([root_orient, pose_body, pose_jaw, pose_eye], dim=-1)

    if model_type == "mano":
        if pose_hand is None:
            pose_hand = _expand_body_model_buffer(body_model, "init_pose_hand", batch_size)
        return torch.cat([root_orient, pose_hand], dim=-1)

    if model_type in {"animal_horse", "animal_dog", "animal_rat"}:
        if pose_body is None:
            pose_body = _expand_body_model_buffer(body_model, "init_pose_body", batch_size)
        return torch.cat([root_orient, pose_body], dim=-1)

    raise ValueError(f"Unsupported body model type: {model_type}")


def _body_model_joints_from_shape(
    body_model,
    *,
    batch_size: int,
    shape_components,
    shapedirs: torch.Tensor,
    v_template: Optional[torch.Tensor],
    joints: Optional[torch.Tensor],
    v_shaped: Optional[torch.Tensor],
) -> torch.Tensor:
    from human_body_prior.body_model.lbs import vertices2joints

    if joints is not None:
        return joints
    if v_shaped is not None:
        return vertices2joints(body_model.J_regressor, v_shaped)

    if v_template is None:
        template_joints = vertices2joints(body_model.J_regressor, body_model.init_v_template)
        template_joints = template_joints.expand(batch_size, -1, -1)
    else:
        template_joints = vertices2joints(body_model.J_regressor, v_template)

    if shape_components is _NO_SHAPE_ARG:
        return template_joints

    joint_shapedirs = torch.einsum("ji,ikl->jkl", body_model.J_regressor, shapedirs)
    return template_joints + torch.einsum("bl,jkl->bjk", shape_components, joint_shapedirs)


def _full_body_model_joints_fallback(
    body_model,
    *,
    root_orient: Optional[torch.Tensor],
    pose_body: Optional[torch.Tensor],
    pose_hand: Optional[torch.Tensor],
    pose_jaw: Optional[torch.Tensor],
    pose_eye: Optional[torch.Tensor],
    betas: Optional[torch.Tensor],
    trans: Optional[torch.Tensor],
    dmpls: Optional[torch.Tensor],
    expression: Optional[torch.Tensor],
    v_template: Optional[torch.Tensor],
    joints: Optional[torch.Tensor],
    v_shaped: Optional[torch.Tensor],
) -> torch.Tensor:
    body_out = body_model(
        root_orient=root_orient,
        pose_body=pose_body,
        pose_hand=pose_hand,
        pose_jaw=pose_jaw,
        pose_eye=pose_eye,
        betas=betas,
        trans=trans,
        dmpls=dmpls,
        expression=expression,
        v_template=v_template,
        joints=joints,
        v_shaped=v_shaped,
    )
    if isinstance(body_out, dict):
        return body_out["Jtr"]
    return body_out.Jtr


def forward_body_model_joints(
    body_model,
    *,
    root_orient: Optional[torch.Tensor] = None,
    pose_body: Optional[torch.Tensor] = None,
    pose_hand: Optional[torch.Tensor] = None,
    pose_jaw: Optional[torch.Tensor] = None,
    pose_eye: Optional[torch.Tensor] = None,
    betas: Optional[torch.Tensor] = None,
    trans: Optional[torch.Tensor] = None,
    dmpls: Optional[torch.Tensor] = None,
    expression: Optional[torch.Tensor] = None,
    v_template: Optional[torch.Tensor] = None,
    joints: Optional[torch.Tensor] = None,
    v_shaped: Optional[torch.Tensor] = None,
    fallback_to_full: bool = True,
) -> torch.Tensor:
    """BodyModel forward variant that returns Jtr without computing mesh vertices."""
    try:
        from human_body_prior.body_model.lbs import batch_rigid_transform, batch_rodrigues

        batch_size = _body_model_batch_size(
            root_orient,
            pose_body,
            pose_hand,
            pose_jaw,
            pose_eye,
            betas,
            trans,
            dmpls,
            expression,
            v_template,
            joints,
        )
        full_pose = _body_model_full_pose(
            body_model,
            batch_size=batch_size,
            root_orient=root_orient,
            pose_body=pose_body,
            pose_hand=pose_hand,
            pose_jaw=pose_jaw,
            pose_eye=pose_eye,
        )
        shape_components, shapedirs = _body_model_shape_components(
            body_model,
            batch_size=batch_size,
            betas=betas,
            dmpls=dmpls,
            expression=expression,
        )
        joint_rest = _body_model_joints_from_shape(
            body_model,
            batch_size=batch_size,
            shape_components=shape_components,
            shapedirs=shapedirs,
            v_template=v_template,
            joints=joints,
            v_shaped=v_shaped,
        )

        dtype = getattr(body_model, "dtype", full_pose.dtype)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)
        jtr, _ = batch_rigid_transform(rot_mats, joint_rest, body_model.kintree_table[0].long(), dtype=dtype)
        if trans is None:
            trans = _expand_body_model_buffer(body_model, "init_trans", batch_size)
        return jtr + trans.unsqueeze(dim=1)
    except Exception:
        if not fallback_to_full:
            raise
        return _full_body_model_joints_fallback(
            body_model,
            root_orient=root_orient,
            pose_body=pose_body,
            pose_hand=pose_hand,
            pose_jaw=pose_jaw,
            pose_eye=pose_eye,
            betas=betas,
            trans=trans,
            dmpls=dmpls,
            expression=expression,
            v_template=v_template,
            joints=joints,
            v_shaped=v_shaped,
        )


def global_to_local_rotmats(
    global_rotmats: torch.Tensor,
    parents: Sequence[int] = SMPL_PARENTS,
    ignored_indices: Optional[Iterable[int]] = _IGNORED_INDICES,
) -> torch.Tensor:
    """Convert global joint rotations to local rotations."""
    if global_rotmats.shape[-2:] != (3, 3):
        raise ValueError(f"global_rotmats must end with [3,3], got {global_rotmats.shape}")

    batch_size, num_joints = global_rotmats.shape[:2]
    local = torch.zeros_like(global_rotmats)
    local[:, 0] = global_rotmats[:, 0]

    for joint_idx in range(1, num_joints):
        parent_idx = parents[joint_idx]
        local[:, joint_idx] = torch.matmul(global_rotmats[:, parent_idx].transpose(-1, -2), global_rotmats[:, joint_idx])

    if ignored_indices:
        ignored = [idx for idx in ignored_indices if idx < num_joints]
        if ignored:
            eye = torch.eye(3, device=global_rotmats.device, dtype=global_rotmats.dtype).view(1, 1, 3, 3)
            local[:, ignored] = eye.expand(batch_size, len(ignored), -1, -1)
    return local


def reduced_root_pose_to_full_global(
    reduced_root_6d: torch.Tensor,
    imu_orientation_mat: torch.Tensor,
    *,
    num_joints: int = 24,
    reduced_indices: Sequence[int] = _REDUCED_INDICES,
    sensor_rot_indices: Sequence[int] = _SENSOR_ROT_INDICES,
    ignored_indices: Sequence[int] = _IGNORED_INDICES,
    parents: Sequence[int] = SMPL_PARENTS,
) -> torch.Tensor:
    """
    Assemble full global rotations from root-relative reduced pose and IMU rotations.

    Args:
        reduced_root_6d: [N, R, 6], reduced joint rotations in root frame.
        imu_orientation_mat: [N, S, 3, 3], root IMU is global, other IMUs are root-relative.
    """
    if reduced_root_6d.dim() != 3 or reduced_root_6d.shape[-1] != 6:
        raise ValueError(f"reduced_root_6d must be [N,R,6], got {reduced_root_6d.shape}")
    if imu_orientation_mat.dim() != 4 or imu_orientation_mat.shape[-2:] != (3, 3):
        raise ValueError(f"imu_orientation_mat must be [N,S,3,3], got {imu_orientation_mat.shape}")

    n = reduced_root_6d.shape[0]
    device = reduced_root_6d.device
    dtype = reduced_root_6d.dtype
    root_rot = imu_orientation_mat[:, 0]
    reduced_local = rotation_6d_to_matrix(reduced_root_6d.reshape(-1, 6)).reshape(n, len(reduced_indices), 3, 3)
    reduced_global = torch.matmul(root_rot.unsqueeze(1), reduced_local)

    full = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(n, num_joints, 1, 1)
    valid_reduced = [idx for idx in reduced_indices if idx < num_joints]
    full[:, valid_reduced] = reduced_global[:, : len(valid_reduced)]

    sensor_count = min(imu_orientation_mat.shape[1], len(sensor_rot_indices))
    sensor_global = imu_orientation_mat[:, :sensor_count].clone()
    if sensor_count > 1:
        sensor_global[:, 1:] = torch.matmul(root_rot.unsqueeze(1), imu_orientation_mat[:, 1:sensor_count])
    valid_sensor_indices = [idx for idx in sensor_rot_indices[:sensor_count] if idx < num_joints]
    full[:, valid_sensor_indices] = sensor_global[:, : len(valid_sensor_indices)]

    ignored = sorted(idx for idx in ignored_indices if idx < num_joints)
    for idx in ignored:
        parent_idx = int(parents[idx])
        if 0 <= parent_idx < num_joints:
            full[:, idx] = full[:, parent_idx]
    return full


def compute_smpl_joints_from_global(
    full_global_rotmats: torch.Tensor,
    body_model,
    *,
    num_joints: int = 24,
    parents: Sequence[int] = SMPL_PARENTS,
) -> Optional[torch.Tensor]:
    """Run SMPL/SMPLH FK from full global rotations."""
    if body_model is None:
        return None
    if full_global_rotmats.dim() != 5 or full_global_rotmats.shape[-2:] != (3, 3):
        raise ValueError(f"full_global_rotmats must be [B,T,J,3,3], got {full_global_rotmats.shape}")

    batch_size, seq_len, joint_count = full_global_rotmats.shape[:3]
    bt = batch_size * seq_len
    full_global = full_global_rotmats.reshape(bt, joint_count, 3, 3)
    if joint_count < num_joints:
        eye = torch.eye(3, device=full_global.device, dtype=full_global.dtype).view(1, 1, 3, 3)
        pad = eye.expand(bt, num_joints - joint_count, -1, -1)
        full_global = torch.cat((full_global, pad), dim=1)
    elif joint_count > num_joints:
        full_global = full_global[:, :num_joints]

    local_pose = global_to_local_rotmats(full_global, parents=parents)
    pose_aa = matrix_to_axis_angle(local_pose.reshape(-1, 3, 3)).reshape(bt, num_joints, 3)

    try:
        joints = forward_body_model_joints(
            body_model,
            pose_body=pose_aa[:, 1:22].reshape(bt, 63),
            root_orient=pose_aa[:, 0].reshape(bt, 3),
        )
    except Exception:
        return None

    if joints.size(1) > num_joints:
        joints = joints[:, :num_joints]
    elif joints.size(1) < num_joints:
        pad = torch.zeros(
            joints.size(0),
            num_joints - joints.size(1),
            3,
            device=joints.device,
            dtype=joints.dtype,
        )
        joints = torch.cat((joints, pad), dim=1)
    return joints.reshape(batch_size, seq_len, num_joints, 3)


def compute_root_velocity_from_trans(trans: torch.Tensor, fps: float) -> torch.Tensor:
    """Compute root velocity by first difference."""
    if trans.dim() == 2:
        trans = trans.unsqueeze(1)
    vel = torch.zeros_like(trans)
    if trans.size(1) > 1:
        vel[:, 1:] = (trans[:, 1:] - trans[:, :-1]) * float(fps)
        vel[:, 0] = vel[:, 1]
    return vel


def integrate_root_velocity(root_velocity: torch.Tensor, fps: float, trans_init: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Integrate velocity to root translation, keeping frame 0 at trans_init."""
    trans = torch.zeros_like(root_velocity)
    if root_velocity.size(1) > 1:
        trans[:, 1:] = torch.cumsum(root_velocity[:, 1:], dim=1) / float(fps)
    if trans_init is not None:
        if trans_init.dim() == 1:
            trans_init = trans_init.unsqueeze(0)
        trans = trans + trans_init.unsqueeze(1).to(device=trans.device, dtype=trans.dtype)
    return trans


def full_global_to_reduced_root_6d(
    full_global_rotmats: torch.Tensor,
    *,
    reduced_indices: Sequence[int] = _REDUCED_INDICES,
) -> torch.Tensor:
    """Convert full global rotations to root-relative reduced 6D rotations."""
    root_rot = full_global_rotmats[:, :, 0]
    reduced_global = full_global_rotmats[:, :, reduced_indices]
    reduced_root = torch.matmul(root_rot.unsqueeze(2).transpose(-1, -2), reduced_global)
    return matrix_to_rotation_6d(reduced_root.reshape(-1, 3, 3)).reshape(
        full_global_rotmats.shape[0],
        full_global_rotmats.shape[1],
        len(reduced_indices),
        6,
    )
