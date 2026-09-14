"""Regression tests for body-model family detection and the palm anchor.

Two bugs motivated these:

1. The palm anchor used to be the *index-finger* MCP (SMPL-H 22 / SMPL-X 25),
   named "metacarpal".  That joint sits 2.2 cm off the wrist axis towards the
   thumb and 4.2 cm from SMPL's own hand joint.  The anchor is now middle1.
2. OMOMO and PAHOI are SMPL-X captures that were being run through a SMPL-H
   body model.  The two families' root joints differ by 12.7 cm, which
   displaced the whole body relative to the object trajectory.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.human_pose import (  # noqa: E402
    BODY_MODEL_TYPE_BY_NATIVE_COUNT,
    PALM_JOINT_INDICES_BY_NATIVE_COUNT,
    SMPLH_PALM_JOINT_INDICES,
    SMPLX_PALM_JOINT_INDICES,
    SMPL_PALM_JOINT_INDICES,
    SMPL_PARENTS,
    BodyModelRegistry,
    build_smpl24_joint_positions,
    clamp_num_betas,
    infer_body_model_type_from_pose_width,
    normalize_body_model_type,
    normalize_gender,
    resolve_body_model_path,
)
from utils.dataset_config import resolve_evaluation_body_model_path  # noqa: E402

SUPPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "smpl_models"
)
# MANO orders each hand as index/middle/pinky/ring/thumb, 3 joints each.
MANO_MIDDLE1_OFFSET = 3


def test_palm_indices_are_the_middle_finger_mcp():
    # SMPL-H: 0..21 body, 22..36 left hand, 37..51 right hand.
    assert SMPLH_PALM_JOINT_INDICES == (22 + MANO_MIDDLE1_OFFSET, 37 + MANO_MIDDLE1_OFFSET)
    # SMPL-X inserts jaw/eyes at 22..24, so the hand chains start 3 later.
    assert SMPLX_PALM_JOINT_INDICES == (25 + MANO_MIDDLE1_OFFSET, 40 + MANO_MIDDLE1_OFFSET)
    # Plain SMPL already exposes the hands as joints 22/23.
    assert SMPL_PALM_JOINT_INDICES == (22, 23)


def test_smplx_palm_indices_are_offset_by_the_face_joints():
    """SMPL-X inserts jaw + two eyes at 22..24, shifting both hand chains by 3."""
    assert SMPLX_PALM_JOINT_INDICES != SMPLH_PALM_JOINT_INDICES
    assert SMPLX_PALM_JOINT_INDICES == tuple(i + 3 for i in SMPLH_PALM_JOINT_INDICES)
    # Cross-applying a table therefore lands on a different finger segment:
    # SMPL-X's (28, 43) read against SMPL-H output gives middle2 and ring1.
    assert all(index < 52 for index in SMPLX_PALM_JOINT_INDICES)


def test_build_smpl24_selects_the_configured_palm_rows():
    for native_count, palm_indices in PALM_JOINT_INDICES_BY_NATIVE_COUNT.items():
        joints = torch.arange(native_count * 3, dtype=torch.float32).reshape(1, native_count, 3)
        out = build_smpl24_joint_positions(joints)
        assert out.shape == (1, 24, 3)
        assert torch.equal(out[0, :22], joints[0, :22])
        assert torch.equal(out[0, 22], joints[0, palm_indices[0]])
        assert torch.equal(out[0, 23], joints[0, palm_indices[1]])


def test_build_smpl24_passes_through_a_24_joint_input():
    joints = torch.randn(2, 3, 24, 3)
    assert torch.equal(build_smpl24_joint_positions(joints), joints)


def test_build_smpl24_rejects_an_unknown_layout():
    for bad in (22, 45, 73):
        try:
            build_smpl24_joint_positions(torch.zeros(1, bad, 3))
        except ValueError:
            continue
        raise AssertionError(f"{bad}-joint input should not be accepted")


def test_pose_width_identifies_the_family():
    assert infer_body_model_type_from_pose_width(72) == "smpl"
    assert infer_body_model_type_from_pose_width(156) == "smplh"
    assert infer_body_model_type_from_pose_width(165) == "smplx"
    # 66 = root + 21 body joints only; shared by every family, so undecidable.
    assert infer_body_model_type_from_pose_width(66) is None


def test_native_joint_count_maps_to_family():
    assert BODY_MODEL_TYPE_BY_NATIVE_COUNT == {24: "smpl", 52: "smplh", 55: "smplx"}


def test_model_type_and_gender_normalisation():
    for alias in ("SMPL-X", "smplx", "smpl_x", " SMPLX "):
        assert normalize_body_model_type(alias) == "smplx"
    for alias in ("SMPL+H", "smplh"):
        assert normalize_body_model_type(alias) == "smplh"
    assert normalize_body_model_type("nonsense") is None
    assert normalize_gender(np.str_("MALE")) == "male"
    assert normalize_gender(None) == "neutral"
    assert normalize_gender("unknown") == "neutral"


def _registry():
    return BodyModelRegistry(SUPPORT_DIR, device=torch.device("cpu"), num_betas=16)


def test_infer_type_prefers_a_declared_field_then_pose_width():
    registry = _registry()
    # numpy 0-d string arrays are how npz captures store the field
    assert registry.infer_type({"surface_model_type": np.array("smplx")}) == "smplx"
    assert registry.infer_type({"model": "smplh", "poses": np.zeros((4, 165))}) == "smplh"
    assert registry.infer_type({"poses": np.zeros((4, 156))}) == "smplh"
    assert registry.infer_type({"poses": np.zeros((4, 1, 165))}) == "smplx"
    # nothing to go on falls back to the registry default
    assert registry.infer_type({"gender": "male"}) == registry.default_type


def test_omomo_rest_offsets_identify_smplx():
    """OMOMO ships no model-type field; only the skeleton itself gives it away."""
    if resolve_body_model_path(SUPPORT_DIR, "smplx", "male") is None:
        return  # models not available in this checkout
    registry = _registry()
    joints = registry.rest_joints("smplx", "male", betas=None).numpy().astype(np.float64)
    rest_offsets = np.zeros((24, 3), dtype=np.float64)
    for joint in range(1, 22):
        rest_offsets[joint] = joints[joint] - joints[SMPL_PARENTS[joint]]

    assert registry.match_type_by_rest_offsets(rest_offsets, gender="male") == "smplx"
    # The same offsets must not be mistaken for SMPL-H: the families' body bones
    # differ by up to 6 cm, far above the 1 cm tolerance.
    smplh = registry.rest_joints("smplh", "male", betas=None).numpy().astype(np.float64)
    worst = max(
        float(np.linalg.norm(rest_offsets[j] - (smplh[j] - smplh[SMPL_PARENTS[j]])))
        for j in range(1, 22)
    )
    assert worst > 0.01, worst


def test_registry_resolves_both_release_layouts():
    if resolve_body_model_path(SUPPORT_DIR, "smplx", "male") is None:
        return
    # AMASS ships smplh/<gender>/model.npz, the SMPL-X release ships SMPLX_<GENDER>.npz
    assert resolve_body_model_path(SUPPORT_DIR, "smplh", "male").endswith("model.npz")
    assert "SMPLX_MALE" in resolve_body_model_path(SUPPORT_DIR, "smplx", "male")
    registry = _registry()
    for model_type, expected in (("smplh", 52), ("smplx", 55)):
        joints = registry.rest_joints(model_type, "male", betas=None)
        assert joints.shape[0] == expected, (model_type, joints.shape)


def test_palm_bone_length_is_consistent_across_families():
    """Both families must place the anchor at the same point on the hand."""
    if resolve_body_model_path(SUPPORT_DIR, "smplx", "male") is None:
        return
    registry = _registry()
    lengths = {}
    for model_type in ("smplh", "smplx"):
        joints = build_smpl24_joint_positions(
            registry.rest_joints(model_type, "male", betas=None).unsqueeze(0)
        )[0]
        lengths[model_type] = float(torch.linalg.vector_norm(joints[22] - joints[20]))
    assert abs(lengths["smplh"] - lengths["smplx"]) < 0.01, lengths


def test_clamp_num_betas_respects_the_file():
    if resolve_body_model_path(SUPPORT_DIR, "smplh", "male") is None:
        return
    smplh = resolve_body_model_path(SUPPORT_DIR, "smplh", "male")
    assert clamp_num_betas(smplh, 16) == 16
    assert clamp_num_betas(smplh, 64) == 16
    # a path that cannot be read must not raise, so callers keep working
    assert clamp_num_betas("does_not_exist.npz", 16) == 16


def test_evaluation_body_model_resolves_from_dataset_or_samples():
    config = {
        "body_model_path": "fallback.npz",
        "body_model_paths": {
            "smplh": "smplh.npz",
            "smplx": "smplx.npz",
        },
    }
    assert resolve_evaluation_body_model_path(
        config,
        {},
        [{"body_model_type": "smplx"}],
    ) == ("smplx.npz", "smplx", "sample.body_model_type")
    assert resolve_evaluation_body_model_path(
        config,
        {"body_model_type": "smplh"},
        [{"body_model_type": "smplh"}],
    ) == ("smplh.npz", "smplh", "dataset.body_model_type")
    assert resolve_evaluation_body_model_path(
        config,
        {},
        [{"body_model_type": "smplx"}],
        explicit_path="override.npz",
    ) == ("override.npz", None, "cli")

    with pytest.raises(ValueError, match="cannot mix body-model families"):
        resolve_evaluation_body_model_path(
            config,
            {},
            [
                {"body_model_type": "smplx"},
                {"body_model_type": "smplh"},
            ],
        )
