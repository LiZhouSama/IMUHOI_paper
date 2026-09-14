import inspect
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from process import preprocess_INTERCAP as intercap
from process import preprocess_PAHOI as pahoi


class _FakeRegistry:
    def __init__(self):
        self.model = object()
        self.canonical_requests = []
        self.infer_requests = []

    def get_canonical(self, model_type):
        self.canonical_requests.append(model_type)
        return self.model

    def get(self, *args, **kwargs):
        raise AssertionError("dataset preprocessors must not use gendered registry.get")

    def infer_type(self, seq_data, gender=None):
        self.infer_requests.append((seq_data, gender))
        return seq_data.get("surface_model_type", "smplx")


@pytest.mark.parametrize("module", [pahoi, intercap])
def test_body_model_loader_forces_neutral_smplx(module):
    loader = module.BodyModelLoader.__new__(module.BodyModelLoader)
    loader.registry = _FakeRegistry()
    loader._model_type = "smplx"

    assert loader.get(gender="female") is loader.registry.model
    model, family = loader.resolve(
        {"surface_model_type": "smplx", "gender": "male"},
        gender="male",
    )

    assert model is loader.registry.model
    assert family == "smplx"
    assert loader.registry.canonical_requests == ["smplx", "smplx"]
    assert loader.registry.infer_requests[0][1] == "neutral"
    with pytest.raises(ValueError, match="requires SMPL-X"):
        loader.get(model_type="smplh")


class _FakeBodyModel:
    def __init__(self, native_joint_count):
        self.num_betas = 4
        self.native_joint_count = native_joint_count
        parents = torch.tensor(
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
            dtype=torch.long,
        )
        self.kintree_table = torch.stack((parents, torch.arange(22)))
        self.calls = []

    def __call__(self, *, root_orient, pose_body, trans, betas):
        self.calls.append({
            "root_orient": root_orient.detach().clone(),
            "pose_body": pose_body.detach().clone(),
            "trans": trans.detach().clone(),
            "betas": betas.detach().clone(),
        })
        batch = root_orient.shape[0]
        joints = torch.zeros(
            batch,
            self.native_joint_count,
            3,
            dtype=root_orient.dtype,
            device=root_orient.device,
        )
        joints = joints + trans[:, None, :] + betas.sum(dim=-1)[:, None, None]
        return SimpleNamespace(Jtr=joints)


@pytest.mark.parametrize(
    ("module", "native_joint_count", "expected_joint_count"),
    [(pahoi, 55, 24), (intercap, 22, 22)],
)
def test_motion_features_pass_only_explicit_zero_betas(
    module, native_joint_count, expected_joint_count
):
    body_model = _FakeBodyModel(native_joint_count)
    frame_count = 5
    root_orient = np.zeros((frame_count, 3), dtype=np.float32)
    pose_body = np.zeros((frame_count, 63), dtype=np.float32)
    trans = np.arange(frame_count * 3, dtype=np.float32).reshape(frame_count, 3)

    positions, _, rotations = module.compute_motion_features(
        bm=body_model,
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans,
        device=torch.device("cpu"),
        chunk_len=2,
    )

    assert positions.shape == (frame_count, expected_joint_count, 3)
    assert rotations.shape == (frame_count, expected_joint_count, 3, 3)
    assert len(body_model.calls) == 3
    assert all(torch.count_nonzero(call["betas"]).item() == 0 for call in body_model.calls)


def test_pahoi_source_gender_and_betas_do_not_enter_sequence_payload(tmp_path):
    frame_count = 3
    concat_path = tmp_path / "concat.npz"
    raw_path = tmp_path / "raw.npz"
    np.savez(
        concat_path,
        poses=np.zeros((frame_count, 165), dtype=np.float32),
        trans=np.zeros((frame_count, 3), dtype=np.float32),
        gender=np.asarray("female"),
        mocap_frame_rate=np.asarray(30),
        surface_model_type=np.asarray("smplx"),
        betas=np.full((1, 16), 7.0, dtype=np.float32),
    )
    np.savez(raw_path, betas=np.full((1, 16), 11.0, dtype=np.float32))
    sequence = pahoi.PAHOISequence(
        seq_name="1_1",
        subject_id="1",
        concat_npz_path=concat_path,
        raw_npz_path=raw_path,
        object_fbx_path=tmp_path / "object.fbx",
        worldpos_npy_path=None,
        worldpos_csv_path=None,
    )

    loaded = pahoi.load_pahoi_human_sequence(sequence, num_betas=16)

    assert loaded["gender"] == "neutral"
    assert loaded["surface_model_type"] == "smplx"
    assert "betas" not in loaded


def test_intercap_source_gender_and_betas_do_not_enter_sequence_payload(tmp_path):
    frame_count = 3
    segment_dir = tmp_path / "P01" / "01" / "Seg_0"
    mesh_dir = segment_dir / "Mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "00000_second_obj.ply").write_text("ply\n", encoding="utf8")
    payload = {
        "global_orient": np.zeros((frame_count, 3), dtype=np.float32),
        "transl": np.zeros((frame_count, 3), dtype=np.float32),
        "body_pose": np.zeros((frame_count, 63), dtype=np.float32),
        "ob_pose": np.zeros((frame_count, 3), dtype=np.float32),
        "ob_trans": np.zeros((frame_count, 3), dtype=np.float32),
        "betas": np.full((1, 16), 13.0, dtype=np.float32),
    }
    with (segment_dir / "res_2.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    loaded = intercap.load_intercap_sequence(
        segment_dir=segment_dir,
        num_betas=16,
        source_fps=30,
        target_fps=30,
        gender="female",
        max_frames=None,
    )

    assert loaded["gender"] == "neutral"
    assert loaded["surface_model_type"] == "smplx"
    assert "betas" not in loaded


@pytest.mark.parametrize(
    ("module", "process_name"),
    [
        (pahoi, "process_pahoi_sequence"),
        (intercap, "process_intercap_sequence"),
    ],
)
def test_saved_schema_uses_canonical_metadata_without_betas(module, process_name):
    source = inspect.getsource(getattr(module, process_name))
    assert "canonical_skeleton_metadata(body_model_type)" in source
    assert '"betas":' not in source


def test_intercap_derived_features_and_saved_joints_use_canonical_smpl24():
    source = inspect.getsource(intercap.process_intercap_sequence)
    assert "append_virtual_palm_joints(position_global)" in source
    assert "append_virtual_palm_rotations(rotation_global)" in source
    assert '"position_global_full_gt_world": canonical_feature_positions.float()' in source
    assert '"rotation_global": canonical_feature_rotations.float()' in source
    assert "precompute_sequence_features(" in source
    assert "canonical_feature_positions.to(device)" in source
