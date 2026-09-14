"""Lightweight regression tests for canonical OMOMO preprocessing."""
from types import SimpleNamespace

import numpy as np
import torch

from process import preprocess
from utils.human_pose import BodyModelRegistry, build_smpl24_joint_positions


class _RecordingBodyModel:
    model_type = "smplx"
    num_betas = 16

    def __init__(self):
        self.calls = []
        base = torch.arange(55 * 3, dtype=torch.float32).reshape(55, 3) / 1000.0
        base[0] = torch.tensor([0.1, 0.2, 0.3])
        self.base_joints = base

    def __call__(self, **kwargs):
        self.calls.append({key: value.detach().clone() for key, value in kwargs.items()})
        trans = kwargs["trans"]
        joints = self.base_joints.to(device=trans.device, dtype=trans.dtype)
        joints = joints.unsqueeze(0).expand(trans.shape[0], -1, -1).clone()
        joints = joints + trans.unsqueeze(1)
        return SimpleNamespace(Jtr=joints)


class _RecordingRegistry(BodyModelRegistry):
    def __init__(self, body_model):
        self.body_model = body_model
        self.default_type = "smplh"
        self.default_gender = "neutral"
        self.num_betas = 16
        self.support_dir = "mock-support-dir"
        self.match_calls = []
        self.get_calls = []

    def available_genders(self, model_type):
        return ("neutral",) if model_type in ("smplx", "smplh") else ()

    def match_type_by_rest_offsets(self, rest_offsets, **kwargs):
        self.match_calls.append((np.asarray(rest_offsets).copy(), kwargs))
        return "smplx"

    def get_canonical(self, model_type=None):
        self.get_calls.append(model_type)
        return self.body_model


class _ShapePoisonedSequence(dict):
    def get(self, key, default=None):
        if key in {"gender", "betas"}:
            raise AssertionError(f"source {key} must not be read")
        return super().get(key, default)


def _source_sequence(length=10, with_object=False):
    sequence = {
        "seq_name": "sub_object_trial",
        "root_orient": np.zeros((length, 3), dtype=np.float32),
        "pose_body": np.zeros((length, 63), dtype=np.float32),
        "trans": np.stack(
            [
                np.linspace(0.0, 0.09, length),
                np.linspace(0.1, 0.19, length),
                np.linspace(-0.1, -0.01, length),
            ],
            axis=-1,
        ).astype(np.float32),
        "rest_offsets": np.zeros((24, 3), dtype=np.float32),
        "gender": "female",
        "betas": np.full((1, 16), 7.0, dtype=np.float32),
        "human_imu_real": torch.full((length, 6, 9), 123.0),
    }
    if with_object:
        sequence.update(
            {
                "obj_scale": np.ones(3, dtype=np.float32),
                "obj_rot": np.repeat(np.eye(3, dtype=np.float32)[None], length, axis=0),
                "obj_com_pos": np.stack(
                    [
                        np.linspace(0.2, 0.29, length),
                        np.linspace(0.3, 0.39, length),
                        np.linspace(0.4, 0.49, length),
                    ],
                    axis=-1,
                ).astype(np.float32),
            }
        )
    return sequence


def test_family_resolution_uses_neutral_zero_shape_only():
    body_model = _RecordingBodyModel()
    registry = _RecordingRegistry(body_model)
    sequence = _ShapePoisonedSequence(rest_offsets=np.zeros((24, 3), dtype=np.float32))

    resolved_model, model_type = preprocess._resolve_omomo_canonical_body_model(
        registry, sequence
    )

    assert resolved_model is body_model
    assert model_type == "smplx"
    assert registry.get_calls == ["smplx"]
    assert len(registry.match_calls) == 1
    _, match_kwargs = registry.match_calls[0]
    assert match_kwargs["gender"] == "neutral"
    assert np.array_equal(match_kwargs["betas"], np.zeros(16, dtype=np.float32))
    assert set(match_kwargs["candidates"]) == {"smplx", "smplh"}


def test_canonical_body_model_forward_passes_explicit_zero_betas():
    body_model = _RecordingBodyModel()
    length = 4
    result = preprocess._forward_canonical_body_model(
        body_model,
        torch.zeros(length, 3),
        torch.zeros(length, 63),
        torch.zeros(length, 3),
    )

    assert result.Jtr.shape == (length, 55, 3)
    assert len(body_model.calls) == 1
    assert body_model.calls[0]["betas"].shape == (length, 16)
    assert torch.count_nonzero(body_model.calls[0]["betas"]).item() == 0


def test_process_sequence_builds_all_human_geometry_from_canonical_joints(
    tmp_path, monkeypatch
):
    body_model = _RecordingBodyModel()
    registry = _RecordingRegistry(body_model)
    source = _source_sequence(with_object=True)
    captured = {}

    def _fake_contact_labels(
        obj_trans,
        obj_rot,
        obj_scale,
        position_global,
        obj_mesh_dir,
        object_name,
        device,
        sequence_length,
    ):
        captured["contact_joints"] = position_global.detach().cpu().clone()
        ones = torch.ones(sequence_length, device=device, dtype=torch.bool)
        return ones, ones.clone(), ones.clone()

    monkeypatch.setattr(preprocess, "compute_improved_contact_labels", _fake_contact_labels)

    saved = preprocess.process_sequence(
        source,
        seq_key=7,
        save_dir=str(tmp_path),
        bm=registry,
        device="cpu",
        obj_mesh_dir=None,
        contact_max_hole_frames=0,
        contact_min_duration_frames=1,
    )
    assert saved == 1

    output = torch.load(tmp_path / "7.pt", map_location="cpu", weights_only=False)
    assert output["gender"] == "neutral"
    assert output["body_model_type"] == "smplx"
    assert output["canonical_body_model_type"] == "smplx"
    assert output["canonical_body_model_gender"] == "neutral"
    assert output["canonical_betas"] == "zero"
    assert output["canonical_num_betas"] == 16
    assert output["canonical_joint_space"] == "neutral_zero_betas"
    assert "betas" not in output

    assert len(body_model.calls) == 2  # Jtr_0 correction and posed joints
    for call in body_model.calls:
        assert torch.count_nonzero(call["betas"]).item() == 0

    rot_x90 = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    source_trans = np.asarray(source["trans"], dtype=np.float32)
    neutral_root = body_model.base_joints[0].numpy()
    expected_trans = (source_trans + neutral_root) @ rot_x90.T - neutral_root
    assert np.allclose(output["trans"].numpy(), expected_trans, atol=1e-6)

    expected_native = body_model.base_joints.unsqueeze(0) + output["trans"].unsqueeze(1)
    expected_joints = build_smpl24_joint_positions(expected_native)
    saved_joints = output["position_global_full_gt_world"]
    assert torch.allclose(saved_joints, expected_joints)
    assert torch.equal(captured["contact_joints"], saved_joints)

    # Source human_imu_real is intentionally ignored; canonical positions drive
    # the saved IMU, velocities, contacts, hand directions and bone lengths.
    assert output["human_imu"].shape == source["human_imu_real"].shape
    assert not torch.equal(output["human_imu"], source["human_imu_real"])
    expected_human_imu = preprocess._build_human_imu_root(
        saved_joints,
        output["rotation_global"],
        preprocess._SENSOR_POS_INDICES,
        preprocess._SENSOR_ROT_INDICES,
        preprocess.FRAME_RATE,
    )
    assert torch.allclose(output["human_imu"], expected_human_imu)
    assert output["lhand_contact"].all()
    assert output["rhand_contact"].all()
    hand_positions = saved_joints[:, [22, 23], :]
    expected_left_lb = torch.linalg.vector_norm(
        output["obj_trans"] - hand_positions[:, 0], dim=-1
    )
    expected_right_lb = torch.linalg.vector_norm(
        output["obj_trans"] - hand_positions[:, 1], dim=-1
    )
    assert torch.allclose(output["lhand_lb"], expected_left_lb)
    assert torch.allclose(output["rhand_lb"], expected_right_lb)
    object_to_hand = output["obj_trans"].unsqueeze(1) - hand_positions
    expected_direction = object_to_hand / torch.linalg.vector_norm(
        object_to_hand, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    expected_direction = torch.matmul(
        output["obj_rot"].transpose(-1, -2).unsqueeze(1),
        expected_direction.unsqueeze(-1),
    ).squeeze(-1)
    assert torch.allclose(output["lhand_obj_direction"], expected_direction[:, 0])
    assert torch.allclose(output["rhand_obj_direction"], expected_direction[:, 1])
