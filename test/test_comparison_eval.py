import math
import os
import sys
import inspect
from argparse import Namespace

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Comparisons.common.adapters import (
    TIP_ORDER_FROM_IMUHOI,
    TIP_POSE_18,
    TRANSPOSE_VEL_SCALE,
    _globalpose_input,
    _tip_input,
    _transpose_input,
    adapt_batch,
)
from Comparisons.common.geometry import from_to_rotation_matrix
from Comparisons.common.dataset import IMUDataset as ComparisonIMUDataset
from Comparisons.common.globalpose_smpl import GlobalPoseSMPLModel
from Comparisons.common.geometry import (
    r6d_columns_to_rotation_matrix,
    rotation_matrix_to_r6d_columns,
    rotation_matrix_to_tip_2axis,
    tip_2axis_to_rotation_matrix,
)
from Comparisons.common.losses import TransPoseLoss, gaussian_nll
from Comparisons.common.metrics import (
    METRIC_UNITS,
    MetricAccumulator,
    add_compatibility_aliases,
    comparison_metric_stats,
    so3_geodesic_degrees,
)
from Comparisons.dip18.model import DIP18HOIModel, DIP18ReferenceHOIModel
from Comparisons.common.inference import object_velocity_to_world
from Comparisons.eval_comparison import (
    _build_model,
    _checkpoint_protocol,
    _checkpoint_state,
)
from Comparisons.transpose import TransPoseHOIModel
from Comparisons.tip import TIPHOIModel
from Comparisons.tip.sbp_cache import _angular_velocity_from_pair
from Comparisons.train_comparison import (
    METHOD_TRAINING_DEFAULTS,
    TIP_N_HEADS,
    TIP_STATE_NOISE,
    initialize_from_reference,
    split_training_files,
)
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d
from Comparisons.common.imu_noise import apply_imu_noise
from dataset.dataset_IMUHOI import IMUDataset as MainIMUDataset
from process.imu_noise import apply_imu_noise as apply_main_imu_noise
import utils.human_pose as main_human_pose


def test_comparison_protocol_extensions_are_isolated_from_main_modules():
    main_dataset_parameters = inspect.signature(MainIMUDataset.__init__).parameters
    comparison_dataset_parameters = inspect.signature(
        ComparisonIMUDataset.__init__
    ).parameters
    assert "imu_protocol" not in main_dataset_parameters
    assert "imu_protocol" in comparison_dataset_parameters

    main_noise_parameters = inspect.signature(apply_main_imu_noise).parameters
    comparison_noise_parameters = inspect.signature(apply_imu_noise).parameters
    assert "rotation_layout" not in main_noise_parameters
    assert "rotation_layout" in comparison_noise_parameters

    assert not hasattr(main_human_pose, "GlobalPoseSMPLModel")
    assert GlobalPoseSMPLModel.__module__ == "Comparisons.common.globalpose_smpl"


def _make_batch(batch_size=1, seq_len=5):
    eye = torch.eye(3)
    local = eye.view(1, 1, 1, 3, 3).repeat(batch_size, seq_len, 24, 1, 1)
    global_rot = local.clone()
    position = torch.zeros(batch_size, seq_len, 24, 3)
    offsets = torch.arange(24, dtype=torch.float32).view(1, 1, 24, 1) * 0.01
    position[..., 0:1] = offsets
    trans = torch.zeros(batch_size, seq_len, 3)

    imu_rot = eye.view(1, 1, 1, 3, 3).repeat(batch_size, seq_len, 6, 1, 1)
    human_imu = torch.cat(
        [
            torch.zeros(batch_size, seq_len, 6, 3),
            matrix_to_rotation_6d(imu_rot.reshape(-1, 3, 3)).reshape(
                batch_size, seq_len, 6, 6
            ),
        ],
        dim=-1,
    )
    obj_rot6d = matrix_to_rotation_6d(
        eye.view(1, 3, 3).repeat(batch_size * seq_len, 1, 1)
    ).reshape(batch_size, seq_len, 6)
    return {
        "human_imu": human_imu,
        "obj_imu": torch.cat(
            [torch.zeros(batch_size, seq_len, 3), obj_rot6d], dim=-1
        ),
        "obj_trans": torch.zeros(batch_size, seq_len, 3),
        "has_object": torch.ones(batch_size, dtype=torch.bool),
        "pose": torch.zeros(batch_size, seq_len, 66),
        "position_global": position,
        "rotation_global": global_rot,
        "trans": trans,
        "root_vel": torch.zeros(batch_size, seq_len, 3),
        "lfoot_contact": torch.zeros(batch_size, seq_len),
        "rfoot_contact": torch.zeros(batch_size, seq_len),
        "lhand_contact": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "rhand_contact": torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }


def test_transpose_targets_are_root_frame_and_velocity_scaled():
    batch = _make_batch(seq_len=4)
    angle = torch.tensor(math.pi / 2)
    root = axis_angle_to_matrix(torch.tensor([0.0, angle, 0.0])).view(1, 1, 3, 3)
    batch["rotation_global"][:, :, 0] = root
    batch["position_global"][:, :, 1] = torch.tensor([1.0, 0.0, 0.0])
    batch["trans"][0, :, 0] = torch.arange(4) / 30.0

    adapted = adapt_batch(batch, "transpose", protocol_version=2)
    expected_position = torch.einsum(
        "i,ij->j", torch.tensor([1.0, 0.0, 0.0]), root[0, 0]
    )
    assert torch.allclose(adapted["full_target"][0, 0, :3], expected_position)
    assert torch.allclose(adapted["root_vel_target"][0, 0], torch.zeros(3))
    assert torch.allclose(
        adapted["root_vel_target"][0, 1], expected_position / TRANSPOSE_VEL_SCALE
    )


def test_corrected_imu_adapters_do_not_read_gt_positions():
    batch = _make_batch(seq_len=6)
    transpose_before = _transpose_input(batch, fps=30.0)
    globalpose_before = _globalpose_input(batch, fps=30.0)
    batch["position_global"] = torch.randn_like(batch["position_global"]) * 100.0
    batch["rotation_global"] = axis_angle_to_matrix(
        torch.randn(batch["rotation_global"].shape[:-2] + (3,))
    )
    assert torch.allclose(_transpose_input(batch, fps=30.0), transpose_before)
    assert torch.allclose(_globalpose_input(batch, fps=30.0), globalpose_before)


def test_deployable_initial_states_do_not_use_sequence_ground_truth():
    batch = _make_batch(seq_len=4)
    tip_before = adapt_batch(batch, "tip", protocol_version=2)["initial_state"]
    globalpose_before = adapt_batch(batch, "globalpose", protocol_version=2)

    batch["pose"] = torch.randn_like(batch["pose"])
    batch["position_global"] = torch.randn_like(batch["position_global"]) * 100.0
    batch["rotation_global"] = axis_angle_to_matrix(
        torch.randn(batch["rotation_global"].shape[:-2] + (3,))
    )
    batch["trans"] = torch.randn_like(batch["trans"]) * 100.0
    batch["lfoot_contact"][:] = 1.0
    batch["rfoot_contact"][:] = 1.0

    tip_after = adapt_batch(batch, "tip", protocol_version=2)["initial_state"]
    globalpose_after = adapt_batch(batch, "globalpose", protocol_version=2)
    assert torch.allclose(tip_after, tip_before)
    assert torch.allclose(globalpose_after["pl_init"], globalpose_before["pl_init"])
    assert torch.allclose(globalpose_after["vr_init"], torch.zeros(1, 9))


def test_globalpose_reference_training_initializes_from_first_target_frame():
    batch = _make_batch(seq_len=4)
    adapted = adapt_batch(
        batch,
        "globalpose",
        protocol_version=2,
        globalpose_init_mode="reference",
    )
    assert adapted["globalpose_init_mode"] == "reference"
    assert torch.allclose(
        adapted["pl_init"], adapted["target"][:, 0, :18], atol=1e-6
    )
    assert torch.allclose(
        adapted["vr_init"], torch.zeros_like(adapted["target"][:, 0, -9:]), atol=1e-6
    )
    train_adapted = adapt_batch(
        batch,
        "globalpose",
        protocol_version=2,
        globalpose_init_mode="reference_train",
    )
    assert torch.allclose(
        train_adapted["vr_init"], train_adapted["target"][:, 0, -9:], atol=1e-6
    )


def test_object_supervision_is_first_frame_relative():
    batch = _make_batch(seq_len=4)
    batch["obj_trans"][0, :, 0] = torch.tensor([4.0, 5.0, 7.0, 8.0])
    adapted = adapt_batch(batch, "dip18", protocol_version=2)
    assert torch.allclose(adapted["obj_trans_init"], torch.tensor([[[4.0, 0.0, 0.0]]]))
    assert torch.allclose(
        adapted["obj_trans_target"][0, :, 0], torch.tensor([0.0, 1.0, 3.0, 4.0])
    )
    assert torch.allclose(
        adapted["obj_vel_target"][0, :, 0], torch.tensor([0.0, 30.0, 60.0, 30.0])
    )


def test_method_object_velocity_frames_round_trip_to_world():
    batch = _make_batch(seq_len=4)
    root = axis_angle_to_matrix(torch.tensor([0.0, math.pi / 2, 0.0]))
    batch["rotation_global"][:, :, 0] = root
    imu_rot = torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 4, 6, 1, 1)
    imu_rot[:, :, 0] = root
    batch["human_imu"][..., 3:9] = matrix_to_rotation_6d(
        imu_rot.reshape(-1, 3, 3)
    ).reshape(1, 4, 6, 6)
    batch["obj_trans"][0, :, 0] = torch.arange(4) / 30.0
    expected_world = torch.zeros(1, 4, 3)
    expected_world[:, 1:, 0] = 1.0
    local_pose = batch["rotation_global"].clone()

    tip = adapt_batch(batch, "tip", protocol_version=2)
    tip_world, tip_frame = object_velocity_to_world(
        "tip", tip["obj_vel_target"], batch, local_pose, 2
    )
    assert tip_frame == "initial_root"
    assert torch.allclose(tip_world, expected_world, atol=1e-6)

    globalpose = adapt_batch(batch, "globalpose", protocol_version=2)
    gp_world, gp_frame = object_velocity_to_world(
        "globalpose", globalpose["obj_vel_target"], batch, local_pose, 2
    )
    assert gp_frame == "current_root"
    assert torch.allclose(gp_world, expected_world, atol=1e-6)


def test_from_to_rotation_handles_antiparallel_vectors():
    source = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    rotation = from_to_rotation_matrix(source, target)
    mapped = rotation.matmul(source.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(mapped, target, atol=1e-5)
    assert torch.allclose(
        rotation.transpose(-1, -2).matmul(rotation),
        torch.eye(3).expand(2, 3, 3),
        atol=1e-5,
    )


def test_official_column_r6d_round_trip_and_protocol_targets():
    rotations = axis_angle_to_matrix(
        torch.tensor([[0.4, -0.2, 0.7], [-0.3, 0.8, 0.1]])
    )
    encoded = rotation_matrix_to_r6d_columns(rotations)
    decoded = r6d_columns_to_rotation_matrix(encoded)
    assert torch.allclose(decoded, rotations, atol=1e-6)

    batch = _make_batch(seq_len=2)
    batch["rotation_global"][:, :, 1] = rotations[0]
    corrected = adapt_batch(batch, "transpose", protocol_version=2)
    legacy = adapt_batch(batch, "transpose", protocol_version=1)
    expected_column = rotation_matrix_to_r6d_columns(rotations[0])
    expected_row = matrix_to_rotation_6d(rotations[0])
    assert torch.allclose(corrected["pose_target"][0, 0, :6], expected_column)
    assert torch.allclose(legacy["pose_target"][0, 0, :6], expected_row)


def test_tip_uses_interleaved_two_axis_rotation_layout():
    rotation = axis_angle_to_matrix(torch.tensor([0.4, -0.2, 0.7]))
    encoded = rotation_matrix_to_tip_2axis(rotation)
    assert torch.allclose(encoded, rotation[:, :2].reshape(6))
    assert not torch.allclose(encoded, rotation_matrix_to_r6d_columns(rotation))
    assert torch.allclose(tip_2axis_to_rotation_matrix(encoded), rotation, atol=1e-6)

    batch = _make_batch(seq_len=2)
    # TIP joint 1 is local, so update the flattened local axis-angle target too.
    batch["pose"][:, :, 3:6] = torch.tensor([0.4, -0.2, 0.7])
    adapted = adapt_batch(batch, "tip", protocol_version=2)
    assert torch.allclose(adapted["state_target"][0, 0, 6:12], encoded)


def test_tip_noise_preserves_column_layout():
    """Noise must decode/re-encode TIP's first-two-column representation."""
    rotation = axis_angle_to_matrix(torch.tensor([0.4, -0.2, 0.7]))
    encoded = rotation_matrix_to_tip_2axis(rotation).view(1, 1, 6)
    cfg = {
        "acc_bias_std": 0.0,
        "acc_bias_rw_std": 0.0,
        "acc_noise_std": 0.0,
        "acc_scale_std": 0.0,
        "acc_misalignment_std": 0.0,
        "ori_bias_std": 0.0,
        "ori_rw_std": 0.0,
        "ori_noise_std": 0.0,
        "ori_misalignment_std": 0.0,
    }
    acc = torch.zeros(1, 1, 3)
    noisy_acc, noisy_rot, _ = apply_imu_noise(
        acc,
        encoded,
        fps=30.0,
        noise_cfg=cfg,
        rotation_layout="tip_columns",
    )
    assert torch.allclose(noisy_acc, acc)
    assert torch.allclose(noisy_rot, encoded, atol=1e-6)


def test_tip_imu_input_matches_official_mixed_root_frame_protocol():
    batch = _make_batch(seq_len=2)
    root = axis_angle_to_matrix(torch.tensor([0.0, 0.0, math.pi / 2]))
    relative = axis_angle_to_matrix(torch.tensor([0.3, -0.2, 0.1]))
    imu_rot = torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 2, 6, 1, 1)
    imu_rot[:, :, 0] = root
    imu_rot[:, :, 4] = relative
    acc_root_frame = torch.zeros(1, 2, 6, 3)
    acc_root_frame[:, :, 0] = torch.tensor([1.0, 0.0, 0.0])
    acc_root_frame[:, :, 4] = torch.tensor([0.0, 1.0, 0.0])
    batch["human_imu"] = torch.cat(
        [
            acc_root_frame,
            matrix_to_rotation_6d(imu_rot.reshape(-1, 3, 3)).reshape(1, 2, 6, 6),
        ],
        dim=-1,
    )

    tip_input = _tip_input(batch)
    orientation = tip_input[:, :, :54].reshape(1, 2, 6, 3, 3)
    acceleration = tip_input[:, :, 54:72].reshape(1, 2, 6, 3)
    assert torch.allclose(orientation[0, 0, 0], root, atol=1e-6)
    # IMUHOI sensor 4 becomes TIP sensor 1.
    assert TIP_ORDER_FROM_IMUHOI[1] == 4
    assert torch.allclose(orientation[0, 0, 1], relative, atol=1e-6)
    # Stored non-root acceleration is (a_sensor - a_root) in the root frame.
    expected_absolute_root = torch.tensor([1.0, 1.0, 0.0])
    assert torch.allclose(
        acceleration[0, 0, 1], expected_absolute_root, atol=1e-6
    )


def test_tip_object_imu_uses_root0_two_axis_and_acceleration_sum():
    batch = _make_batch(seq_len=45)
    root = axis_angle_to_matrix(torch.tensor([0.0, math.pi / 2, 0.0]))
    object_rotation = axis_angle_to_matrix(torch.tensor([0.2, -0.3, 0.1]))
    human_rot = torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 45, 6, 1, 1)
    human_rot[:, :, 0] = root
    batch["human_imu"][..., 3:9] = matrix_to_rotation_6d(
        human_rot.reshape(-1, 3, 3)
    ).reshape(1, 45, 6, 6)
    batch["obj_imu"][..., :3] = torch.tensor([1.0, 0.0, 0.0])
    batch["obj_imu"][..., 3:9] = matrix_to_rotation_6d(
        object_rotation.view(1, 3, 3).repeat(45, 1, 1)
    ).reshape(1, 45, 6)

    obj_imu = adapt_batch(batch, "tip", protocol_version=2)["obj_imu"]
    expected_acc = torch.einsum(
        "i,ij->j", torch.tensor([1.0, 0.0, 0.0]), root
    )
    expected_orientation = rotation_matrix_to_tip_2axis(
        root.transpose(-1, -2).matmul(object_rotation)
    )
    assert torch.allclose(obj_imu[0, 0, :3], expected_acc, atol=1e-6)
    assert torch.allclose(
        obj_imu[0, 0, 3:9], expected_orientation, atol=1e-6
    )
    assert torch.allclose(obj_imu[0, 0, 9:12], expected_acc / 15.0)
    assert torch.allclose(obj_imu[0, 39, 9:12], expected_acc * (40.0 / 15.0))
    assert torch.allclose(obj_imu[0, 44, 9:12], expected_acc * (40.0 / 15.0))


def test_tip_pose_state_uses_nimble_joint_order():
    assert TIP_POSE_18 == [
        0, 1, 4, 7, 3, 6, 9, 13, 16, 18, 12, 15, 14, 17, 19, 2, 5, 8
    ]


def test_tip_training_context_preserves_rolling_sum_and_next_state_length():
    batch = _make_batch(seq_len=80)
    batch["human_imu"][:, :, 0, 0] = 1.0
    adapted = adapt_batch(
        batch,
        "tip",
        protocol_version=2,
        tip_history_context=39,
    )
    assert adapted["imu"].shape[1] == 41
    assert adapted["state_target"].shape[1] == 41
    # Root acceleration-sum channel at source frame 39 contains 40 samples.
    assert torch.allclose(
        adapted["imu"][0, 0, 72], torch.tensor(40.0 / 15.0), atol=1e-6
    )
    # Training slices off the last IMU and predicts the remaining 40 next states.
    assert adapted["imu"][:, :-1].shape[1] == 40


def test_globalpose_gravity_is_world_up_expressed_in_root_frame():
    batch = _make_batch(seq_len=2)
    root = axis_angle_to_matrix(torch.tensor([0.4, -0.2, 0.7]))
    imu_rot = torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 2, 6, 1, 1)
    imu_rot[:, :, 0] = root
    batch["human_imu"] = torch.cat(
        [
            torch.zeros(1, 2, 6, 3),
            matrix_to_rotation_6d(imu_rot.reshape(-1, 3, 3)).reshape(1, 2, 6, 6),
        ],
        dim=-1,
    )
    batch["rotation_global"][:, :, 0] = root
    expected = -root[1]
    gp_input = _globalpose_input(batch, fps=30.0)
    adapted = adapt_batch(batch, "globalpose", protocol_version=2)
    assert torch.allclose(gp_input[0, 0, 81:84], expected, atol=1e-6)
    assert torch.allclose(adapted["target"][0, 0, 15:18], expected, atol=1e-6)


def test_globalpose_uses_exact_mesh_sensor_positions_when_available():
    batch = _make_batch(seq_len=4)
    exact = torch.zeros(1, 4, 6, 3)
    exact[:, :, 0, 0] = 10.0
    exact[:, :, 5, 0] = 2.0
    batch["globalpose_sensor_positions"] = exact
    adapted = adapt_batch(batch, "globalpose", protocol_version=2)
    # PL pRB is the first five mesh sensors relative to the root, expressed
    # in the root frame.  The joint-proxy fallback would produce 0.01 here.
    assert torch.allclose(adapted["target"][0, 0, 0], torch.tensor(8.0))
    assert torch.allclose(adapted["target"][0, 0, 1:3], torch.zeros(2))


def test_geodesic_angle_and_perfect_metrics():
    identity = torch.eye(3).view(1, 1, 3, 3)
    half_turn = axis_angle_to_matrix(torch.tensor([0.0, math.pi, 0.0])).view(
        1, 1, 3, 3
    )
    assert torch.allclose(
        so3_geodesic_degrees(identity, half_turn), torch.tensor([[180.0]])
    )

    batch = _make_batch(seq_len=5)
    prediction = {
        "local_rot": batch["rotation_global"].clone(),
        "joints": batch["position_global"].clone(),
        "root_trans": batch["trans"].clone(),
        "obj_trans": batch["obj_trans"].clone(),
    }
    metrics = comparison_metric_stats(prediction, batch).means()
    assert all(abs(value) < 1e-6 for value in metrics.values())


def test_metrics_prefer_body_model_reconstructed_target_joints():
    batch = _make_batch(seq_len=5)
    target_joints = batch["position_global"].clone()
    batch["position_global"] = torch.randn_like(target_joints) * 100.0
    prediction = {
        "local_rot": batch["rotation_global"].clone(),
        "joints": target_joints.clone(),
        "target_joints": target_joints,
        "root_trans": batch["trans"].clone(),
        "obj_trans": batch["obj_trans"].clone(),
    }
    metrics = comparison_metric_stats(prediction, batch).means()
    assert metrics["human_pos_err_cm"] == 0.0
    assert metrics["hoi_err_cm"] == 0.0


def test_dip_sigma_is_positive_and_nll_is_differentiable():
    model = DIP18HOIModel(
        hidden_size=8,
        input_hidden_size=8,
        output_hidden_size=8,
        num_layers=1,
    )
    output = model(torch.zeros(1, 3, 60), torch.zeros(1, 3, 12))
    assert torch.isfinite(output["pose_sigma"]).all()
    assert (output["pose_sigma"] > 0).all()
    loss = gaussian_nll(
        output["pose_mu"], output["pose_sigma"], torch.zeros(1, 3, 135)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_metric_macro_average_and_stable_missing_schema():
    short = MetricAccumulator()
    short.add("human_pos_err_cm", torch.tensor([0.0]))
    long = MetricAccumulator()
    long.add("human_pos_err_cm", torch.full((100,), 10.0))

    macro = MetricAccumulator()
    macro.add_sequence_means(short)
    macro.add_sequence_means(long)
    assert macro.means()["human_pos_err_cm"] == 5.0

    stable = add_compatibility_aliases({})
    assert all(stable[name] is None for name in METRIC_UNITS)
    assert stable["ang_err"] is None
    assert stable["pos_err"] is None
    assert stable["obj_err"] is None
    assert stable["obj_trans_jitter"] is None


def test_metrics_skip_uninitialized_object_head():
    batch = _make_batch(seq_len=5)
    prediction = {
        "local_rot": batch["rotation_global"].clone(),
        "joints": batch["position_global"].clone(),
        "root_trans": batch["trans"].clone(),
        "obj_trans": torch.full_like(batch["obj_trans"], 100.0),
        "object_metrics_available": False,
    }
    metrics = comparison_metric_stats(prediction, batch).means()
    assert "human_pos_err_cm" in metrics
    assert "obj_err_cm" not in metrics
    assert "hoi_err_cm" not in metrics
    assert "obj_trans_jitter_mm_frame2" not in metrics


def test_tip_sbp_angular_velocity_is_expressed_in_world_frame():
    theta = 0.1
    dt = 0.05
    base = axis_angle_to_matrix(torch.tensor([0.0, 0.0, math.pi / 2]))
    prev = base.matmul(axis_angle_to_matrix(torch.tensor([-theta, 0.0, 0.0])))
    next_rotation = base.matmul(
        axis_angle_to_matrix(torch.tensor([theta, 0.0, 0.0]))
    )
    omega = _angular_velocity_from_pair(prev.numpy(), next_rotation.numpy(), dt)
    assert np.allclose(omega, np.array([0.0, theta / dt, 0.0]), atol=1e-5)


def test_checkpoint_protocol_and_training_reference_defaults():
    args = Namespace(
        n_sbps=None,
        protocol_version=None,
        object_output_mode=None,
        tip_n_heads=None,
        method="transpose",
        dip_variant=None,
    )
    assert _checkpoint_protocol(args, {}) == (
        1,
        "absolute",
        5,
        16,
        False,
        True,
        False,
        True,
        True,
        True,
    )
    checkpoint = {
        "comparison_protocol_version": 2,
        "object_output_mode": "velocity",
        "model_kwargs": {"n_sbps": 4, "n_heads": 8},
        "object_head_causal": True,
    }
    assert _checkpoint_protocol(args, checkpoint) == (
        2,
        "velocity",
        4,
        8,
        True,
        False,
        True,
        False,
        False,
        False,
    )
    assert TIP_N_HEADS == 16
    assert TIP_STATE_NOISE == 0.15
    assert METHOD_TRAINING_DEFAULTS["dip18"]["epochs"] == 80
    assert METHOD_TRAINING_DEFAULTS["dip18"]["scheduler"] == "dip_exponential"
    assert METHOD_TRAINING_DEFAULTS["tip"]["epochs"] == 1100
    assert METHOD_TRAINING_DEFAULTS["tip"]["scheduler"] == "tip_cosine"
    assert METHOD_TRAINING_DEFAULTS["transpose"]["epochs"] == 300
    assert METHOD_TRAINING_DEFAULTS["globalpose"]["epochs"] == 100
    assert METHOD_TRAINING_DEFAULTS["tip"]["window_size"] == 40
    assert METHOD_TRAINING_DEFAULTS["globalpose"]["window_size"] == 120
    assert METHOD_TRAINING_DEFAULTS["dip18"]["obj_loss_weight"] == 1.0
    assert METHOD_TRAINING_DEFAULTS["tip"]["obj_loss_weight"] == 100.0
    assert METHOD_TRAINING_DEFAULTS["transpose"]["obj_loss_weight"] == 1.0
    assert METHOD_TRAINING_DEFAULTS["globalpose"]["obj_loss_weight"] == 1.0

    tip = TIPHOIModel(n_sbps=0)
    assert tip.tf_encode.layers[0].dropout.p == 0.1


def test_checkpoint_protocol_is_inferred_from_raw_v2_weight_shapes():
    args = Namespace(
        n_sbps=None,
        protocol_version=None,
        object_output_mode=None,
        tip_n_heads=None,
        method="transpose",
        dip_variant=None,
    )
    signatures = [
        {"temporal.input_layer.0.weight": torch.empty(1, 60)},
        {"pose_s1.linear1.weight": torch.empty(1, 72)},
        {"plnet.rnn.weight_ih_l0": torch.empty(1, 84)},
        {
            "in_linear.weight": torch.empty(1, 236),
            "linear.weight": torch.empty(134, 1),
        },
    ]
    for state in signatures:
        protocol = _checkpoint_protocol(args, state)
        assert protocol[0] == 2
        assert protocol[1] == "velocity"
        assert protocol[2] == 5


def test_raw_official_transpose_object_checkpoint_uses_corrected_protocol():
    args = Namespace(
        n_sbps=None,
        protocol_version=None,
        object_output_mode=None,
        tip_n_heads=None,
        method="transpose",
        transpose_variant=None,
    )
    state = {
        "base.pose_s1.linear1.weight": torch.empty(256, 72),
        "object_net.net.linear1.weight": torch.empty(128, 84),
    }
    protocol = _checkpoint_protocol(args, state)
    assert protocol[0] == 2
    assert protocol[1] == "velocity"


def test_checkpoint_protocol_infers_tip_sbp_count_from_raw_weights():
    args = Namespace(
        n_sbps=None,
        protocol_version=None,
        object_output_mode=None,
        tip_n_heads=None,
    )
    state = {
        # 114 fixed outputs + four values for each of three SBPs.
        "linear.weight": torch.empty(126, 1),
        # 90 human IMU + 12 object IMU + 123 human state + 3 object feedback.
        "in_linear.weight": torch.empty(1, 228),
    }
    protocol = _checkpoint_protocol(args, state)
    assert protocol[0] == 2
    assert protocol[1] == "velocity"
    assert protocol[2] == 3
    assert protocol[4] is True


def test_v2_human_branches_do_not_consume_object_imu():
    dip = DIP18HOIModel(
        hidden_size=8,
        input_hidden_size=8,
        output_hidden_size=8,
        num_layers=1,
    )
    transpose = TransPoseHOIModel()
    from Comparisons.globalpose import GlobalPoseHOIModel

    globalpose = GlobalPoseHOIModel(hidden_size=8)
    assert dip.temporal.input_layer[0].in_features == 60
    assert dip.obj_head.input_layer[0].in_features == 72
    assert transpose.pose_s1.linear1.in_features == 72
    assert transpose.obj_head.linear1.in_features == 84
    assert globalpose.plnet.rnn.input_size == 84
    assert globalpose.vrnet.rnn.input_size == 243
    assert globalpose.object_with_init is True
    assert globalpose.obj_head.rnn.input_size == 255
    assert globalpose.obj_head.init_net[0].in_features == 3


def test_transpose_v2_loss_has_finite_gradients_and_causal_checkpoint_loads():
    batch = _make_batch(batch_size=2, seq_len=30)
    batch["trans"][:, :, 0] = torch.arange(30).float() / 300.0
    batch["obj_trans"][:, :, 2] = torch.arange(30).float() / 150.0
    adapted = adapt_batch(batch, "transpose", protocol_version=2)
    model = TransPoseHOIModel(
        object_bidirectional=False,
        object_condition_on_full=False,
    )
    output = model(adapted["imu"], adapted["obj_imu"])
    losses = TransPoseLoss(obj_weight=1.0)(output, adapted)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    checkpoint = {
        "model": model.state_dict(),
        "comparison_protocol_version": 2,
        "object_output_mode": "velocity",
        "object_head_causal": True,
        "model_kwargs": {
            "object_bidirectional": False,
            "object_condition_on_full": False,
        },
    }
    args = Namespace(
        n_sbps=None,
        protocol_version=None,
        object_output_mode=None,
        tip_n_heads=None,
    )
    protocol = _checkpoint_protocol(args, checkpoint)
    restored = _build_model(
        "transpose",
        protocol[2],
        protocol[3],
        protocol[4],
        protocol[5],
        protocol[6],
        protocol[7],
        protocol[8],
        protocol[9],
    )
    restored.load_state_dict(_checkpoint_state(checkpoint), strict=True)
    assert restored.object_bidirectional is False
    assert restored.object_condition_on_full is False


def test_transpose_online_chunks_match_full_forward_when_window_covers_sequence():
    torch.manual_seed(7)
    model = TransPoseHOIModel().eval()
    imu = torch.randn(1, 6, 72)
    obj_imu = torch.randn(1, 6, 12)

    with torch.no_grad():
        expected = model(imu, obj_imu)
        actual = model.reconstruct_chunks(
            imu,
            obj_imu,
            past_frames=6,
            future_frames=6,
            window_batch_size=32,
        )

    assert actual.keys() == expected.keys()
    for key in expected:
        assert torch.allclose(actual[key], expected[key], atol=1e-5, rtol=1e-5), key


def test_dip_online_chunks_match_full_forward_when_window_covers_sequence():
    torch.manual_seed(11)
    model = DIP18HOIModel(
        hidden_size=8,
        input_hidden_size=8,
        output_hidden_size=8,
        num_layers=1,
    ).eval()
    imu = torch.randn(2, 6, 60)
    obj_imu = torch.randn(2, 6, 12)

    with torch.no_grad():
        expected = model(imu, obj_imu)
        actual = model.reconstruct_chunks(
            imu,
            obj_imu,
            past_frames=6,
            future_frames=6,
            window_batch_size=16,
        )

    assert torch.allclose(actual["pose_mu"], expected["pose_mu"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual["obj_trans"], expected["obj_trans"], atol=1e-5, rtol=1e-5)


def test_reference_initializers_preserve_human_weights_and_new_object_parameters(tmp_path):
    torch.manual_seed(13)

    tip = TIPHOIModel(n_sbps=5)
    tip_before_object_output = tip.linear.weight[-3:].clone()
    tip_reference = {}
    for key, value in tip.state_dict().items():
        if key == "in_linear.weight":
            tip_reference[key] = torch.randn(value.shape[0], 221)
        elif key == "linear.weight":
            tip_reference[key] = torch.randn(131, value.shape[1])
        elif key == "linear.bias":
            tip_reference[key] = torch.randn(131)
        else:
            tip_reference[key] = torch.randn_like(value)
    tip_path = tmp_path / "tip.pt"
    torch.save(tip_reference, tip_path)
    info = initialize_from_reference("tip", tip, tip_path)
    assert info["object_parameters_initialized"] is False
    assert torch.equal(tip.in_linear.weight[:, :90], tip_reference["in_linear.weight"][:, :90])
    assert torch.count_nonzero(tip.in_linear.weight[:, 90:102]) == 0
    assert torch.equal(tip.in_linear.weight[:, 102:233], tip_reference["in_linear.weight"][:, 90:])
    assert torch.count_nonzero(tip.in_linear.weight[:, 233:]) == 0
    assert torch.equal(tip.linear.weight[:131], tip_reference["linear.weight"])
    assert torch.equal(tip.linear.weight[-3:], tip_before_object_output)

    transpose = TransPoseHOIModel()
    transpose_reference = {
        key: torch.randn_like(value)
        for key, value in transpose.state_dict().items()
        if not key.startswith("obj_head.")
    }
    transpose_object_before = {
        key: value.clone()
        for key, value in transpose.state_dict().items()
        if key.startswith("obj_head.")
    }
    transpose_path = tmp_path / "transpose.pt"
    torch.save(transpose_reference, transpose_path)
    initialize_from_reference("transpose", transpose, transpose_path)
    for key, value in transpose_reference.items():
        assert torch.equal(transpose.state_dict()[key], value)
    for key, value in transpose_object_before.items():
        assert torch.equal(transpose.state_dict()[key], value)


def test_training_file_split_is_deterministic_and_disjoint():
    files = [f"/{index}.pt" for index in range(20)]
    train_a, validation_a = split_training_files(files, 0.2, seed=42)
    train_b, validation_b = split_training_files(reversed(files), 0.2, seed=42)
    assert train_a == train_b
    assert validation_a == validation_b
    assert len(train_a) == 16
    assert len(validation_a) == 4
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a + validation_a) == set(files)


def test_official_dip_checkpoint_signature_and_model_load():
    args = Namespace(
        n_sbps=None,
        protocol_version=None,
        object_output_mode=None,
        tip_n_heads=None,
        method="dip18",
        dip_variant=None,
    )
    state = {"glb.linear1.weight": torch.empty(36, 72)}
    from Comparisons.eval_comparison import _checkpoint_dip_variant

    assert _checkpoint_dip_variant(args, state) == "official"
    model = DIP18ReferenceHOIModel()
    assert model.glb.linear1.in_features == 72
    assert model.human_branch.linear2.out_features == 22 * 9
