import torch
import torch.nn as nn

from model.rnn.human_pose import HumanPoseModule
from model.rnn.online import update_data_inits_from_history


def _module(*, prevent_floor_penetration=True):
    module = HumanPoseModule.__new__(HumanPoseModule)
    nn.Module.__init__(module)
    module.no_trans = False
    module.fps = 30.0
    module.prevent_floor_penetration = prevent_floor_penetration
    module.floor_height = 0.0
    module._foot_joint_indices = (7, 8)
    module.prob_threshold = (0.5, 0.9)
    module.gravity_velocity = torch.tensor([0.0, -0.018 * module.fps, 0.0])
    return module


def test_window_first_foot_velocity_uses_predecessor_position():
    module = _module(prevent_floor_penetration=False)
    root_vel_local = torch.zeros(1, 1, 3)
    contact_logits = torch.tensor([[[10.0, -10.0]]])
    joints = torch.zeros(1, 1, 24, 3)
    joints[:, :, 7, 0] = 1.0
    root_rotation = torch.eye(3).view(1, 1, 3, 3)
    previous_feet = torch.zeros(1, 2, 3)

    velocity = module._fuse_velocities_batched(
        root_vel_local,
        contact_logits,
        joints,
        root_rotation,
        previous_foot_pos=previous_feet,
    )

    assert torch.allclose(velocity[0, 0, 0], torch.tensor(-30.0), atol=1e-5)
    assert torch.allclose(
        velocity[0, 0, 1], torch.tensor(-0.018 * module.fps), atol=1e-5
    )


def test_root_velocity_integration_keeps_frame_zero_at_init():
    module = _module(prevent_floor_penetration=False)
    velocity = torch.tensor(
        [[[9.0, 6.0, 3.0], [3.0, -6.0, 9.0], [6.0, 3.0, -3.0]]]
    )

    displacement = module._integrate_root_velocity(velocity)

    assert torch.equal(displacement[:, 0], torch.zeros(1, 3))
    assert torch.allclose(displacement[:, 1], velocity[:, 1] / module.fps)
    assert torch.allclose(
        displacement[:, 2], (velocity[:, 1] + velocity[:, 2]) / module.fps
    )


def test_contact_ground_correction_lowers_hovering_feet_during_squat():
    module = _module()
    root = torch.zeros(1, 2, 3)
    root[..., 1] = 1.0
    joints = torch.zeros(1, 2, 24, 3)
    joints[:, 0, 7:9, 1] = -1.0
    joints[:, 1, 7:9, 1] = -0.5
    contact_logits = torch.full((1, 2, 2), 10.0)

    corrected, correction = module._apply_ground_correction(
        root, joints, contact_logits
    )

    assert torch.allclose(corrected[:, 0], root[:, 0])
    assert torch.allclose(corrected[:, 1, 1], torch.tensor([0.5]), atol=1e-5)
    foot_world_y = joints[:, 1, 7:9, 1] + corrected[:, 1, 1].unsqueeze(-1)
    assert torch.allclose(foot_world_y, torch.zeros_like(foot_world_y), atol=1e-5)
    assert correction[0, 1] < 0.0


def test_noncontact_swing_foot_is_not_forced_down():
    module = _module()
    root = torch.zeros(1, 2, 3)
    root[..., 1] = 1.0
    joints = torch.zeros(1, 2, 24, 3)
    joints[:, 0, 7:9, 1] = -1.0
    joints[:, 1, 7:9, 1] = -0.5
    contact_logits = torch.full((1, 2, 2), -10.0)

    corrected, correction = module._apply_ground_correction(
        root, joints, contact_logits
    )

    assert torch.allclose(corrected, root)
    assert torch.equal(correction, torch.zeros_like(correction))


def test_later_contact_uses_shared_ground_height_not_initial_swing_height():
    module = _module()
    root = torch.zeros(1, 2, 3)
    root[..., 1] = 1.0
    joints = torch.zeros(1, 2, 24, 3)
    joints[:, 0, 7, 1] = -1.0
    joints[:, 0, 8, 1] = -0.7
    joints[:, 1, 7, 1] = -0.3
    joints[:, 1, 8, 1] = -0.5
    contact_logits = torch.tensor([[[-10.0, -10.0], [-10.0, 10.0]]])

    corrected, _ = module._apply_ground_correction(
        root, joints, contact_logits
    )

    right_foot_world_y = joints[:, 1, 8, 1] + corrected[:, 1, 1]
    assert torch.allclose(right_foot_world_y, torch.zeros_like(right_foot_world_y))


def test_world_height_penetration_correction_includes_initial_root_height():
    module = _module()
    root = torch.zeros(1, 2, 3)
    root[..., 1] = 1.0
    joints = torch.zeros(1, 2, 24, 3)
    joints[:, 0, 7:9, 1] = -1.0
    joints[:, 1, 7:9, 1] = -1.1
    contact_logits = torch.full((1, 2, 2), -10.0)

    corrected, correction = module._apply_ground_correction(
        root, joints, contact_logits
    )

    assert torch.allclose(corrected[:, 0, 1], torch.tensor([1.0]))
    assert torch.allclose(corrected[:, 1, 1], torch.tensor([1.1]), atol=1e-6)
    assert torch.allclose(correction[:, 1], torch.tensor([0.1]), atol=1e-6)


def test_online_frame_advances_translation_once():
    module = _module(prevent_floor_penetration=False)
    previous_trans = torch.tensor([[[1.0, 2.0, 3.0]]])
    history = {
        "root_trans_pred": previous_trans,
    }
    raw_velocity = torch.tensor([[[0.3, -0.6, 0.9]]])
    frame = {
        "root_vel_unadjusted_pred": raw_velocity,
        "root_trans_pred": torch.full((1, 1, 3), 99.0),
        "pred_joints_local": torch.zeros(1, 1, 24, 3),
    }

    published = module.finalize_online_frame(frame, history)

    expected_trans = previous_trans + raw_velocity / module.fps
    assert torch.allclose(published["root_trans_pred"], expected_trans)
    assert torch.allclose(published["root_vel_pred"], raw_velocity)
    implied_velocity = (
        published["root_trans_pred"] - previous_trans
    ) * module.fps
    assert torch.allclose(implied_velocity, published["root_vel_pred"], atol=1e-6)


def test_online_init_update_carries_feet_and_window_trans():
    history = {
        "pred_foot_pos": torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3),
        "root_trans_pred": torch.arange(9, dtype=torch.float32).reshape(1, 3, 3),
    }

    updated = update_data_inits_from_history(
        {}, history, index=1, trans_index=2
    )

    assert torch.equal(updated["prev_foot_pos"], history["pred_foot_pos"][:, 1])
    assert torch.equal(updated["trans_init"], history["root_trans_pred"][:, 2])
