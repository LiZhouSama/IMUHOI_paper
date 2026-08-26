import torch
import torch.nn as nn

from model.rnn.human_pose import HumanPoseModule
from model.rnn.online import update_data_inits_from_history


GRAVITY_BIAS_MPS = -0.018


def _module(*, prevent_floor_penetration=True, floor_height=0.0):
    module = HumanPoseModule.__new__(HumanPoseModule)
    nn.Module.__init__(module)
    module.no_trans = False
    module.fps = 30.0
    module.prevent_floor_penetration = prevent_floor_penetration
    module.floor_height = floor_height
    module._foot_joint_indices = (7, 8)
    module.prob_threshold = (0.5, 0.9)
    module.gravity_velocity = torch.tensor([0.0, GRAVITY_BIAS_MPS, 0.0])
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
        velocity[0, 0, 1], torch.tensor(GRAVITY_BIAS_MPS), atol=1e-5
    )


def test_gravity_bias_is_a_velocity_not_a_per_frame_displacement():
    """The bias is a slow drift corrector, not the dominant vertical signal.

    Read as a per-frame displacement it sinks the root about a metre per
    60-frame window, so the floor clamp ends up setting the root height on
    every supported frame and the network's vertical output is discarded.
    """
    seq_len = 60
    module = _module(prevent_floor_penetration=False)
    root_vel_local = torch.zeros(1, seq_len, 3)
    contact_logits = torch.full((1, seq_len, 2), 10.0)
    joints = torch.zeros(1, seq_len, 24, 3)
    root_rotation = torch.eye(3).view(1, 1, 3, 3).expand(1, seq_len, 3, 3)

    velocity = module._fuse_velocities_batched(
        root_vel_local, contact_logits, joints, root_rotation
    )
    sink = module._integrate_root_velocity(velocity)[0, -1, 1]

    expected = GRAVITY_BIAS_MPS * (seq_len - 1) / module.fps
    assert torch.allclose(sink, torch.tensor(expected), atol=1e-6)
    assert abs(float(sink)) < 0.05


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


def test_feet_above_the_floor_are_never_pulled_down():
    """Ankles clear the ground by ~0.08 m while standing; keep them there.

    The contact head is trained on a speed threshold, so a stationary foot may
    legitimately hover (standing, sitting, kneeling, standing on a box).
    """
    module = _module()
    root = torch.zeros(1, 3, 3)
    root[..., 1] = 1.0
    joints = torch.zeros(1, 3, 24, 3)
    joints[:, :, 7:9, 1] = -0.92

    corrected, correction = module._apply_ground_correction(root, joints)

    assert torch.allclose(corrected, root)
    assert torch.equal(correction, torch.zeros_like(correction))


def test_world_height_penetration_correction_includes_initial_root_height():
    module = _module()
    root = torch.zeros(1, 2, 3)
    root[..., 1] = 1.0
    joints = torch.zeros(1, 2, 24, 3)
    joints[:, 0, 7:9, 1] = -1.0
    joints[:, 1, 7:9, 1] = -1.1

    corrected, correction = module._apply_ground_correction(root, joints)

    assert torch.allclose(corrected[:, 0, 1], torch.tensor([1.0]))
    assert torch.allclose(corrected[:, 1, 1], torch.tensor([1.1]), atol=1e-6)
    assert torch.allclose(correction[:, 1], torch.tensor([0.1]), atol=1e-6)


def test_floor_height_is_the_ankle_clearance_not_the_sole():
    """Joints 7/8 are ankles, so the clamp plane sits one ankle above the sole."""
    module = _module(floor_height=0.07)
    root = torch.zeros(1, 3, 3)
    root[..., 1] = 1.0
    joints = torch.zeros(1, 3, 24, 3)
    joints[:, 0, 7:9, 1] = -0.90  # ankle at 0.10, standing clear of the plane
    joints[:, 1, 7:9, 1] = -0.93  # ankle exactly on the plane
    joints[:, 2, 7:9, 1] = -0.95  # ankle 0.02 below the plane

    _, correction = module._apply_ground_correction(root, joints)

    assert torch.allclose(correction[:, 1], torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(correction[:, 2], torch.tensor([0.02]), atol=1e-6)


def test_penetration_correction_never_decreases_within_a_window():
    module = _module()
    root = torch.zeros(1, 3, 3)
    root[..., 1] = 1.0
    joints = torch.zeros(1, 3, 24, 3)
    joints[:, 0, 7:9, 1] = -1.0
    joints[:, 1, 7:9, 1] = -1.2  # 0.2 m below the floor
    joints[:, 2, 7:9, 1] = -0.9  # clears the floor again

    corrected, correction = module._apply_ground_correction(root, joints)

    assert torch.allclose(correction[:, 1], torch.tensor([0.2]), atol=1e-6)
    assert torch.allclose(correction[:, 2], correction[:, 1], atol=1e-6)
    assert torch.allclose(corrected[:, 2, 1], torch.tensor([1.2]), atol=1e-6)


def test_online_frame_penetration_correction_matches_offline_recursion():
    """The single-frame online update reproduces the offline cummax offsets."""
    module = _module()
    root = torch.zeros(1, 3, 3)
    root[..., 1] = 1.0
    joints = torch.zeros(1, 3, 24, 3)
    joints[:, 0, 7:9, 1] = -1.0
    joints[:, 1, 7:9, 1] = -1.2
    joints[:, 2, 7:9, 1] = -1.05

    offline, _ = module._apply_ground_correction(root, joints)

    online = root[:, :1]
    for frame_idx in range(1, root.shape[1]):
        raw = online[:, -1:] + (root[:, frame_idx : frame_idx + 1] - root[:, frame_idx - 1 : frame_idx])
        stepped, _ = module._apply_ground_correction(
            raw, joints[:, frame_idx : frame_idx + 1], preserve_first_frame=False
        )
        online = torch.cat((online, stepped), dim=1)

    assert torch.allclose(online, offline, atol=1e-6)


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
