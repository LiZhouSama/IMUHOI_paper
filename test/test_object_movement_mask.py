import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from process.preprocess import (
    OBJ_MOTION_HALF_WINDOW_FRAMES,
    compute_improved_contact_labels,
    compute_object_movement_mask,
)


def _jittered_static_track(frames, rng, trans_sigma_m=0.0008, rot_sigma_deg=0.5):
    trans = np.array([0.3, 0.8, 0.2]) + rng.normal(0.0, trans_sigma_m, (frames, 3))
    rot = Rotation.from_rotvec(
        np.radians(rng.normal(0.0, rot_sigma_deg, (frames, 3)))
    ).as_matrix()
    return trans, rot


class ObjectMovementMaskTest(unittest.TestCase):
    def test_fitting_jitter_on_static_object_is_not_movement(self):
        rng = np.random.default_rng(0)
        trans, rot = _jittered_static_track(300, rng)
        # The removed per-frame rule (0.8 mm or |dR|_F > 0.005) fires on most of these frames.
        per_frame_rot = np.linalg.norm((rot[1:] - rot[:-1]).reshape(299, -1), axis=-1)
        self.assertGreater((per_frame_rot > 0.005).mean(), 0.5)

        moving = compute_object_movement_mask(torch.from_numpy(trans), torch.from_numpy(rot))
        self.assertEqual(moving.dtype, torch.bool)
        self.assertEqual(moving.shape, (300,))
        self.assertLess(moving.float().mean().item(), 0.02)

    def test_carried_object_is_detected_with_window_margin(self):
        rng = np.random.default_rng(1)
        frames, move_start, move_end = 240, 90, 150
        trans, rot = _jittered_static_track(frames, rng)
        # 0.5 m/s carry between move_start and move_end, static elsewhere.
        offset = np.clip(np.arange(frames) - move_start, 0, move_end - move_start) / 30.0 * 0.5
        trans[:, 0] += offset
        moving = compute_object_movement_mask(trans, rot).numpy()

        self.assertTrue(moving[move_start + 3:move_end - 3].all())
        margin = OBJ_MOTION_HALF_WINDOW_FRAMES + 3
        self.assertFalse(moving[:move_start - margin].any())
        self.assertFalse(moving[move_end + margin:].any())

    def test_pure_rotation_is_detected(self):
        frames = 120
        angles = np.clip(np.arange(frames) - 40, 0, 40) * np.radians(2.0)  # 60 deg/s
        rot = Rotation.from_rotvec(angles[:, None] * np.array([0.0, 1.0, 0.0])).as_matrix()
        trans = np.zeros((frames, 3))
        moving = compute_object_movement_mask(trans, rot).numpy()
        self.assertTrue(moving[45:75].all())
        self.assertFalse(moving[:30].any())
        self.assertFalse(moving[90:].any())

    def test_short_sequences(self):
        self.assertEqual(compute_object_movement_mask(np.zeros((1, 3)), np.eye(3)[None]).shape, (1,))
        trans = np.linspace(0.0, 0.2, 6)[:, None] * np.array([[1.0, 0.0, 0.0]])
        moving = compute_object_movement_mask(trans, np.repeat(np.eye(3)[None], 6, axis=0))
        self.assertEqual(moving.shape, (6,))
        self.assertTrue(moving.all())

    def test_contact_labels_follow_supplied_mask_and_reject_bad_length(self):
        frames = 20
        obj_trans = torch.zeros(frames, 3)
        obj_rot = torch.eye(3).repeat(frames, 1, 1)
        positions = torch.zeros(frames, 24, 3)
        with self.assertRaises(ValueError):
            compute_improved_contact_labels(
                obj_trans, obj_rot, torch.ones(frames), positions,
                "unused_mesh_dir", "unused", torch.device("cpu"), frames,
                obj_movement_mask=torch.zeros(frames - 1, dtype=torch.bool),
            )
        # An all-static mask short-circuits before any mesh is loaded.
        lhand, rhand, obj = compute_improved_contact_labels(
            obj_trans, obj_rot, torch.ones(frames), positions,
            "unused_mesh_dir", "unused", torch.device("cpu"), frames,
            obj_movement_mask=torch.zeros(frames, dtype=torch.bool),
        )
        self.assertFalse(obj.any() or lhand.any() or rhand.any())


if __name__ == "__main__":
    unittest.main()
