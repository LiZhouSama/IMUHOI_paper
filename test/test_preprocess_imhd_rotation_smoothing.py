import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from process.preprocess_IMHD import (
    _moving_avg_axis_angle_timewise,
    _moving_avg_timewise,
)


def _step_degrees(rotvec):
    rot = Rotation.from_rotvec(rotvec.reshape(-1, 3))
    return np.degrees((rot[:-1].inv() * rot[1:]).magnitude())


class AxisAngleSmoothingTest(unittest.TestCase):
    def _rotation_crossing_pi(self, frames=40):
        # Constant-speed rotation about a fixed axis passing through theta = pi,
        # where the axis-angle representation wraps.
        angles = np.linspace(np.pi - 0.3, np.pi + 0.3, frames)
        axis = np.array([0.3, 0.8, -0.52])
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(angles[:, None] * axis).as_rotvec().astype(np.float32)

    def test_component_average_breaks_at_pi_wrap(self):
        rotvec = self._rotation_crossing_pi()
        self.assertGreater(_step_degrees(_moving_avg_timewise(rotvec, k=3)).max(), 90.0)

    def test_quaternion_average_stays_continuous_at_pi_wrap(self):
        rotvec = self._rotation_crossing_pi()
        raw_step = _step_degrees(rotvec).max()
        smoothed = _moving_avg_axis_angle_timewise(rotvec, k=3)
        self.assertEqual(smoothed.shape, rotvec.shape)
        self.assertEqual(smoothed.dtype, rotvec.dtype)
        self.assertLess(_step_degrees(smoothed).max(), raw_step * 1.5)
        deviation = np.degrees(
            (Rotation.from_rotvec(rotvec).inv() * Rotation.from_rotvec(smoothed)).magnitude()
        )
        self.assertLess(deviation.max(), 0.5)

    def test_multi_joint_flat_layout(self):
        rng = np.random.default_rng(0)
        frames, joints = 30, 22
        base = rng.normal(size=(joints, 3))
        drift = np.linspace(0.0, 0.2, frames)[:, None, None]
        rotvec = (base[None] + drift).reshape(frames, joints * 3).astype(np.float32)
        smoothed = _moving_avg_axis_angle_timewise(rotvec, k=3)
        self.assertEqual(smoothed.shape, (frames, joints * 3))
        deviation = np.degrees(
            (
                Rotation.from_rotvec(rotvec.reshape(-1, 3)).inv()
                * Rotation.from_rotvec(smoothed.reshape(-1, 3))
            ).magnitude()
        )
        self.assertLess(deviation.max(), 1.0)


if __name__ == "__main__":
    unittest.main()
