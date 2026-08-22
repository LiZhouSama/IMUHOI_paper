import math
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hoi_metrics import compute_hoi_error_at_gt_contact


def test_hoi_error_uses_relative_displacement_vector_at_gt_contact_frames():
    gt_obj = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pred_obj = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    gt_hands = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    pred_hands = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 3.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )

    error_cm = compute_hoi_error_at_gt_contact(
        pred_obj,
        gt_obj,
        pred_hands,
        gt_hands,
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
    )

    assert error_cm == 200.0


def test_hoi_error_is_nan_without_ground_truth_interaction():
    positions = torch.zeros(2, 3)
    hands = torch.zeros(2, 2, 3)

    error_cm = compute_hoi_error_at_gt_contact(
        positions,
        positions,
        hands,
        hands,
        torch.tensor([False, False]),
        torch.tensor([False, False]),
    )

    assert math.isnan(error_cm)