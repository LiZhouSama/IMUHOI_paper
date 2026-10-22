import math
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hoi_metrics import compute_hoi_error_at_gt_contact
from utils.hoi_metrics import compute_hoi_error_egoimu, hoi_metric_metadata
from types import SimpleNamespace
import pytest


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


def test_egoimu_counts_both_hands_and_unheld_root_observations():
    obj = torch.zeros(3, 3, dtype=torch.float64)
    gt = torch.zeros(3, 24, 3, dtype=torch.float64)
    pred = gt.clone()
    pred[0, 20, 0] = 0.01
    pred[0, 21, 0] = 0.03
    pred[1, 21, 0] = 0.05
    pred[2, 0, 0] = 0.11
    # Palms must not replace wrists; idle-frame wrists must not be counted.
    pred[:, 22:24] = 100
    pred[2, 20:22] = 100
    error = compute_hoi_error_egoimu(obj, obj, pred, gt,
                                    torch.tensor([True, False, False]),
                                    torch.tensor([True, True, False]))
    assert error == pytest.approx((1 + 3 + 5 + 11) / 4)


def test_egoimu_all_unheld_sequences_are_finite_and_translation_invariant():
    gt_obj = torch.zeros(2, 3, dtype=torch.float64)
    pred_obj = gt_obj + torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
    gt = torch.zeros(2, 22, 3, dtype=torch.float64)
    pred = gt.clone()
    labels = torch.zeros(2, dtype=torch.bool)
    error = compute_hoi_error_egoimu(pred_obj, gt_obj, pred, gt, labels, labels)
    assert error == pytest.approx(10)
    shift = torch.tensor([5., -3., 7.], dtype=torch.float64)
    shifted = compute_hoi_error_egoimu(pred_obj + shift, gt_obj + shift, pred + shift, gt + shift, labels, labels)
    assert shifted == pytest.approx(error)


def test_metric_protocol_defaults_and_unknown_protocol():
    assert hoi_metric_metadata(SimpleNamespace())["hoi_metric_protocol"] == "contact_palm"
    assert hoi_metric_metadata(SimpleNamespace(eval_hoi_protocol="egoimu"))["hoi_noncontact_reference"] == "root_0"
    with pytest.raises(ValueError, match="Unknown"):
        hoi_metric_metadata(SimpleNamespace(eval_hoi_protocol="typo"))


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_common_evaluator_uses_egoimu_protocol_for_both_inference_modes(monkeypatch, mode):
    import eval_IMUHOI as evaluator
    from easydict import EasyDict

    frames = 3
    prediction = torch.zeros(1, frames, 24, 3)
    prediction[0, 0, 20, 0] = .01
    prediction[0, 0, 21, 0] = .03
    prediction[0, 1, 21, 0] = .05
    prediction[0, 2, 0, 0] = .11
    class Model(torch.nn.Module):
        def inference(self, *args, **kwargs):
            assert kwargs["inference_mode"] == mode
            return {"pred_joints_global": prediction,
                    "root_trans_pred": torch.zeros(1, frames, 3),
                    "pred_obj_trans": torch.zeros(1, frames, 3)}
    batch = {"position_global": torch.zeros(1, frames, 24, 3), "trans": torch.zeros(1, frames, 3),
             "obj_trans": torch.zeros(1, frames, 3),
             "lhand_contact": torch.tensor([[True, False, False]]),
             "rhand_contact": torch.tensor([[True, True, False]])}
    data = {"human_imu": torch.zeros(1, frames, 6, 9), "has_object": torch.ones(1, dtype=torch.bool)}
    monkeypatch.setattr(evaluator, "build_model_input_dict", lambda *a, **kw: data)
    monkeypatch.setattr(evaluator, "_infer_model_kind", lambda model: "pipeline")
    monkeypatch.setattr(evaluator, "_build_published_object_trans", lambda raw, *a: raw)
    cfg = EasyDict(eval_hoi_protocol="egoimu")
    metrics = evaluator.evaluate_model(Model(), [batch], cfg, torch.device("cpu"),
                                        evaluate_contacts=False, inference_mode=mode, compare_three=False)
    assert metrics["hoi_err_fusion"] == pytest.approx(5)
    assert metrics["hoi_err_fusion_egoimu"] == pytest.approx(5)
    assert metrics["hoi_err_fusion_contact_palm"] == pytest.approx(0)
