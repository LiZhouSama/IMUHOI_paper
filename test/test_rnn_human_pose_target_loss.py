import torch

from train.rnn.loss.human_pose_loss import HumanPoseLoss


def _batch(seq_len=5):
    human_imu = torch.zeros(1, seq_len, 6, 9)
    human_imu[..., -6:] = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    return {
        "human_imu": human_imu,
        "trans": torch.zeros(1, seq_len, 3),
    }


def test_target_only_loss_ignores_non_target_frames():
    batch = _batch()
    pred = {"root_trans_pred": torch.zeros(1, 5, 3)}
    pred["root_trans_pred"][:, 0] = 100.0
    pred["root_trans_pred"][:, 4] = 100.0

    full_loss, _, _ = HumanPoseLoss()(pred, batch, torch.device("cpu"))
    target_loss, losses, _ = HumanPoseLoss(target_frame_index=2)(
        pred, batch, torch.device("cpu")
    )

    assert full_loss.item() > 0.0
    assert target_loss.item() == 0.0
    assert losses["root_trans"].item() == 0.0


def test_target_only_loss_uses_requested_online_target_index():
    batch = _batch()
    pred = {"root_trans_pred": torch.zeros(1, 5, 3)}
    pred["root_trans_pred"][:, 2] = 2.0

    target_loss, losses, _ = HumanPoseLoss(target_frame_index=2)(
        pred, batch, torch.device("cpu")
    )

    assert target_loss.item() > 0.0
    assert losses["root_trans"].item() == 4.0


def test_foot_contact_logits_receive_supervision():
    batch = _batch(seq_len=2)
    batch["lfoot_contact"] = torch.tensor([[1.0, 0.0]])
    batch["rfoot_contact"] = torch.tensor([[0.0, 1.0]])
    pred = {"contact_pred": torch.zeros(1, 2, 2)}

    total_loss, losses, weighted = HumanPoseLoss(weights={"contact": 2.0})(
        pred, batch, torch.device("cpu")
    )

    expected = torch.log(torch.tensor(2.0))
    assert torch.allclose(losses["contact"], expected)
    assert torch.allclose(weighted["contact"], expected * 2.0)
    assert torch.allclose(total_loss, expected * 2.0)
