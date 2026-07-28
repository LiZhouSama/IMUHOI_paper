"""接触段端点检测（heatmap + 解码 + 置信度）的单元测试。"""
import math
import os
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.dataset_IMUHOI import build_contact_endpoint_labels
from model.rnn.velocity_contact import (
    VelocityContactModule,
    decode_contact_segments,
    find_heatmap_peaks,
    recover_contact_state_from_endpoints,
)
from model.rnn.online import update_data_inits_from_history
from train.rnn.loss.velocity_contact_loss import VelocityContactLoss
from train.rnn.train_velocity_contact import (
    VelocityContactTrainer,
    _vc_parameter_group,
    build_velocity_contact_optimizer_and_scheduler,
)


def _contact_mask(length, *runs):
    mask = torch.zeros(length, dtype=torch.bool)
    for start, end in runs:
        mask[start:end + 1] = True
    return mask


# ---------- 标签生成 ----------

def test_endpoint_labels_mark_segment_boundaries():
    length = 40
    lhand = _contact_mask(length, (10, 19))
    rhand = torch.zeros(length, dtype=torch.bool)
    obj = _contact_mask(length, (10, 19))

    hard, gauss = build_contact_endpoint_labels(lhand, rhand, obj)

    assert hard.shape == (length, 6)
    assert gauss.shape == (length, 6)
    # lhand: start@10 / end@19
    assert hard[10, 0] == 1.0 and hard[:, 0].sum() == 1.0
    assert hard[19, 1] == 1.0 and hard[:, 1].sum() == 1.0
    # rhand: 无接触
    assert hard[:, 2].sum() == 0.0 and hard[:, 3].sum() == 0.0
    # obj: start@10 / end@19
    assert hard[10, 4] == 1.0 and hard[19, 5] == 1.0
    # 高斯标签在端点处为峰值1
    assert torch.isclose(gauss[10, 0], torch.tensor(1.0))
    assert torch.isclose(gauss[19, 1], torch.tensor(1.0))
    assert gauss[10, 0] == gauss[:, 0].max()


def test_endpoint_labels_mask_window_truncated_segments():
    length = 40
    obj = _contact_mask(length, (0, 15), (30, 39))
    zeros = torch.zeros(length, dtype=torch.bool)

    hard, gauss = build_contact_endpoint_labels(
        zeros, zeros, obj,
        prev_contact=(False, False, True),
        next_contact=(False, False, True),
    )

    # 段 [0,15] 被窗口头部截断：不打 start，end@15 保留
    assert hard[0, 4] == 0.0
    assert hard[15, 5] == 1.0
    # 段 [30,39] 被窗口尾部截断：start@30 保留，不打 end
    assert hard[30, 4] == 1.0
    assert hard[39, 5] == 0.0
    assert hard[:, 4].sum() == 1.0 and hard[:, 5].sum() == 1.0
    assert gauss[0, 4] < 1.0 and gauss[39, 5] < 1.0


def test_endpoint_labels_keep_genuine_window_border_endpoints():
    length = 40
    obj = _contact_mask(length, (0, 15))
    zeros = torch.zeros(length, dtype=torch.bool)

    hard, _ = build_contact_endpoint_labels(zeros, zeros, obj)

    # 窗口前一帧无接触 → t=0 是真实起点
    assert hard[0, 4] == 1.0


# ---------- 峰值提取 ----------

def test_find_heatmap_peaks_applies_threshold_and_nms():
    prob = torch.zeros(50)
    prob[10] = 0.9
    prob[12] = 0.7   # 距峰值10只有2帧，应被NMS抑制
    prob[30] = 0.6
    prob[45] = 0.2   # 低于阈值

    peaks = find_heatmap_peaks(prob, threshold=0.3, radius=5)

    assert [index for index, _ in peaks] == [10, 30]
    assert abs(peaks[0][1] - 0.9) < 1e-6
    assert abs(peaks[1][1] - 0.6) < 1e-6


# ---------- 解码与置信度 ----------

def test_decode_pairs_peaks_with_geometric_mean_confidence():
    seq_len = 60
    endpoint_prob = torch.zeros(1, seq_len, 6)
    endpoint_prob[0, 10, 4] = 0.9   # obj_start
    endpoint_prob[0, 40, 5] = 0.8   # obj_end
    interior_prob = torch.zeros(1, seq_len, 3)
    interior_prob[0, 10:41, 2] = 0.9

    segments, raster = decode_contact_segments(
        endpoint_prob, interior_prob,
        peak_threshold=0.3, segment_threshold=0.25, peak_radius=5,
    )

    assert len(segments) == 1
    assert len(segments[0]) == 1
    segment = segments[0][0]
    assert segment["channel"] == 2 and segment["channel_name"] == "obj"
    assert segment["start"] == 10 and segment["end"] == 40
    expected_conf = math.exp((math.log(0.9) + math.log(0.8) + math.log(0.9)) / 3)
    assert abs(segment["confidence"] - expected_conf) < 1e-5
    assert abs(segment["start_confidence"] - 0.9) < 1e-6
    assert abs(segment["end_confidence"] - 0.8) < 1e-6
    # 栅格化：段内为置信度，段外为0
    assert torch.allclose(raster[0, 10:41, 2], torch.full((31,), expected_conf), atol=1e-5)
    assert raster[0, 9, 2] == 0.0 and raster[0, 41, 2] == 0.0
    assert raster[0, :, 0].sum() == 0.0 and raster[0, :, 1].sum() == 0.0


def test_decode_emits_open_segments_at_stream_borders():
    seq_len = 60
    endpoint_prob = torch.zeros(1, seq_len, 6)
    endpoint_prob[0, 5, 5] = 0.7    # obj_end，前面没有start：入流时已在接触
    endpoint_prob[0, 50, 4] = 0.8   # obj_start，后面没有end：接触仍在进行
    interior_prob = torch.zeros(1, seq_len, 3)
    interior_prob[0, :6, 2] = 0.8
    interior_prob[0, 50:, 2] = 0.8

    segments, raster = decode_contact_segments(
        endpoint_prob, interior_prob,
        peak_threshold=0.3, segment_threshold=0.25, peak_radius=5,
    )

    assert len(segments[0]) == 2
    open_left, open_right = segments[0]
    assert open_left["start"] == 0 and open_left["end"] == 5
    assert open_left["start_confidence"] is None
    assert abs(open_left["confidence"] - math.sqrt(0.7 * 0.8)) < 1e-5
    assert open_right["start"] == 50 and open_right["end"] == seq_len - 1
    assert open_right["end_confidence"] is None
    assert abs(open_right["confidence"] - math.sqrt(0.8 * 0.8)) < 1e-5
    assert raster[0, 0:6, 2].min() > 0.0
    assert raster[0, 50:, 2].min() > 0.0
    assert raster[0, 6:50, 2].sum() == 0.0


def test_decode_uses_initial_contact_for_fully_open_segments():
    seq_len = 60
    endpoint_prob = torch.zeros(1, seq_len, 6)
    interior_prob = torch.full((1, seq_len, 3), 0.05)
    initial_contact = torch.tensor([[1.0, 0.0, 0.8]])

    segments, raster = decode_contact_segments(
        endpoint_prob,
        interior_prob,
        initial_contact=initial_contact,
        peak_threshold=0.3,
        segment_threshold=0.25,
        peak_radius=5,
    )

    assert [segment["channel_name"] for segment in segments[0]] == ["lhand", "obj"]
    for segment in segments[0]:
        assert segment["start"] == 0 and segment["end"] == seq_len - 1
        assert segment["start_confidence"] is None
        assert segment["end_confidence"] is None
        assert abs(
            segment["initial_confidence"]
            - float(initial_contact[0, segment["channel"]])
        ) < 1e-6
    assert torch.allclose(raster[0, :, 0], torch.ones(seq_len))
    assert raster[0, :, 1].sum() == 0.0
    assert torch.allclose(raster[0, :, 2], torch.full((seq_len,), 0.8))


def test_online_contact_init_uses_window_first_frame_raw_prediction():
    decoded = torch.zeros(1, 5, 3)
    raw = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3) / 15.0
    history = {
        "pred_hand_contact_prob": decoded,
        "pred_hand_contact_prob_frame": raw,
    }

    updated = update_data_inits_from_history(
        {"contact_init": torch.zeros(1, 3)},
        history,
        index=1,
        contact_index=2,
    )

    assert torch.equal(updated["contact_init"], raw[:, 2])


def test_decode_filters_low_confidence_segments():
    seq_len = 60
    endpoint_prob = torch.zeros(1, seq_len, 6)
    endpoint_prob[0, 10, 4] = 0.35
    endpoint_prob[0, 20, 5] = 0.4
    interior_prob = torch.zeros(1, seq_len, 3)
    interior_prob[0, 10:21, 2] = 0.05  # 段内逐帧概率极低 → 置信度不足

    segments, raster = decode_contact_segments(
        endpoint_prob, interior_prob,
        peak_threshold=0.3, segment_threshold=0.25, peak_radius=5,
    )

    assert segments[0] == []
    assert raster.sum() == 0.0


def test_decode_without_interior_prob_uses_endpoint_peaks_only():
    seq_len = 60
    endpoint_prob = torch.zeros(1, seq_len, 6)
    endpoint_prob[0, 10, 0] = 0.9   # lhand_start
    endpoint_prob[0, 30, 1] = 0.9   # lhand_end

    segments, _ = decode_contact_segments(
        endpoint_prob, None,
        peak_threshold=0.3, segment_threshold=0.25, peak_radius=5,
    )

    assert len(segments[0]) == 1
    assert segments[0][0]["channel_name"] == "lhand"
    assert abs(segments[0][0]["confidence"] - 0.9) < 1e-5


# ---------- 模块前向与损失 ----------

def _vc_cfg(**overrides):
    cfg = SimpleNamespace(
        imu_dim=9,
        obj_imu_dim=9,
        num_human_imus=6,
        velocity_hidden_dim=32,
        velocity_num_layers=1,
        velocity_dropout=0.0,
        boundary_group_hidden=16,
        boundary_hidden_dim=16,
        boundary_bidirectional=True,
        velocity_contact_use_geom_features=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _vc_inputs(batch_size=1, seq_len=8):
    torch.manual_seed(7)
    return {
        "human_imu": torch.randn(batch_size, seq_len, 6, 9),
        "obj_imu": torch.randn(batch_size, seq_len, 9),
        "hand_vel_glb_init": torch.zeros(batch_size, 2, 3),
        "obj_vel_init": torch.zeros(batch_size, 3),
        "contact_init": torch.zeros(batch_size, 3),
        "obj_trans_init": torch.zeros(batch_size, 3),
    }


def test_velocity_contact_forward_outputs_endpoint_heatmaps():
    model = VelocityContactModule(_vc_cfg()).eval()
    batch_size, seq_len = 1, 8

    with torch.no_grad():
        out = model(_vc_inputs(batch_size, seq_len))

    assert out["pred_contact_endpoint_logits"].shape == (batch_size, seq_len, 6)
    assert out["pred_contact_endpoint_prob"].shape == (batch_size, seq_len, 6)
    assert out["pred_contact_endpoint_prob"].min() >= 0.0
    assert out["pred_contact_endpoint_prob"].max() <= 1.0
    assert out["pred_contact_from_endpoints_prob"].shape == (
        batch_size,
        seq_len,
        3,
    )
    # 逐帧接触概率保持原接口
    assert out["pred_hand_contact_prob"].shape == (batch_size, seq_len, 3)
    assert not hasattr(model, "boundary_head")
    assert "pred_interaction_boundary_logits" not in out
    assert "pred_interaction_boundary_prob" not in out


def test_velocity_only_forward_skips_contact_branches():
    model = VelocityContactModule(_vc_cfg()).eval()

    with torch.no_grad():
        out = model(_vc_inputs(), velocity_only=True)

    assert set(out) == {"pred_hand_glb_vel", "pred_obj_vel"}


def test_endpoint_state_recovery_uses_last_active_frame_end_semantics():
    endpoint_prob = torch.zeros(1, 7, 6)
    endpoint_prob[0, 2, 4] = 1.0  # object start
    endpoint_prob[0, 4, 5] = 1.0  # object end (frame 4 remains active)

    recovered = recover_contact_state_from_endpoints(
        endpoint_prob,
        torch.zeros(1, 3),
    )

    assert torch.equal(
        recovered[0, :, 2],
        torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
    )


def test_endpoint_state_recovery_keeps_fully_open_initial_contact():
    endpoint_prob = torch.zeros(1, 8, 6)
    initial_contact = torch.tensor([[1.0, 0.0, 0.8]])

    recovered = recover_contact_state_from_endpoints(
        endpoint_prob,
        initial_contact,
    )

    assert torch.allclose(recovered[:, :, 0], torch.ones(1, 8))
    assert torch.allclose(recovered[:, :, 1], torch.zeros(1, 8))
    assert torch.allclose(recovered[:, :, 2], torch.full((1, 8), 0.8))


def test_velocity_contact_inference_contact_from_segments():
    cfg = _vc_cfg(contact_from_segments=True)
    model = VelocityContactModule(cfg).eval()
    batch_size, seq_len = 1, 8

    with torch.no_grad():
        out = model.inference(_vc_inputs(batch_size, seq_len), inference_mode="offline")

    assert "pred_contact_segments" in out
    assert isinstance(out["pred_contact_segments"], list)
    assert len(out["pred_contact_segments"]) == batch_size
    # 逐帧概率被段栅格化结果替换，原逐帧输出保留在 _frame
    assert out["pred_hand_contact_prob"].shape == (batch_size, seq_len, 3)
    assert out["pred_hand_contact_prob_frame"].shape == (batch_size, seq_len, 3)


def test_velocity_contact_inference_passes_frame_zero_contact_to_decoder(monkeypatch):
    cfg = _vc_cfg(contact_from_segments=True)
    model = VelocityContactModule(cfg).eval()
    inputs = _vc_inputs(batch_size=1, seq_len=8)
    inputs["contact_init"] = torch.tensor([[1.0, 0.0, 0.8]])

    def _fake_forward(data_dict, hp_out=None):
        del data_dict, hp_out
        return {
            "pred_contact_endpoint_prob": torch.zeros(1, 8, 6),
            "pred_hand_contact_prob": torch.full((1, 8, 3), 0.05),
        }

    monkeypatch.setattr(model, "forward", _fake_forward)
    out = model.inference(inputs, inference_mode="offline")

    assert torch.allclose(out["pred_hand_contact_prob"][0, :, 0], torch.ones(8))
    assert out["pred_hand_contact_prob"][0, :, 1].sum() == 0.0
    assert torch.allclose(
        out["pred_hand_contact_prob"][0, :, 2],
        torch.full((8,), 0.8),
    )


def test_velocity_contact_loss_includes_endpoint_terms():
    batch_size, seq_len = 2, 16
    loss_fn = VelocityContactLoss(weights={})
    endpoint_logits = torch.zeros(batch_size, seq_len, 6)
    endpoint_gauss = torch.zeros(batch_size, seq_len, 6)
    endpoint_gauss[:, 4, 4] = 1.0
    endpoint_gauss[:, 12, 5] = 1.0

    pred_dict = {"pred_contact_endpoint_logits": endpoint_logits}
    batch = {
        "human_imu": torch.randn(batch_size, seq_len, 6, 9),
        "contact_endpoint_gauss": endpoint_gauss,
        "has_object": torch.ones(batch_size, dtype=torch.bool),
    }

    total, losses, weighted = loss_fn(pred_dict, batch, torch.device("cpu"))

    assert torch.isfinite(total)
    assert losses["contact_endpoint"] > 0.0
    assert torch.isfinite(losses["contact_endpoint_order"])
    assert "contact_endpoint" in weighted
    assert "interaction_boundary" not in losses


def test_endpoint_contact_consistency_backpropagates_only_to_endpoint_state():
    batch_size, seq_len = 2, 8
    endpoint_logits = torch.zeros(
        batch_size,
        seq_len,
        6,
        requires_grad=True,
    )
    endpoint_prob = torch.sigmoid(endpoint_logits)
    endpoint_state = recover_contact_state_from_endpoints(
        endpoint_prob,
        torch.zeros(batch_size, 3),
    )
    frame_contact = torch.full(
        (batch_size, seq_len, 3),
        0.1,
        requires_grad=True,
    )
    loss_fn = VelocityContactLoss(
        weights={"endpoint_contact_consistency": 1.0}
    )
    batch = {
        "human_imu": torch.randn(batch_size, seq_len, 6, 9),
        "has_object": torch.ones(batch_size, dtype=torch.bool),
    }
    pred_dict = {
        "pred_contact_from_endpoints_prob": endpoint_state,
        "pred_hand_contact_prob": frame_contact,
    }

    total, _, weighted = loss_fn(
        pred_dict,
        batch,
        torch.device("cpu"),
        active_loss_keys={"endpoint_contact_consistency"},
    )
    total.backward()

    assert set(weighted) == {"endpoint_contact_consistency"}
    assert endpoint_logits.grad is not None
    assert torch.isfinite(endpoint_logits.grad).all()
    assert endpoint_logits.grad.abs().sum() > 0
    assert frame_contact.grad is None


def test_velocity_contact_stages_keep_one_optimizer_with_task_schedules():
    cfg = _vc_cfg(
        lr=0.002,
        weight_decay=1e-4,
        use_multi_gpu=False,
        gpus=[0],
        epochs=6,
        milestones=[],
        gamma=0.2,
        vc_staged_training=True,
        vc_velocity_stage_epochs=2,
        vc_joint_stage_epochs=2,
        vc_contact_lr_milestones=[],
        vc_velocity_lr_milestones=[],
        vc_endpoint_joint_lr_milestones=[],
        vc_endpoint_finetune_lr_scale=0.2,
        vc_endpoint_finetune_lr_milestones=[],
        vc_group_lr_gamma=0.2,
    )
    model = VelocityContactModule(cfg)
    optimizer, scheduler = build_velocity_contact_optimizer_and_scheduler(model, cfg)

    groups = {
        group["group_name"]: group
        for group in optimizer.param_groups
    }
    assert set(groups) == {"contact", "velocity", "endpoint"}
    optimized_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    trainable_ids = [
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert len(optimized_ids) == len(set(optimized_ids))
    assert set(optimized_ids) == set(trainable_ids)
    assert math.isclose(groups["velocity"]["lr"], 0.002)
    assert math.isclose(groups["contact"]["lr"], 0.0)
    assert math.isclose(groups["endpoint"]["lr"], 0.0)

    optimizer.step()
    scheduler.step()
    assert math.isclose(groups["velocity"]["lr"], 0.002)
    assert math.isclose(groups["contact"]["lr"], 0.0)
    assert math.isclose(groups["endpoint"]["lr"], 0.0)

    optimizer.step()
    scheduler.step()
    assert math.isclose(groups["velocity"]["lr"], 0.0)
    assert math.isclose(groups["contact"]["lr"], 0.002)
    assert math.isclose(groups["endpoint"]["lr"], 0.002)

    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    assert math.isclose(groups["velocity"]["lr"], 0.0)
    assert math.isclose(groups["contact"]["lr"], 0.0)
    assert math.isclose(groups["endpoint"]["lr"], 0.0004)


def test_velocity_contact_stage_freezing_matches_dependency_order():
    model = VelocityContactModule(_vc_cfg())
    trainer = object.__new__(VelocityContactTrainer)
    trainer.model = model
    trainer.current_stage = None

    expected_active_groups = {
        "velocity": {"velocity"},
        "joint": {"contact", "endpoint"},
        "endpoint": {"endpoint"},
    }
    for stage, expected_groups in expected_active_groups.items():
        trainer._configure_stage(stage)
        active_groups = {
            _vc_parameter_group(name)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        assert active_groups == expected_groups

    trainer._configure_stage("endpoint")
    assert not model.vel_net.training
    assert not model.boundary_rnn.training
    assert not model.hand_contact_net.training
    assert model.contact_endpoint_rnn.training
