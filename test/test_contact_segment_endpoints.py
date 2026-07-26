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
)
from train.rnn.loss.velocity_contact_loss import VelocityContactLoss


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
    assert segment["start_confidence"] == 0.9
    assert segment["end_confidence"] == 0.8
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
    # 逐帧接触概率保持原接口
    assert out["pred_hand_contact_prob"].shape == (batch_size, seq_len, 3)


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
