"""Focused tests for the standalone mesh-aware l_b VQ and latent diffusion."""
from types import SimpleNamespace

import torch

from configs import _REDUCED_POSE_NAMES
from model.rnn.lb_vq_diffusion import (
    LBLatentDiffusion,
    LBVQAutoEncoder,
    build_lb_observation_sequence,
    build_lb_privileged_sequence,
    compute_lb_quantile_targets,
    masked_lb_quantile_loss,
)


def _cfg():
    prior = SimpleNamespace(
        max_seq_len=32,
        bps_num_basis=16,
        bps_radius=1.0,
        mesh_hidden_dim=24,
        mesh_feature_dim=12,
        temporal_downsample=4,
        latent_dim=8,
        num_codes=16,
        vq_model_dim=32,
        vq_nhead=4,
        vq_encoder_layers=1,
        vq_decoder_layers=1,
        vq_dropout=0.0,
        commitment_beta=0.25,
        soft_assignment_temperature=0.1,
        observation_encoder_layers=1,
        diffusion_cond_dim=32,
        diffusion_model_dim=32,
        diffusion_nhead=4,
        diffusion_layers=1,
        diffusion_dropout=0.0,
        diffusion_cond_dropout=0.0,
        diffusion_timesteps=20,
        diffusion_prediction_type="x0",
    )
    return SimpleNamespace(obj_imu_dim=9, lb_vq_prior=prior)


def _streams(batch_size=2, seq_len=17):
    pose_dim = len(_REDUCED_POSE_NAMES) * 6
    root = torch.randn(batch_size, seq_len, 3)
    hp_out = {
        "p_pred": torch.randn(batch_size, seq_len, pose_dim),
        "root_trans_pred": root,
        "pred_palm_glb_pos": root.unsqueeze(2) + torch.randn(batch_size, seq_len, 2, 3) * 0.2,
    }
    vc_out = {"pred_hand_contact_prob": torch.sigmoid(torch.randn(batch_size, seq_len, 3))}
    obj_imu = torch.randn(batch_size, seq_len, 9)
    obj_rot = torch.randn(batch_size, seq_len, 6)
    obj_trans = root + torch.randn(batch_size, seq_len, 3) * 0.4
    obj_scale = torch.ones(batch_size, seq_len)
    observation = build_lb_observation_sequence(hp_out, vc_out, obj_imu)
    privileged = build_lb_privileged_sequence(hp_out, vc_out, obj_rot, obj_trans, obj_scale)
    return observation, privileged


def test_lb_quantiles_ignore_noncontact_zeros_and_mask_missing_hand():
    left = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.0]])
    right = torch.tensor([[0.0, 4.0, 0.0, 0.0, 0.0]])
    left_contact = torch.tensor([[False, True, True, True, False]])
    right_contact = torch.tensor([[False, True, False, False, False]])

    target, valid, counts = compute_lb_quantile_targets(
        left,
        right,
        left_contact,
        right_contact,
        quantiles=(0.0, 0.5, 1.0),
        min_contact_frames=2,
    )

    assert torch.allclose(target[0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert valid.tolist() == [[True, False]]
    assert counts.tolist() == [[3, 1]]


def test_vq_outputs_low_rate_codes_and_ordered_hand_quantiles():
    torch.manual_seed(3)
    cfg = _cfg()
    observation, privileged = _streams()
    mesh = torch.randn(2, 32, 3) * 0.3
    model = LBVQAutoEncoder(cfg)

    output = model(privileged, mesh)

    assert observation.shape[-1] == model.privileged_dim - 1
    assert output["z_q"].shape == (2, 5, 8)
    assert output["code_indices"].shape == (2, 5)
    assert output["lb_quantiles"].shape == (2, 2, 3)
    assert torch.all(output["lb_quantiles"][..., 0] <= output["lb_quantiles"][..., 1])
    assert torch.all(output["lb_quantiles"][..., 1] <= output["lb_quantiles"][..., 2])
    assert 0.0 <= float(output["active_code_fraction"].item()) <= 1.0
    assert torch.isfinite(output["perplexity"])

    target = torch.full((2, 2, 3), 0.4)
    valid = torch.tensor([[True, True], [True, False]])
    loss = masked_lb_quantile_loss(output["lb_quantiles"], target, valid)
    loss.backward()
    assert model.quantile_decoder.output.weight.grad is not None


def test_observation_diffusion_matches_vq_latent_shape_and_decodes_samples():
    torch.manual_seed(4)
    cfg = _cfg()
    observation, privileged = _streams()
    mesh = torch.randn(2, 32, 3) * 0.3
    vq = LBVQAutoEncoder(cfg).eval()
    diffusion = LBLatentDiffusion(cfg)

    with torch.no_grad():
        target_latents = vq.encode(privileged, mesh)["z_q"]
    output = diffusion(observation, target_latents)
    assert output["x0_pred"].shape == target_latents.shape

    sampled = diffusion.sample(
        observation,
        steps=4,
        eta=0.0,
        initial_noise=torch.zeros_like(target_latents),
    )["latents"]
    decoded = vq.decode_latents(sampled, project_to_codebook=True)
    assert sampled.shape == (2, 5, 8)
    assert decoded["lb_quantiles"].shape == (2, 2, 3)
    assert decoded["code_indices"].shape == (2, 5)
