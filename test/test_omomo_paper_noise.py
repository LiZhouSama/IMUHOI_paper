"""Noise-protocol and queue gates; no dataset or GPU required."""
from types import SimpleNamespace
import json

import pytest
import torch

from scripts.run_behave_full_pipeline_trust import configure_noise_profile
from scripts.run_omomo_paper_noise import numerical_gate, evaluation_path, read_evaluation, offline_command, VARIANTS
from utils.utils import _add_split_imu_noise, apply_eval_imu_noise_to_sequence


def test_original_profile_disables_sequence_noise_and_uses_all_nine_channels():
    cfg = {"simulate_imu_noise": True, "imu_rot_noise_std": 0.005}
    train_args, eval_args = configure_noise_profile(cfg, "egoimu_gaussian", 42)
    assert train_args == ["--no_simulate_imu_noise"]
    assert eval_args == []
    assert cfg["simulate_imu_noise"] is False
    assert cfg["validation_add_imu_noise"] is True
    imu = torch.zeros(10000, 9)
    noisy = _add_split_imu_noise(imu, torch.tensor(False), cfg["imu_acc_noise_std"],
                                 cfg["imu_rot_noise_std"], generator=torch.Generator().manual_seed(42))
    assert torch.all((noisy.std(dim=0) - 0.1).abs() < 0.004)
    assert torch.equal(imu, torch.zeros_like(imu))
    clean = _add_split_imu_noise(imu, torch.tensor(False), 0.1, 0.1, add_noise=False)
    assert torch.equal(clean, imu)


def test_current_runner_noise_policy_is_unchanged():
    cfg = {"simulate_imu_noise": True, "imu_rot_noise_std": 0.005}
    before = dict(cfg)
    train, evaluate = configure_noise_profile(cfg, "current", 7)
    assert cfg == before
    assert train == ["--simulate_imu_noise"]
    assert evaluate == ["--simulate_imu_noise", "--simulate_imu_noise_seed", "7"]


def test_previously_noised_sequences_are_not_perturbed_again():
    cfg = {}
    configure_noise_profile(cfg, "egoimu_gaussian", 42)
    imu = torch.zeros(1, 100, 6, 9)
    sample = {"human_imu": imu, "imu_noise_applied": True}
    result = apply_eval_imu_noise_to_sequence(sample, SimpleNamespace(**cfg), seed=42)
    assert torch.equal(result["human_imu"], imu)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.1, True, 11.01])
def test_gate_never_passes_missing_invalid_or_outside_budget_metrics(bad):
    result = numerical_gate({"obj": 1.0, "hoi": bad}, {"obj": 10.0, "hoi": 10.0}, 0.1)
    assert not result["passed"]


def test_gate_requires_every_metric_and_accepts_better_results():
    assert numerical_gate({"obj": 8.0, "hoi": 11.0}, {"obj": 10.0, "hoi": 10.0}, 0.1)["passed"]
    assert not numerical_gate({"obj": 8.0}, {"obj": 10.0, "hoi": 10.0}, 0.1)["passed"]


def test_offline_gate_does_not_use_better_online_numbers(tmp_path):
    for mode, value, inference in (("online", 1.0, "batch_online"), ("offline", 30.0, "offline")):
        path = evaluation_path(tmp_path, "baseline", mode, "final")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"inference_mode": inference, "results": {"omomo": {"obj": value}}}))
    offline = read_evaluation(tmp_path, "baseline", "offline", "final")
    assert not numerical_gate(offline, {"obj": 10.0}, 0.1)["passed"]
    assert read_evaluation(tmp_path, "baseline", "online", "final")["obj"] == 1.0


def test_mislabeled_offline_output_is_rejected(tmp_path):
    path = evaluation_path(tmp_path, "baseline", "offline", "final")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"inference_mode": "batch_online", "results": {"omomo": {"obj": 1.0}}}))
    with pytest.raises(ValueError, match="Wrong evaluation mode"):
        read_evaluation(tmp_path, "baseline", "offline", "final")


@pytest.mark.parametrize("method,tuned", [("baseline", "human_pose"), ("proposed", "velocity_contact")])
def test_offline_uses_matching_joint_weights_and_clean_full_sequence_eval(tmp_path, method, tuned):
    stage = tmp_path / method / "joint"
    stage.mkdir(parents=True)
    initial = {}
    for name in ("human_pose", "velocity_contact"):
        path = tmp_path / method / f"initial_{name}.pt"
        path.touch()
        initial[name] = {"checkpoint": str(path)}
    for name in (tuned, "object_trans"):
        (stage / f"final_{name}.pt").touch()
    manifest = {"stages": initial, "pipelines": {VARIANTS[method][1]: {
        "stage_dir": str(stage), "config": "frozen_config.yaml", "object_trans_arch": "imuhoi" if method == "baseline" else "v2"}}}
    (tmp_path / method / "experiment_manifest.json").write_text(json.dumps(manifest))
    cmd = offline_command(tmp_path, method, {"num_workers": 0, "seed": 42}, "final")
    assert cmd[cmd.index("--inference_mode") + 1] == "offline"
    tuned_flag = "--hp_ckpt" if tuned == "human_pose" else "--vc_ckpt"
    assert cmd[cmd.index(tuned_flag) + 1] == str(stage / f"final_{tuned}.pt")
    assert "--simulate_imu_noise" not in cmd
    assert "--no_trans" not in cmd
    assert cmd[cmd.index("--output_json") + 1].endswith("eval_omomo_offline_final.json")
