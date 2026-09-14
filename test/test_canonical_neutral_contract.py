"""Contracts for the neutral, zero-betas canonical skeleton migration."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml
from easydict import EasyDict

import eval_IMUHOI
import eval_batch_online
from dataset.dataset_IMUHOI import CanonicalSkeletonMismatchError, IMUDataset
from utils.dataset_config import configure_canonical_body_model_for_datasets
from utils.human_pose import (
    CANONICAL_BETAS,
    CANONICAL_BODY_MODEL_GENDER,
    CANONICAL_SKELETON_VERSION,
    canonical_skeleton_metadata,
    is_neutral_body_model_path,
    load_body_models_from_cfg,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "IMUHOI_train_rnn.yaml",
    "IMUHOI_train_dit.yaml",
    "IMUHOI_train_mamba.yaml",
    "IMUHOI_train_mamba_simple.yaml",
    "IMUHOI_fineTuning.yaml",
)


def test_all_model_configs_use_neutral_zero_shape_and_disable_legacy_weights():
    for name in CONFIGS:
        config = yaml.safe_load((ROOT / "configs" / name).read_text())
        assert config["canonical_body_model_gender"] == "neutral"
        assert config["canonical_betas"] == "zero"
        assert config["canonical_skeleton_version"] == 1
        assert config["require_canonical_skeleton"] is True
        for dataset_name, dataset_config in config["datasets"].items():
            assert dataset_config.get("body_model_type") in {"smplh", "smplx"}, (
                name, dataset_name, dataset_config
            )
        for path in config["body_model_paths"].values():
            assert is_neutral_body_model_path(path), (name, path)
        assert is_neutral_body_model_path(config["body_model_path"])
        for value in (config.get("pretrained_modules") or {}).values():
            assert value is None, (name, value)


def test_canonical_metadata_is_explicit_and_family_specific():
    for family in ("smplh", "smplx"):
        metadata = canonical_skeleton_metadata(family)
        assert metadata == {
            "body_model_type": family,
            "gender": CANONICAL_BODY_MODEL_GENDER,
            "canonical_body_model_gender": CANONICAL_BODY_MODEL_GENDER,
            "canonical_betas": CANONICAL_BETAS,
            "canonical_skeleton_version": CANONICAL_SKELETON_VERSION,
        }


def test_train_model_loader_rejects_non_neutral_paths_before_loading():
    config = EasyDict(
        body_model_paths={"smplx": "models/smplx/SMPLX_MALE.npz"},
        body_model_path="models/smplx/SMPLX_MALE.npz",
        canonical_body_model_gender="neutral",
        canonical_betas="zero",
    )
    with pytest.raises(ValueError, match="neutral"):
        load_body_models_from_cfg(config)


def _write_minimal_sequence(path: Path, *, canonical: bool):
    data = {
        "position_global_full_gt_world": torch.zeros(3, 24, 3),
        "pose": torch.zeros(3, 72),
        "has_object": False,
        "body_model_type": "smplx",
        "gender": "neutral" if canonical else "female",
    }
    if canonical:
        data.update(
            canonical_body_model_gender="neutral",
            canonical_betas="zero",
            canonical_skeleton_version=1,
        )
    torch.save(data, path)


def test_dataset_refuses_legacy_shaped_data_when_canonical_is_required(tmp_path):
    _write_minimal_sequence(tmp_path / "legacy.pt", canonical=False)
    with pytest.raises(CanonicalSkeletonMismatchError, match="Re-run preprocessing"):
        IMUDataset(str(tmp_path), full_sequence=True, require_canonical_skeleton=True)


def test_dataset_accepts_explicit_neutral_canonical_metadata(tmp_path):
    _write_minimal_sequence(tmp_path / "canonical.pt", canonical=True)
    dataset = IMUDataset(str(tmp_path), full_sequence=True, require_canonical_skeleton=True)
    assert len(dataset) == 1
    info = dataset.sequence_info[0]
    assert info["canonical_body_model_gender"] == "neutral"
    assert info["canonical_betas"] == "zero"
    assert info["canonical_skeleton_version"] == 1


def test_evaluators_use_model_and_dataset_joints_without_eval_body_model():
    signature = inspect.signature(eval_IMUHOI.evaluate_model)
    assert "smpl_model" not in signature.parameters
    source = inspect.getsource(eval_IMUHOI.evaluate_model)
    assert 'batch_device.get("position_global")' in source
    assert 'pred_dict.get("pred_joints_global")' in source
    assert "smpl_model(" not in source
    batch_source = inspect.getsource(eval_batch_online.main)
    assert "load_smpl_model" not in batch_source
    assert "Joint metrics: model pred_joints_global" in batch_source

@pytest.mark.parametrize(
    ("dataset_name", "family", "path_fragment"),
    [
        ("omomo", "smplx", "SMPLX_NEUTRAL.npz"),
        ("pahoi", "smplx", "SMPLX_NEUTRAL.npz"),
        ("imhd", "smplh", "smplh/neutral/model.npz"),
        ("behave", "smplh", "smplh/neutral/model.npz"),
        ("hodome", "smplh", "smplh/neutral/model.npz"),
    ],
)
def test_training_selects_the_dataset_family_neutral_model(dataset_name, family, path_fragment):
    config = EasyDict(yaml.safe_load((ROOT / "configs/IMUHOI_train_rnn.yaml").read_text()))
    config.model_arch = "dit"
    path, families = configure_canonical_body_model_for_datasets(config, [dataset_name])
    assert families == (family,)
    assert path_fragment in path.replace("\\", "/")
    assert config.body_model_path == path


def test_non_rnn_mixed_family_training_is_rejected_instead_of_silently_misrouting():
    config = EasyDict(yaml.safe_load((ROOT / "configs/IMUHOI_train_rnn.yaml").read_text()))
    config.model_arch = "dit"
    with pytest.raises(ValueError, match="mixed canonical families"):
        configure_canonical_body_model_for_datasets(config, ["omomo", "imhd"])


def test_rnn_mixed_family_training_uses_its_per_sample_model_map():
    config = EasyDict(yaml.safe_load((ROOT / "configs/IMUHOI_train_rnn.yaml").read_text()))
    config.model_arch = "rnn"
    path, families = configure_canonical_body_model_for_datasets(config, ["omomo", "imhd"])
    assert path is None
    assert families == ("smplh", "smplx")