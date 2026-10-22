"""Saved-split identity, seed, and exact current-recipe preservation."""
import copy
import json
from pathlib import Path

import pytest
import torch

from scripts import run_omomo_egoimu_split as runner


def test_saved_training_seed_is_distinct_from_default_seed(tmp_path):
    path = tmp_path / "training.yaml"
    path.write_text("noise: 0.1\nseed: 10\n")
    assert runner.reference_seed(path) == 10
    path.write_text("seed: 42\nseed: 10\n")
    with pytest.raises(ValueError):
        runner.reference_seed(path)


def test_split_rejects_overlap_and_path_traversal():
    with pytest.raises(ValueError, match="Duplicate"):
        runner.validate_split({"train_files": ["1.pt"], "test_files": ["1.pt"], "total_files": 2})
    with pytest.raises(ValueError, match="Invalid"):
        runner.validate_split({"train_files": ["../1.pt"], "test_files": [], "total_files": 1})


def test_recipe_only_changes_data_paths_seed_and_evaluation_policy():
    previous_path = runner.ROOT / "outputs/IMUHOI++/paper_noise_omomo_20261016/recipe.yaml"
    import yaml
    previous = yaml.safe_load(previous_path.read_text())
    before = copy.deepcopy(previous)
    recipe = runner.training_recipe(previous, runner.ROOT / "outputs/test_restored/data", 10)
    assert previous == before
    allowed = {"datasets", "seed", "simulate_imu_noise_seed", "eval_hoi_protocol"}
    changes = {key for key in set(previous) | set(recipe) if previous.get(key) != recipe.get(key)}
    assert changes == allowed
    assert recipe["seed"] == 10
    assert recipe["eval_hoi_protocol"] == "egoimu"
    assert recipe["pretrained_modules"] == {"human_pose": None, "velocity_contact": None, "object_trans": None}
    assert recipe["epochs"] == previous["epochs"] == 300
    assert recipe["joint_train_epochs"] == previous["joint_train_epochs"] == 100


def _sample(name):
    return {"seq_name": name, "obj_trans": torch.zeros(62, 3),
            "canonical_body_model_gender": "neutral", "canonical_betas": "zero"}


def test_restore_reads_saved_membership_and_copies_current_bytes(tmp_path):
    old, unified, output = tmp_path / "old", tmp_path / "unified", tmp_path / "output"
    unified.mkdir()
    for split, name in (("train", "1.pt"), ("test", "0.pt")):
        (old / split).mkdir(parents=True)
        torch.save(_sample(name), old / split / name)
        sample = _sample(name)
        sample["new_contact_label"] = True
        torch.save(sample, unified / name)
    info = {"train_files": ["1.pt"], "test_files": ["0.pt"], "total_files": 2}
    inventory = runner.materialize_split(unified, old, info, output)
    assert inventory["train"][0]["name"] == "1.pt"
    assert inventory["test"][0]["name"] == "0.pt"
    assert (output / "train/1.pt").read_bytes() == (unified / "1.pt").read_bytes()
    assert not (output / "test/1.pt").exists()
    assert json.loads((output / "inventory.json").read_text())["test"][0]["seq_name"] == "0.pt"
    # Copies are independent: rewriting output cannot affect the source.
    (output / "train/1.pt").write_bytes(b"changed")
    assert (unified / "1.pt").read_bytes() != b"changed"


def test_restore_rejects_matching_filename_with_different_identity(tmp_path):
    old, unified = tmp_path / "old", tmp_path / "unified"
    (old / "train").mkdir(parents=True)
    unified.mkdir()
    torch.save(_sample("old_identity"), old / "train/0.pt")
    torch.save(_sample("different_identity"), unified / "0.pt")
    with pytest.raises(ValueError, match="identity"):
        runner.materialize_split(unified, old, {"train_files": ["0.pt"], "test_files": [], "total_files": 1}, tmp_path / "out")
