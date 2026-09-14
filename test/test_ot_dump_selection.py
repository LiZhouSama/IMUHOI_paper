"""Pure path/selection tests: no torch, datasets, checkpoints or GPU required."""
import json
from pathlib import Path

import pytest

from utils.ot_dump_selection import (
    read_sequence_list,
    resolve_dump_data_path,
    select_sequence_paths,
)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"full_sequence_test_path": "original/test", "test_path": "seg/test"}, "original/test"),
        ({"test_path": "split/test", "data_dir": "old/test"}, "split/test"),
        ({"data_dir": "old/test"}, "old/test"),
        ({"full_sequence_test_path": "", "test_path": "split/test"}, "split/test"),
    ],
)
def test_default_test_path_matches_existing_precedence(tmp_path, config, expected):
    assert resolve_dump_data_path({"omomo": config}, "omomo", tmp_path) == tmp_path / expected


def test_omomo_train_prefers_full_corpus_without_changing_default_test(tmp_path):
    configs = {
        "omomo": {"train_path": "small/train", "test_path": "small/test"},
        "omomo_full": {"train_path": "full/train", "test_path": "full/test"},
    }
    assert resolve_dump_data_path(configs, "omomo", tmp_path, split="train") == tmp_path / "full/train"
    assert resolve_dump_data_path(configs, "omomo", tmp_path) == tmp_path / "small/test"
    assert resolve_dump_data_path(configs, "omomo_full", tmp_path) == tmp_path / "full/test"


@pytest.mark.parametrize("dataset", ["omomo", "omomo_full"])
def test_original_train_is_derived_from_declared_full_test_before_segmented_path(tmp_path, dataset):
    config = {
        "train_path": "process/processed_seg_data_OMOMO_full/train",
        "test_path": "process/processed_seg_data_OMOMO_full/test",
        "full_sequence_test_path": "process/processed_split_data_OMOMO_full/test",
    }
    assert resolve_dump_data_path({dataset: config}, dataset, tmp_path, split="train") == (
        tmp_path / "process/processed_split_data_OMOMO_full/train"
    )


def test_explicit_full_train_key_wins(tmp_path):
    config = {
        "full_sequence_train_path": "original/train",
        "full_sequence_test_path": "different/test",
        "train_path": "process/processed_seg_data_OMOMO/train",
    }
    assert resolve_dump_data_path({"omomo": config}, "omomo", tmp_path, split="train") == (
        tmp_path / "original/train"
    )


def test_train_without_full_corpus_uses_only_train_key(tmp_path):
    config = {"train_path": "train", "test_path": "test", "data_dir": "test"}
    assert resolve_dump_data_path({"omomo": config}, "omomo", tmp_path, split="train") == tmp_path / "train"


@pytest.mark.parametrize("config", [{}, {"test_path": "test"}, {"data_dir": "test"}])
def test_train_never_falls_back_to_test_or_data_dir(tmp_path, config):
    with pytest.raises(ValueError, match="train path"):
        resolve_dump_data_path({"omomo": config}, "omomo", tmp_path, split="train")


def test_segmented_train_requires_full_path_or_override(tmp_path):
    configs = {"omomo": {"train_path": "process/processed_seg_data_OMOMO/train"}}
    with pytest.raises(ValueError, match="Segmented"):
        resolve_dump_data_path(configs, "omomo", tmp_path, split="train")
    assert resolve_dump_data_path(
        configs, "omomo", tmp_path, split="train", data_path="full/train",
    ) == tmp_path / "full/train"


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("absolute", [False, True])
def test_explicit_data_path_overrides_config(tmp_path, split, absolute):
    configs = {"omomo": {"train_path": "train", "test_path": "test"}}
    override = tmp_path / "custom" if absolute else "custom"
    assert resolve_dump_data_path(
        configs, "omomo", tmp_path, split=split, data_path=override,
    ) == tmp_path / "custom"


@pytest.mark.parametrize("override", ["test", "train/../test", ".", "test/subdir"])
def test_training_override_cannot_overlap_configured_test(tmp_path, override):
    configs = {"omomo": {"train_path": "train", "test_path": "test"}}
    with pytest.raises(ValueError, match="overlaps configured test"):
        resolve_dump_data_path(configs, "omomo", tmp_path, split="train", data_path=override)


def test_training_rejects_test_paths_from_full_corpus_alias(tmp_path):
    configs = {
        "omomo": {"train_path": "train", "test_path": "test"},
        "omomo_full": {"train_path": "full/train", "test_path": "full/test"},
    }
    with pytest.raises(ValueError, match="overlaps configured test"):
        resolve_dump_data_path(configs, "omomo", tmp_path, split="train", data_path="full/test")


def test_path_validation_errors(tmp_path):
    configs = {"omomo": {}}
    with pytest.raises(ValueError, match="not configured"):
        resolve_dump_data_path(configs, "omomo", tmp_path)
    with pytest.raises(ValueError, match="Unsupported split"):
        resolve_dump_data_path(configs, "omomo", tmp_path, split="validation")
    with pytest.raises(ValueError, match="Unknown dataset"):
        resolve_dump_data_path(configs, "missing", tmp_path)


@pytest.fixture
def split_dirs(tmp_path):
    train = tmp_path / "train"
    test = tmp_path / "test"
    train.mkdir()
    test.mkdir()
    # Deliberately create files in non-sorted order; never deserialize them.
    for name in ("c.pt", "a.pt", "b.pt"):
        (train / name).touch()
    (test / "a.pt").touch()
    return train, test


def test_directory_selection_is_sorted_before_limit(split_dirs):
    train, _ = split_dirs
    expected = [train / "a.pt", train / "b.pt"]
    assert select_sequence_paths(train, limit=2) == expected
    assert select_sequence_paths(train, limit=2) == expected


def test_text_list_order_is_kept_and_selected_before_limit(tmp_path, split_dirs):
    train, _ = split_dirs
    manifest = tmp_path / "selection.txt"
    manifest.write_text("\ufeff# Holdout from TRAIN, not test\n\nc.pt\na.pt\nb.pt\n", encoding="utf-8")
    assert select_sequence_paths(train, manifest, limit=1) == [train / "c.pt"]
    assert select_sequence_paths(train, manifest) == [train / name for name in ("c.pt", "a.pt", "b.pt")]


def test_json_list_allows_absolute_paths_inside_split(tmp_path, split_dirs):
    train, _ = split_dirs
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps([str(train / "b.pt"), "a.pt"]), encoding="utf-8")
    assert select_sequence_paths(train, manifest) == [train / "b.pt", train / "a.pt"]


@pytest.mark.parametrize("outside", ["../test/a.pt", "absolute_test"])
def test_list_cannot_leak_test_even_after_limit(tmp_path, split_dirs, outside):
    train, test = split_dirs
    entry = str(test / "a.pt") if outside == "absolute_test" else outside
    manifest = tmp_path / "selection.txt"
    manifest.write_text(f"c.pt\n{entry}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside selected split"):
        select_sequence_paths(train, manifest, limit=1)


@pytest.mark.parametrize("second", ["a.pt", "./a.pt"])
def test_duplicate_paths_are_rejected_even_after_limit(tmp_path, split_dirs, second):
    train, _ = split_dirs
    manifest = tmp_path / "selection.txt"
    manifest.write_text(f"a.pt\n{second}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate sequence"):
        select_sequence_paths(train, manifest, limit=1)


@pytest.mark.parametrize("entry", ["missing.pt", "not_tensor.txt", "directory.pt"])
def test_bad_list_entry_is_not_silently_skipped(tmp_path, split_dirs, entry):
    train, _ = split_dirs
    (train / "not_tensor.txt").touch()
    (train / "directory.pt").mkdir()
    manifest = tmp_path / "selection.txt"
    manifest.write_text(entry, encoding="utf-8")
    with pytest.raises(ValueError, match="existing .pt file"):
        select_sequence_paths(train, manifest)


@pytest.mark.parametrize("content", ["[]", '{}', '["a.pt", 12]', '[""]'])
def test_invalid_json_lists(tmp_path, content):
    manifest = tmp_path / "selection.json"
    manifest.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        read_sequence_list(manifest)


def test_empty_text_list_and_empty_directory_are_errors(tmp_path):
    manifest = tmp_path / "selection.txt"
    manifest.write_text("# comment only\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        read_sequence_list(manifest)
    with pytest.raises(ValueError, match="No .pt sequences"):
        select_sequence_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        select_sequence_paths(tmp_path / "missing")


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_is_rejected(split_dirs, limit):
    with pytest.raises(ValueError, match="positive integer"):
        select_sequence_paths(split_dirs[0], limit=limit)


def test_symlinks_cannot_bypass_split_validation(tmp_path, split_dirs):
    train, test = split_dirs
    try:
        (train / "leak.pt").symlink_to(test / "a.pt")
        (tmp_path / "test_alias").symlink_to(test, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable on this filesystem")
    with pytest.raises(ValueError, match="outside selected split"):
        select_sequence_paths(train)
    with pytest.raises(ValueError, match="overlaps configured test"):
        resolve_dump_data_path(
            {"omomo": {"test_path": "test"}}, "omomo", tmp_path,
            split="train", data_path="test_alias",
        )


def test_disjoint_training_lists_do_not_use_test(tmp_path, split_dirs):
    train, _ = split_dirs
    train_list = tmp_path / "pilot.txt"
    holdout_list = tmp_path / "holdout.txt"
    train_list.write_text("c.pt\na.pt\n", encoding="utf-8")
    holdout_list.write_text("b.pt\n", encoding="utf-8")
    train_paths = set(select_sequence_paths(train, train_list))
    holdout_paths = set(select_sequence_paths(train, holdout_list))
    assert train_paths.isdisjoint(holdout_paths)
    assert all(path.parent == train for path in train_paths | holdout_paths)


@pytest.fixture
def dump_module():
    # Imported only for these tiny CPU tensor tests; no model/data is loaded.
    import scripts.dump_ot_predictions as dump
    return dump


def test_record_keeps_all_old_tensors_and_adds_gt_free_published_track(dump_module):
    torch = dump_module.torch
    length = 3
    sample = {key: torch.zeros(length, 3) for key in dump_module.GT_KEYS}
    sample.update({
        "human_imu": torch.zeros(length, 6, 9),
        "pose": torch.zeros(length, 156),
        "position_global": torch.full((length, 24, 3), 100.0),
        "seq_file": "example.pt",
    })
    pred = {key: torch.zeros(1, length, 3) for key in dump_module.PRED_KEYS}
    pred.update({
        "pred_obj_trans": torch.full((1, length, 3), -9.0),
        "stage1_pred_palm_glb_pos": torch.zeros(1, length, 2, 3),
        "pred_palm_glb_pos": torch.zeros(1, length, 2, 3),
        "gating_weights": torch.tensor([[[0.5, 0.0, 0.25, 0.25]]] * length).transpose(0, 1),
    })
    pred["pred_palm_glb_pos"][0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    pred["pred_lhand_obj_trans"][0, :, 0] = 10
    pred["pred_imu_obj_trans"][0, :, 0] = 2
    record = dump_module._record_from(sample, pred, length)
    assert set(dump_module.PRED_KEYS).issubset(record)
    assert {f"gt_{key}" for key in dump_module.GT_KEYS}.issubset(record)
    for key in dump_module.PRED_KEYS:
        assert torch.equal(record[key], pred[key][0])
    assert torch.equal(record["pred_obj_trans_published"][:, 0], torch.tensor([6.0, 9.5, 11.75]))
    assert torch.count_nonzero(record["pred_obj_trans_published"][:, 1:]) == 0
    assert torch.equal(record["gt_palm_pos"], torch.full((length, 2, 3), 100.0))
    assert torch.equal(record["gt_palm_pos_eval"], record["gt_palm_pos"])
    # Changing supervision must not change any published prediction.
    sample["obj_trans"] += 1000
    sample["position_global"] += 1000
    sample["pose"] += 1
    again = dump_module._record_from(sample, pred, length)
    assert torch.equal(record["pred_obj_trans_published"], again["pred_obj_trans_published"])
    for value in record.values():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu" and not value.requires_grad


def test_record_without_refine_or_with_empty_publish_keeps_raw(dump_module):
    torch = dump_module.torch
    sample = {"human_imu": torch.zeros(2, 6, 9)}
    pred = {"pred_obj_trans": torch.ones(1, 2, 3)}
    record = dump_module._record_from(sample, pred, 2)
    assert torch.equal(record["pred_obj_trans_published"], pred["pred_obj_trans"][0])
    empty = dump_module._record_from(sample, {"pred_obj_trans": torch.ones(1, 0, 3)}, 0)
    assert tuple(empty["pred_obj_trans_published"].shape) == (0, 3)
