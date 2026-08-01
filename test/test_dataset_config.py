from pathlib import Path

import yaml

from utils.dataset_config import get_full_sequence_test_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_full_sequence_path_prefers_original_split():
    dataset_config = {
        "test_path": "process/processed_seg_data_BEHAVE/test",
        "full_sequence_test_path": "process/processed_split_data_BEHAVE/test",
    }

    assert get_full_sequence_test_path(dataset_config) == (
        "process/processed_split_data_BEHAVE/test"
    )


def test_full_sequence_path_falls_back_for_omomo():
    dataset_config = {
        "test_path": "process/processed_split_data_OMOMO_bps_fixed/test",
    }

    assert get_full_sequence_test_path(dataset_config) == dataset_config["test_path"]


def test_rnn_non_omomo_paths_match_generated_directories():
    with (PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml").open() as config_file:
        datasets = yaml.safe_load(config_file)["datasets"]

    for name in ("imhd", "behave", "pahoi", "hodome"):
        dataset_config = datasets[name]
        assert "_bps" not in dataset_config["train_path"]
        assert "processed_seg_data_" in dataset_config["train_path"]
        assert "processed_seg_data_" in dataset_config["test_path"]
        assert "processed_split_data_" in dataset_config["full_sequence_test_path"]


def test_all_model_configs_define_full_sequence_paths_for_segmented_datasets():
    config_names = (
        "IMUHOI_train_rnn.yaml",
        "IMUHOI_train_dit.yaml",
        "IMUHOI_train_mamba.yaml",
        "IMUHOI_train_mamba_simple.yaml",
        "IMUHOI_fineTuning.yaml",
    )

    for config_name in config_names:
        with (PROJECT_ROOT / "configs" / config_name).open() as config_file:
            datasets = yaml.safe_load(config_file)["datasets"]
        for dataset_config in datasets.values():
            if "processed_seg_data_" not in dataset_config.get("test_path", ""):
                continue
            assert "processed_split_data_" in dataset_config["full_sequence_test_path"]
