import numpy as np
import torch

from process.seg_sample import (
    downsample_sequence,
    find_segment_unheld_object_motion,
    process_pt_file,
    split_sequence,
)


def _sample_sequence(length=12):
    return {
        "seq_name": "sample",
        "position_global_full_gt_world": torch.zeros(length, 22, 3),
        "lhand_contact": torch.zeros(length, dtype=torch.bool),
        "rhand_contact": torch.zeros(length, dtype=torch.bool),
        "lhand_contact_raw": torch.ones(length, dtype=torch.bool),
        "obj_trans": torch.zeros(length, 3),
        "has_object": True,
        "source_frame_indices": torch.arange(length),
        "human_imu_real": torch.zeros(length, 6, 9),
        "array_stream": np.zeros((length, 3), dtype=np.float32),
        "obj_points_canonical": torch.zeros(256, 3),
        "metadata": "kept",
    }


def test_split_sequence_slices_every_matching_temporal_value(monkeypatch):
    monkeypatch.setattr("random.randint", lambda _minimum, _maximum: 6)

    segments = split_sequence(_sample_sequence(), min_len=6, max_len=6)

    assert len(segments) == 2
    for segment in segments:
        assert segment["position_global_full_gt_world"].shape[0] == 6
        assert segment["lhand_contact"].shape[0] == 6
        assert segment["lhand_contact_raw"].shape[0] == 6
        assert segment["source_frame_indices"].shape[0] == 6
        assert segment["human_imu_real"].shape[0] == 6
        assert segment["array_stream"].shape[0] == 6
        assert segment["obj_points_canonical"].shape == (256, 3)
        assert segment["metadata"] == "kept"


def test_downsample_sequence_uses_the_same_temporal_inference():
    downsampled = downsample_sequence(_sample_sequence(), factor=2)

    assert downsampled["position_global_full_gt_world"].shape[0] == 6
    assert downsampled["lhand_contact_raw"].shape[0] == 6
    assert downsampled["source_frame_indices"].tolist() == [0, 2, 4, 6, 8, 10]
    assert downsampled["human_imu_real"].shape[0] == 6
    assert downsampled["array_stream"].shape[0] == 6
    assert downsampled["obj_points_canonical"].shape == (256, 3)


def test_segment_unheld_motion_uses_cleaned_hand_labels():
    segment = _sample_sequence(length=6)
    segment["obj_trans"][:, 0] = torch.linspace(0.0, 0.12, 6)
    segment["lhand_contact_raw"][:] = True

    result = find_segment_unheld_object_motion(segment, displacement_threshold_m=0.10)

    assert result is not None
    assert result["max_displacement_m"] >= 0.10


def test_process_pt_file_drops_only_bad_segment_without_filling_index(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("random.randint", lambda _minimum, _maximum: 6)
    data = _sample_sequence(length=12)
    data["obj_trans"][:6, 0] = torch.linspace(0.0, 0.12, 6)
    data["lhand_contact"][6:] = True
    input_path = tmp_path / "source.pt"
    output_dir = tmp_path / "segments"
    torch.save(data, input_path)
    skipped_records = []

    saved_count = process_pt_file(
        str(input_path),
        str(output_dir),
        min_len=6,
        max_len=6,
        skipped_records=skipped_records,
    )

    assert saved_count == 1
    assert not (output_dir / "sample_seg000.pt").exists()
    assert (output_dir / "sample_seg001.pt").exists()
    assert len(skipped_records) == 1
    assert skipped_records[0]["candidate_segment_index"] == 0
