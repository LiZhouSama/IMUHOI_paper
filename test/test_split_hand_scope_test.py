import numpy as np
import pytest
import torch

from scripts.split_hand_scope_test import find_cuts, retained_intervals, slice_sequence


def test_bridges_only_short_static_gaps():
    moving = np.r_[np.ones(35), np.zeros(15), np.ones(25), np.zeros(16), np.ones(59)]
    cuts = find_cuts(moving, np.zeros(len(moving)))
    assert cuts == [{"start": 0, "end": 75, "frames": 75, "moving_frames": 60, "static_gap_frames": 15}]


def test_hand_contact_blocks_bridge():
    moving = np.r_[np.ones(35), np.zeros(1), np.ones(35)]
    hand = np.zeros(len(moving)); hand[35] = 1
    assert find_cuts(moving, hand) == []


def test_gap_frames_do_not_count_toward_duration():
    moving = np.r_[np.ones(29), np.zeros(15), np.ones(30)]
    assert find_cuts(moving, np.zeros(len(moving))) == []


def test_minimum_retained_includes_exactly_120():
    keep, short = retained_intervals(419, [{"start": 120, "end": 180}, {"start": 299, "end": 419}])
    assert keep == [(0, 120)]
    assert short == [(180, 299)]


def test_empty_and_full_cuts():
    assert retained_intervals(120, []) == ([(0, 120)], [])
    assert retained_intervals(119, []) == ([], [(0, 119)])
    assert retained_intervals(120, [{"start": 0, "end": 120}]) == ([], [])
    assert find_cuts(np.zeros(120), np.zeros(120)) == []
    assert find_cuts(np.ones(120), np.ones(120)) == []


def test_crop_preserves_static_points_even_when_length_is_256():
    data = {"obj_trans": torch.arange(768).reshape(256,3), "obj_points_canonical": torch.ones(256,3),
            "pose": np.ones((256,72)), "canonical_skeleton_version": 1, "body_model_type": "smplh",
            "source_pkls": ["smpl.npz", "obj.npz"]}
    result = slice_sequence(data, 30, 150)
    assert result["obj_trans"].shape == (120,3)
    assert result["pose"].shape == (120,72)
    assert result["obj_points_canonical"].shape == (256,3)
    assert result["canonical_skeleton_version"] == 1
    assert result["source_pkls"] == data["source_pkls"]
    result["obj_trans"][0] = 0
    assert data["obj_trans"][30,0] == 90


def test_frame_partition_and_no_remaining_qualifying_motion():
    rng = np.random.default_rng(42)
    for _ in range(100):
        moving = np.repeat(rng.integers(0,2,30), 20).astype(bool)
        hand = np.repeat(rng.integers(0,2,30), 20).astype(bool)
        cuts = find_cuts(moving, hand)
        keep, short = retained_intervals(len(moving), cuts)
        cover = np.zeros(len(moving), dtype=int)
        for a,b in keep+short+[(c["start"],c["end"]) for c in cuts]:
            cover[a:b] += 1
        assert (cover == 1).all()
        for a,b in keep:
            assert b-a >= 120
            assert not find_cuts(moving[a:b], hand[a:b])


def test_bad_lengths_rejected():
    with pytest.raises(ValueError):
        find_cuts([True], [False,False])
