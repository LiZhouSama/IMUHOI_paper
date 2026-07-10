import torch

from dataset.dataset_IMUHOI import IMUDataset


def _write_synthetic_sequence(tmp_path):
    seq_len = 6
    joint_count = 22
    rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(seq_len, joint_count)
    rotation_global = torch.eye(3).view(1, 1, 3, 3).repeat(seq_len, joint_count, 1, 1)

    position_global = torch.zeros(seq_len, joint_count, 3)
    left_lengths = torch.tensor([0.10, 0.20, 0.30, 0.40, 0.50, 0.50])
    right_lengths = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20, 0.20])
    position_global[:, 20, 0] = left_lengths
    position_global[:, 21, 0] = right_lengths
    position_global[:, 18, :] = position_global[:, 20, :]
    position_global[:, 19, :] = position_global[:, 21, :]

    lhand_contact = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool)
    rhand_contact = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    data = {
        "seq_name": "synthetic_bimanual",
        "gender": "neutral",
        "rotation_local_full_gt_list": rot6d.float(),
        "position_global_full_gt_world": position_global.float(),
        "trans": torch.zeros(seq_len, 3),
        "rotation_global": rotation_global.float(),
        "lfoot_contact": torch.zeros(seq_len),
        "rfoot_contact": torch.zeros(seq_len),
        "obj_name": "box",
        "has_object": True,
        "obj_scale": torch.ones(seq_len),
        "obj_trans": torch.zeros(seq_len, 3),
        "obj_rot": rotation_global[:, 0],
        "obj_com_pos": torch.zeros(seq_len, 3),
        "lhand_contact": lhand_contact,
        "rhand_contact": rhand_contact,
        "obj_contact": lhand_contact | rhand_contact,
        "obj_points_canonical": torch.zeros(8, 3),
        "obj_points_sample_count": 8,
    }
    path = tmp_path / "synthetic.pt"
    torch.save(data, path)
    return path


def test_bimanual_contact_postprocess_drops_higher_variation_hand(tmp_path):
    _write_synthetic_sequence(tmp_path)
    dataset = IMUDataset(
        data_dir=str(tmp_path),
        window_size=1,
        full_sequence=True,
        resolve_bimanual_contact_conflicts=True,
    )

    sample = dataset[0]

    assert not sample["lhand_contact"][:5].any()
    assert sample["rhand_contact"][:5].all()
    assert bool(sample["lhand_contact"][5])
    assert not bool(sample["rhand_contact"][5])
    assert torch.all(sample["lhand_lb"][:5] == 0)
    assert torch.all(sample["rhand_lb"][:5] > 0)


def test_bimanual_contact_postprocess_can_be_disabled(tmp_path):
    _write_synthetic_sequence(tmp_path)
    dataset = IMUDataset(
        data_dir=str(tmp_path),
        window_size=1,
        full_sequence=True,
        resolve_bimanual_contact_conflicts=False,
    )

    sample = dataset[0]

    assert sample["lhand_contact"][:5].all()
    assert sample["rhand_contact"][:5].all()
