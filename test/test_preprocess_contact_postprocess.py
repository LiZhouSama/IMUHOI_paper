import json
import tempfile
import unittest
from pathlib import Path

import torch

from process.preprocess import (
    find_unheld_object_displacement_interval,
    postprocess_contact_labels,
    split_unified_by_json,
)


def _mask(*runs):
    values = []
    for value, length in runs:
        values.extend([value] * length)
    return torch.tensor(values, dtype=torch.bool)


def _positions(*x_coordinates):
    return torch.tensor(
        [[x_coordinate, 0.0, 0.0] for x_coordinate in x_coordinates],
        dtype=torch.float32,
    )


class ContactPostprocessTests(unittest.TestCase):
    def test_postprocess_fills_internal_hole_before_min_duration_filter(self):
        raw = _mask((True, 20), (False, 5), (True, 20))

        result = postprocess_contact_labels(
            raw,
            max_hole_frames=5,
            min_contact_frames=30,
        )

        self.assertTrue(bool(result.all()))

    def test_postprocess_does_not_fill_edge_or_overlong_holes(self):
        raw = _mask(
            (False, 5),
            (True, 30),
            (False, 31),
            (True, 30),
            (False, 5),
        )

        result = postprocess_contact_labels(
            raw,
            max_hole_frames=30,
            min_contact_frames=30,
        )

        self.assertFalse(bool(result[:5].any()))
        self.assertTrue(bool(result[5:35].all()))
        self.assertFalse(bool(result[35:66].any()))
        self.assertTrue(bool(result[66:96].all()))
        self.assertFalse(bool(result[96:].any()))

    def test_postprocess_removes_29_frames_but_keeps_exactly_30(self):
        raw = _mask((True, 29), (False, 31), (True, 30))

        result = postprocess_contact_labels(
            raw,
            max_hole_frames=30,
            min_contact_frames=30,
        )

        self.assertFalse(bool(result[:60].any()))
        self.assertTrue(bool(result[60:].all()))

    def test_skip_detects_unheld_displacement_after_prior_hand_contact(self):
        obj_trans = _positions(0.00, 0.02, 0.04, 0.08, 0.15)
        lhand_contact = _mask((True, 2), (False, 3))
        rhand_contact = _mask((False, 5))

        interval = find_unheld_object_displacement_interval(
            obj_trans,
            lhand_contact,
            rhand_contact,
            min_displacement_m=0.10,
        )

        self.assertIsNotNone(interval)
        self.assertEqual(interval["interval_start_frame"], 2)
        self.assertEqual(interval["interval_end_frame"], 4)
        self.assertAlmostEqual(interval["max_displacement_m"], 0.11, places=6)

    def test_hand_contact_splits_displacement_into_independent_intervals(self):
        obj_trans = _positions(0.00, 0.06, 0.12)
        lhand_contact = _mask((False, 1), (True, 1), (False, 1))
        rhand_contact = _mask((False, 3))

        interval = find_unheld_object_displacement_interval(
            obj_trans,
            lhand_contact,
            rhand_contact,
            min_displacement_m=0.10,
        )

        self.assertIsNone(interval)

    def test_skip_uses_max_pairwise_displacement_not_path_length(self):
        no_hand = _mask((False, 3))
        reaches_threshold = find_unheld_object_displacement_interval(
            _positions(0.05, 0.00, 0.10),
            no_hand,
            no_hand,
            min_displacement_m=0.10,
        )
        below_threshold = find_unheld_object_displacement_interval(
            _positions(0.00, 0.06, 0.00),
            no_hand,
            no_hand,
            min_displacement_m=0.10,
        )

        self.assertIsNotNone(reaches_threshold)
        self.assertEqual(reaches_threshold["motion_start_frame"], 1)
        self.assertEqual(reaches_threshold["motion_end_frame"], 2)
        self.assertIsNone(below_threshold)

    def test_split_preserves_numeric_hole_and_writes_filtered_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            unified_dir = temp_path / "unified"
            train_dir = temp_path / "train"
            test_dir = temp_path / "test"
            unified_dir.mkdir()
            torch.save({"index": 0}, unified_dir / "0.pt")
            torch.save({"index": 2}, unified_dir / "2.pt")

            source_split_info = temp_path / "source_split_info.json"
            source_split_info.write_text(
                json.dumps(
                    {
                        "total_files": 3,
                        "train_files": ["0.pt", "1.pt"],
                        "test_files": ["2.pt"],
                    }
                ),
                encoding="utf-8",
            )
            output_split_info = temp_path / "split_info.json"

            result = split_unified_by_json(
                str(unified_dir),
                str(source_split_info),
                str(train_dir),
                str(test_dir),
                allowed_missing_files={"1.pt"},
                output_split_info_path=str(output_split_info),
            )

            self.assertEqual(sorted(path.name for path in train_dir.glob("*.pt")), ["0.pt"])
            self.assertEqual(sorted(path.name for path in test_dir.glob("*.pt")), ["2.pt"])
            self.assertEqual(result["skipped_files"], ["1.pt"])
            self.assertEqual(result["total_files"], 2)
            self.assertEqual(
                json.loads(output_split_info.read_text(encoding="utf-8"))["skipped_files"],
                ["1.pt"],
            )

    def test_split_rejects_unmarked_missing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            unified_dir = temp_path / "unified"
            unified_dir.mkdir()
            torch.save({"index": 0}, unified_dir / "0.pt")
            source_split_info = temp_path / "source_split_info.json"
            source_split_info.write_text(
                json.dumps(
                    {
                        "total_files": 2,
                        "train_files": ["0.pt"],
                        "test_files": ["1.pt"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "未授权缺失"):
                split_unified_by_json(
                    str(unified_dir),
                    str(source_split_info),
                    str(temp_path / "train"),
                    str(temp_path / "test"),
                )


if __name__ == "__main__":
    unittest.main()
