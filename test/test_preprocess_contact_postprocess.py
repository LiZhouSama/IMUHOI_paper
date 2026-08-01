import json
import tempfile
import unittest
from pathlib import Path

import torch

from process.preprocess import (
    contact_postprocess_metadata,
    find_unheld_object_displacement_interval,
    postprocess_contact_labels,
    postprocess_sequence_contact_labels,
    saved_contact_postprocess_matches,
    split_unified_by_json,
    write_contact_skip_manifest,
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

    def test_full_sequence_cleanup_runs_before_unheld_motion_exclusion(self):
        lhand_raw = _mask((True, 10), (False, 30))
        rhand_raw = _mask((False, 40))
        obj_raw = _mask((True, 20), (False, 5), (True, 15))
        obj_trans = torch.zeros(40, 3)
        obj_trans[-1, 0] = 0.11

        lhand, rhand, obj, interval = postprocess_sequence_contact_labels(
            lhand_raw,
            rhand_raw,
            obj_raw,
            obj_trans,
            max_hole_frames=5,
            min_contact_frames=30,
            skip_object_displacement_without_hand_contact=True,
            unheld_object_displacement_threshold_m=0.10,
        )

        self.assertFalse(bool(lhand.any()))
        self.assertFalse(bool(rhand.any()))
        self.assertTrue(bool(obj.all()))
        self.assertIsNotNone(interval)
        self.assertEqual(interval["interval_start_frame"], 0)
        self.assertEqual(interval["interval_end_frame"], 39)

    def test_unheld_displacement_is_exact_across_distance_blocks(self):
        frame_count = 2050
        obj_trans = torch.zeros(frame_count, 3)
        obj_trans[-1, 0] = 0.11
        no_hand = torch.zeros(frame_count, dtype=torch.bool)

        interval = find_unheld_object_displacement_interval(
            obj_trans,
            no_hand,
            no_hand,
            min_displacement_m=0.10,
        )

        self.assertIsNotNone(interval)
        self.assertEqual(interval["motion_start_frame"], 0)
        self.assertEqual(interval["motion_end_frame"], frame_count - 1)
        self.assertAlmostEqual(interval["max_displacement_m"], 0.11, places=6)

    def test_contact_metadata_controls_skip_existing_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "max_hole_frames": 30,
                "min_contact_frames": 30,
                "skip_object_displacement_without_hand_contact": True,
                "unheld_object_displacement_threshold_m": 0.10,
            }
            output_path = Path(temp_dir) / "1.pt"
            torch.save(contact_postprocess_metadata(**config), output_path)

            self.assertTrue(
                saved_contact_postprocess_matches(str(output_path), **config)
            )
            self.assertFalse(
                saved_contact_postprocess_matches(
                    str(output_path),
                    **{**config, "max_hole_frames": 29},
                )
            )

            manifest_path = write_contact_skip_manifest(
                temp_dir,
                dataset="TEST",
                source_sequences=2,
                saved_sequences=1,
                skipped_sequences=[{"output_file": "2.pt"}],
                **config,
            )
            manifest = json.loads(
                Path(manifest_path).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["saved_sequences"], 1)
            self.assertEqual(manifest["skipped_sequences_count"], 1)
            self.assertEqual(manifest["contact_postprocess_version"], 1)

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
