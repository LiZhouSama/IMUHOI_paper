"""Synthetic, CPU-only publisher calibration tests; never load a real model."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_hoi_publisher.py"
spec = importlib.util.spec_from_file_location("calibrate_hoi_publisher", MODULE_PATH)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def make_record(path, published=1.0, raw=0.0, frames=4):
    def positions(value, hands=False):
        shape = (frames, 2, 3) if hands else (frames, 3)
        tensor = torch.zeros(shape, dtype=torch.float64)
        tensor[..., 0] = value
        return tensor

    return {
        "seq_file": str(path), "seq_name": "same-name-is-not-the-file-id",
        "obj_name": "box", "publish_len": frames, "has_object": True,
        "pred_obj_trans": positions(raw),
        "pred_obj_trans_published": positions(published),
        "gt_obj_trans": positions(0.0),
        "pred_palm_glb_pos": positions(0.0, hands=True),
        "stage1_pred_palm_glb_pos": positions(0.0, hands=True),
        "gt_palm_pos_eval": positions(0.0, hands=True),
        "gt_palm_pos": positions(0.0, hands=True),
        "gt_lhand_contact": torch.ones(frames, dtype=torch.bool),
        "gt_rhand_contact": torch.zeros(frames, dtype=torch.bool),
    }


def make_development(tmp_path, train_values=(1.0, 0.0), val_values=(1.0, 0.0)):
    source = tmp_path / "train"
    source.mkdir()
    records = []
    rows = []
    for name, partition, values in (("a.pt", "train", train_values), ("b.pt", "validation", val_values)):
        path = source / name
        path.touch()
        records.append(make_record(path, *values))
        rows.append({"file": name, "path": str(path), "partition": partition, "group": f"subject_{partition}"})
    manifest = {"source": str(source), "rows": rows}
    payload = {"mode": "online", "data_path": str(source), "online_lookahead": 10, "records": records}
    return payload, manifest


def test_single_hand_oracle_is_zero_and_no_contact_is_undefined():
    hands = torch.randn(5, 2, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    contacts = torch.tensor([[True, False]] * 5)
    assert calibration.object_only_oracle_cm(hands, torch.zeros_like(hands), contacts) == 0.0
    assert calibration.object_only_oracle_cm(hands, torch.zeros_like(hands), torch.zeros_like(contacts)) is None


def test_bimanual_oracle_and_mixed_hand_frame_denominator():
    hands = torch.zeros(3, 2, 3, dtype=torch.float64)
    hands[:, 0, 0] = 1.0
    hands[:, 1, 0] = -1.0
    gt = torch.zeros_like(hands)
    assert calibration.object_only_oracle_cm(hands, gt, torch.ones(3, 2, dtype=torch.bool)) == 100.0
    mixed = torch.tensor([[True, False], [True, True], [False, False]])
    assert calibration.object_only_oracle_cm(hands, gt, mixed) == pytest.approx(200.0 / 3.0)
    shifted = hands + torch.tensor([3.0, 2.0, -4.0])
    assert calibration.object_only_oracle_cm(shifted, gt, mixed) == pytest.approx(200.0 / 3.0)


def test_oracle_bound_is_attained_by_contact_target_midpoint():
    generator = torch.Generator().manual_seed(14)
    final = torch.randn(9, 2, 3, generator=generator, dtype=torch.float64)
    gt_hands = torch.randn(9, 2, 3, generator=generator, dtype=torch.float64)
    gt_object = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    contact = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0], [1, 1], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=torch.bool)
    desired = final + gt_object[:, None] - gt_hands
    oracle = (desired * contact[..., None]).sum(1) / contact.sum(1).clamp_min(1)[:, None]
    attained = calibration.hoi_error_cm(oracle, gt_object, final, gt_hands, contact)
    bound = calibration.object_only_oracle_cm(final, gt_hands, contact)
    assert attained == pytest.approx(bound, abs=1e-10)
    ordinary = calibration.hoi_error_cm(gt_object, gt_object, final, gt_hands, contact)
    assert ordinary >= bound - 1e-10


def test_lambda_zero_identity_one_raw_and_no_mutation():
    published = torch.randn(5, 3)
    raw = torch.randn(5, 3)
    before = published.clone()
    assert calibration.calibrate_positions(published, raw, 0.0) is published
    assert calibration.calibrate_positions(published, raw, 1.0) is raw
    torch.testing.assert_close(calibration.calibrate_positions(published, raw, 0.25), published + 0.25 * (raw - published))
    assert torch.equal(published, before)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_lambda_is_rejected(value):
    with pytest.raises(ValueError, match="lambda"):
        calibration.calibrate_positions(torch.zeros(3, 3), torch.zeros(3, 3), value)


def test_four_way_metrics_distinguish_final_predicted_hands_and_eval_gt():
    record = make_record("a.pt", published=0.1, raw=0.0)
    record["stage1_pred_palm_glb_pos"][..., 0] = 9.0
    record["ot_hand_positions"] = torch.zeros_like(record["pred_palm_glb_pos"])
    record["ot_hand_positions"][..., 0] = 0.2  # OT input takes precedence over Stage-1 fallback.
    record["pred_palm_glb_pos"][..., 0] = 0.4
    record["gt_palm_pos_eval"][..., 0] = 0.5
    report = calibration.score_records([calibration.prepare_record(record)], 0.0)
    four = report["summary"]["hoi_four_way_cm"]
    assert list(four) == list(calibration.FOUR_WAY_KEYS)
    assert list(four.values()) == pytest.approx([20.0, 30.0, 10.0, 20.0])
    assert report["summary"]["calibrated"]["hoi_err_cm"] == pytest.approx(20.0)
    assert report["summary"]["calibrated"]["obj_err_cm"] == pytest.approx(10.0)
    assert report["summary"]["calibrated"]["obj_jitter_mm_per_frame2"] == 0.0


def test_sequence_mean_is_not_frame_mean_and_jitter_has_no_fps_factor():
    short = calibration.prepare_record(make_record("a.pt", published=1.0, frames=3))
    long = calibration.prepare_record(make_record("b.pt", published=3.0, frames=30))
    report = calibration.score_records([short, long], 0.0)
    assert report["summary"]["baseline"]["hoi_err_cm"] == 200.0
    assert report["summary"]["baseline"]["obj_err_cm"] == 200.0
    record = make_record("c.pt", published=0.0, frames=3)
    record["pred_obj_trans_published"][:, 0] = torch.tensor([0.0, 0.001, 0.004])
    result = calibration.score_records([calibration.prepare_record(record)], 0.0)
    assert result["summary"]["baseline"]["obj_jitter_mm_per_frame2"] == pytest.approx(2.0)


@pytest.mark.parametrize("key", ["gt_palm_pos_eval", "pred_palm_glb_pos", "pred_obj_trans_published", "stage1_pred_palm_glb_pos"])
def test_required_geometry_never_silently_falls_back(key):
    record = make_record("a.pt")
    del record[key]
    with pytest.raises(ValueError, match=key):
        calibration.prepare_record(record)


def test_no_contact_or_empty_output_is_json_null_not_nan():
    record = make_record("a.pt", frames=1)
    record["gt_lhand_contact"].zero_()
    empty = make_record("b.pt", frames=0)
    result = calibration.score_records([calibration.prepare_record(record), calibration.prepare_record(empty)], 0.0)
    assert result["summary"]["baseline"]["hoi_err_cm"] is None
    assert result["summary"]["oracle_object_only_hoi_cm"] is None
    assert result["summary"]["baseline"]["obj_jitter_mm_per_frame2"] is None
    assert result["summary"]["object_metric_sequences"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kind", ["nonfinite", "shape", "nonbinary"])
def test_invalid_tensor_schema_is_rejected(kind):
    record = make_record("a.pt")
    if kind == "nonfinite":
        record["gt_palm_pos_eval"][0, 0, 0] = float("nan")
    elif kind == "shape":
        record["pred_obj_trans_published"] = record["pred_obj_trans_published"][:-1]
    else:
        record["gt_lhand_contact"] = torch.full((4,), 0.5)
    with pytest.raises(ValueError):
        calibration.prepare_record(record)


@pytest.mark.parametrize("kind", ["manifest_test", "cache_test", "record_test", "partition_test", "offline", "group_overlap", "row_path_test"])
def test_fit_rejects_test_or_ambiguous_provenance_before_gt_access(tmp_path, kind):
    payload, manifest = make_development(tmp_path)
    test = tmp_path / "test"
    test.mkdir()
    if kind == "manifest_test":
        manifest["source"] = str(test)
        payload["data_path"] = str(test)
    elif kind == "cache_test":
        payload["data_path"] = str(test)
    elif kind == "record_test":
        payload["records"][0]["seq_file"] = str(test / "a.pt")
    elif kind == "partition_test":
        manifest["rows"][1]["partition"] = "test"
    elif kind == "offline":
        payload["mode"] = "offline"
    elif kind == "group_overlap":
        manifest["rows"][1]["group"] = manifest["rows"][0]["group"]
    else:
        manifest["rows"][0]["path"] = str(test / "a.pt")
    # Schema validation would fail on this field if provenance did not run first.
    del payload["records"][0]["gt_palm_pos_eval"]
    with pytest.raises(ValueError) as error:
        calibration.fit_calibration(payload, manifest)
    assert "gt_palm_pos_eval" not in str(error.value)


@pytest.mark.parametrize("kind", ["duplicate_manifest", "duplicate_cache", "missing", "unexpected"])
def test_exact_manifest_coverage_is_required(tmp_path, kind):
    payload, manifest = make_development(tmp_path)
    if kind == "duplicate_manifest":
        manifest["rows"].append(copy.deepcopy(manifest["rows"][0]))
    elif kind == "duplicate_cache":
        payload["records"].append(copy.deepcopy(payload["records"][0]))
    elif kind == "missing":
        payload["records"].pop()
    else:
        payload["records"].append(make_record(Path(manifest["source"]) / "c.pt"))
    with pytest.raises(ValueError):
        calibration.fit_calibration(payload, manifest)


def test_grid_is_fitted_only_on_train_not_validation(tmp_path, monkeypatch):
    payload, manifest = make_development(tmp_path, train_values=(1.0, -1.0), val_values=(0.75, -0.25))
    manifest["rows"][1]["partition"] = "val"  # Accepted alias.
    objective = calibration._train_objective
    seen = []

    def check_train_only(records, value):
        seen.append([r["file"] for r in records])
        return objective(records, value)

    monkeypatch.setattr(calibration, "_train_objective", check_train_only)
    lock, report = calibration.fit_calibration(payload, manifest)
    assert len(seen) == 41 and all(names == ["a.pt"] for names in seen)
    assert lock["train_fitted_lambda"] == 0.5
    assert lock["lambda"] == 0.5  # Validation optimum is 0.75 but must not be searched.
    assert lock["accepted"] is True
    assert report["validation"]["summary"]["calibrated"]["hoi_err_cm"] == 25.0
    assert report["validation"]["per_sequence"][0]["partition"] == "validation"


def test_validation_object_guard_falls_back_to_exact_baseline(tmp_path):
    payload, manifest = make_development(tmp_path, val_values=(0.0, 0.2))
    payload["records"][1]["pred_palm_glb_pos"][..., 0] = 0.3
    lock, report = calibration.fit_calibration(payload, manifest)
    assert lock["train_fitted_lambda"] == 1.0
    assert lock["lambda"] == 0.0
    assert lock["selection"]["validation_hoi_improved"] is True
    assert lock["selection"]["reason"] == "validation_object_worsened"
    assert report["validation"]["summary"]["calibrated"] == report["validation"]["summary"]["baseline"]


def test_validation_hoi_guard_never_searches_an_alternative(tmp_path):
    payload, manifest = make_development(tmp_path, val_values=(0.1, 1.0))
    lock, report = calibration.fit_calibration(payload, manifest)
    assert lock["train_fitted_lambda"] == 1.0
    assert lock["lambda"] == 0.0
    assert lock["selection"]["reason"] == "validation_hoi_not_improved"
    assert report["validation"]["candidate_lambda"] == 1.0


def test_grid_ties_select_zero_and_empty_contact_partitions_fail(tmp_path):
    payload, manifest = make_development(tmp_path, train_values=(1.0, 1.0))
    lock, _ = calibration.fit_calibration(payload, manifest)
    assert lock["train_fitted_lambda"] == lock["lambda"] == 0.0
    for index in (0, 1):
        changed = copy.deepcopy(payload)
        changed["records"][index]["gt_lhand_contact"].zero_()
        with pytest.raises(ValueError, match="no scored GT contact"):
            calibration.fit_calibration(changed, manifest)


def test_evaluate_does_not_reselect_or_reject_lambda_on_test(tmp_path, monkeypatch):
    payload, manifest = make_development(tmp_path)
    lock, _ = calibration.fit_calibration(payload, manifest)
    assert lock["lambda"] == 1.0
    test_payload = {"mode": "online", "data_path": str(tmp_path / "test"), "online_lookahead": 10,
                    "records": [make_record(tmp_path / "test" / "c.pt", published=0.0, raw=2.0)]}

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must never fit or select")

    monkeypatch.setattr(calibration, "fit_calibration", forbidden)
    monkeypatch.setattr(calibration, "_train_objective", forbidden)
    result = calibration.evaluate_calibration(test_payload, lock)
    assert result["lambda"] == 1.0 and result["selection_performed"] is False
    assert result["summary"]["calibrated"]["hoi_err_cm"] == 200.0
    assert result["summary"]["baseline"]["hoi_err_cm"] == 0.0


def test_lock_integrity_and_online_protocol_are_checked(tmp_path):
    payload, manifest = make_development(tmp_path)
    lock, _ = calibration.fit_calibration(payload, manifest)
    tampered = copy.deepcopy(lock)
    tampered["lambda"] = 0.2
    with pytest.raises(ValueError, match="integrity"):
        calibration.evaluate_calibration(payload, tampered)
    changed = copy.deepcopy(payload)
    changed["online_lookahead"] = 0
    with pytest.raises(ValueError, match="protocol mismatch"):
        calibration.evaluate_calibration(changed, lock)


def test_cli_writes_small_lock_and_full_reports_without_overwriting(tmp_path):
    payload, manifest = make_development(tmp_path)
    dump, split = tmp_path / "online.pt", tmp_path / "split_manifest.json"
    torch.save(payload, dump)
    split.write_text(json.dumps(manifest), encoding="utf-8")
    lock_path = tmp_path / "calibration.json"
    assert calibration.main(["fit", "--dump", str(dump), "--split-manifest", str(split), "--output", str(lock_path)]) == 0
    lock = json.loads(lock_path.read_text())
    report = json.loads((tmp_path / "calibration.report.json").read_text())
    assert calibration.validate_calibration(lock) == 1.0
    assert "per_sequence" not in lock and lock_path.stat().st_size < 10000
    assert len(lock["provenance"]["cache_sha256"]) == 64
    assert len(report["train_grid"]) == 41
    assert len(report["train"]["per_sequence"]) == len(report["validation"]["per_sequence"]) == 1
    evaluation = tmp_path / "evaluation.json"
    assert calibration.main(["evaluate", "--dump", str(dump), "--calibration", str(lock_path), "--output", str(evaluation)]) == 0
    result = json.loads(evaluation.read_text())
    assert result["lambda"] == lock["lambda"]
    assert len(result["per_sequence"]) == 2
    before = lock_path.read_bytes()
    with pytest.raises(ValueError, match="distinct new files"):
        calibration.main(["fit", "--dump", str(dump), "--split-manifest", str(split), "--output", str(lock_path)])
    assert lock_path.read_bytes() == before


def test_fitting_uses_final_palms_and_canonical_gt_not_diagnostic_hands(tmp_path):
    payload, manifest = make_development(tmp_path)
    for record in payload["records"]:
        record["pred_palm_glb_pos"][..., 0] = 0.8
        record["gt_palm_pos_eval"][..., 0] = 0.3
        record["stage1_pred_palm_glb_pos"][..., 0] = 0.1
        record["gt_palm_pos"][..., 0] = 0.7
    lock, report = calibration.fit_calibration(payload, manifest)
    # Final/eval geometry targets object x=0.5; diagnostic geometry targets -0.6.
    assert lock["train_fitted_lambda"] == lock["lambda"] == 0.5
    assert report["train"]["summary"]["calibrated"]["hoi_err_cm"] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("kind", ["top_split", "nested_split", "nested_source", "oracle_feedback"])
def test_schema_v2_provenance_cannot_hide_a_test_or_oracle_cache(tmp_path, kind):
    payload, manifest = make_development(tmp_path)
    if kind == "top_split":
        payload["split"] = "test"
    elif kind == "nested_split":
        payload["provenance"] = {"split": "test"}
    elif kind == "nested_source":
        payload["provenance"] = {"data_path": str(tmp_path / "test")}
    else:
        payload["provenance"] = {"feedback_mode": "gt_prev"}
    with pytest.raises(ValueError):
        calibration.fit_calibration(payload, manifest)


def test_schema_v2_locks_checkpoint_and_gt_geometry_contents_not_paths(tmp_path):
    payload, manifest = make_development(tmp_path)
    payload["split"] = "train"
    payload["provenance"] = {
        "split": "train", "feedback_mode": "normal",
        "checkpoints": {"human_pose": {"path": "/old/hp.pt", "sha256": "a" * 64}},
        "gt_palm_pos_eval": {"definition": "canonical FK", "body_model": {"path": "/old/model.npz", "sha256": "b" * 64}, "betas": "zero"},
    }
    lock, _ = calibration.fit_calibration(payload, manifest)
    test_payload = copy.deepcopy(payload)
    test_payload["split"] = "test"
    test_payload["provenance"]["split"] = "test"
    test_payload["data_path"] = str(tmp_path / "test")
    test_payload["provenance"]["checkpoints"]["human_pose"]["path"] = "/new/hp.pt"
    assert calibration.evaluate_calibration(test_payload, lock)["lambda"] == lock["lambda"]
    changed = copy.deepcopy(test_payload)
    changed["provenance"]["checkpoints"]["human_pose"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="protocol mismatch"):
        calibration.evaluate_calibration(changed, lock)
    changed = copy.deepcopy(test_payload)
    changed["provenance"]["gt_palm_pos_eval"]["body_model"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="protocol mismatch"):
        calibration.evaluate_calibration(changed, lock)
