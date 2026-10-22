"""Small, CPU-only tests for the remaining-datasets experiment queue."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("remaining_comparison", ROOT / "scripts/run_remaining_datasets_comparison.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def test_segment_source_ids_keep_original_sequences():
    files = [Path("person_object_seg000.pt"), Path("person_object_seg019.pt"), Path("other.pt")]
    assert queue.source_ids(files) == {"person_object", "other"}


def test_audit_rejects_train_full_sequence_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    for folder, filename in (("train", "same_seg000.pt"), ("test", "other_seg000.pt"), ("full", "same.pt")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / filename).touch()
    cfg = {"batch_size": 1, "windows_per_sequence": 4, "datasets": {"imhd": {
        "train_path": "train", "test_path": "test", "full_sequence_test_path": "full", "body_model_type": "smplh"}}}
    with pytest.raises(ValueError, match="source overlap"):
        queue.audit_dataset(cfg, "imhd")


def test_audit_uses_actual_drop_last_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    for folder, filenames in (("train", ["a_seg000.pt", "a_seg001.pt", "b.pt"]),
                              ("test", ["c_seg000.pt"]), ("full", ["c.pt"])):
        (tmp_path / folder).mkdir()
        for filename in filenames:
            (tmp_path / folder / filename).touch()
    cfg = {"batch_size": 5, "windows_per_sequence": 4, "datasets": {"imhd": {
        "train_path": "train", "test_path": "test", "full_sequence_test_path": "full", "body_model_type": "smplh"}}}
    audit = queue.audit_dataset(cfg, "imhd")
    assert audit["train_steps_per_epoch"] == 2
    assert audit["train_source_sequences"] == 2
    cfg["batch_size"] = 13
    with pytest.raises(ValueError, match="zero training batches"):
        queue.audit_dataset(cfg, "imhd")


def _jobs(tmp_path, status="queued"):
    return [{"dataset": "imhd", "method": method, "variant": spec[1],
             "root": str(tmp_path / "imhd" / method), "status": status}
            for method, spec in queue.METHODS.items()]


def test_partial_summary_does_not_fabricate_zero_results(tmp_path):
    queue.summary(tmp_path, _jobs(tmp_path))
    rows = json.loads((tmp_path / "metrics.json").read_text())
    assert len(rows) == 4
    assert all(row["obj_trans_err_fusion_refine"] is None for row in rows)
    text = (tmp_path / "summary.md").read_text()
    assert "queued" in text and "separate" in text
    assert "20.00%" not in text


def test_completed_summary_uses_matching_best_and_final(tmp_path):
    jobs = _jobs(tmp_path, status="complete")
    for job in jobs:
        path = Path(job["root"]) / "experiment_manifest.json"
        path.parent.mkdir(parents=True)
        proposed = job["method"] == "proposed"
        metrics = {"obj_trans_err_fusion_refine": 8.0 if proposed else 10.0,
                   "hoi_err_fusion": 3.0 if proposed else 6.0, "obj_jitter": float("nan")}
        path.write_text(json.dumps({"status": "complete", "pipelines": {job["variant"]: {
            "status": "complete", "metrics_best": metrics, "metrics_final": metrics}}}))
    queue.summary(tmp_path, jobs)
    text = (tmp_path / "summary.md").read_text()
    assert text.count("20.00%") == 2 and text.count("50.00%") == 2
    records = json.loads((tmp_path / "metrics.json").read_text())
    assert all(row["obj_jitter"] is None for row in records)
