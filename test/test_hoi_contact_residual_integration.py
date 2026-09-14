"""Small CPU integration gates for the real residual train/evaluate entry points.

All dumps, source train files, manifests and checkpoints live under pytest's
temporary directory. No real dataset, SMPL asset, GPU or evaluator import is
needed. A deliberately invalid test-source cache must FAIL CLOSED, even when
its numeric basename also exists in the locked train manifest.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import train_hoi_contact_residual as runner
from utils.hoi_metrics import compute_hoi_error_at_gt_contact
from utils.rotation_conversions import matrix_to_rotation_6d


def _z_rotation(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(-1, 3, 3)


def make_online_record(path, name, length, sensor_shift=0.0):
    """Construct the documented online dump contract, not SMPL internals."""
    t = torch.arange(length, dtype=torch.float32)
    root = torch.stack((0.004 * t, torch.zeros_like(t), 0.002 * t), -1)
    palms = root[:, None] + torch.tensor([[-0.25, 0.6, 0.3], [0.25, 0.6, 0.3]])
    palms = palms + torch.sin(t[:, None, None] * 0.2) * 0.003
    stage1_palms = palms - torch.tensor([0.075, 0.02, 0.0])
    raw = root + torch.tensor([0.12, 0.8, 0.5])
    left_fk = raw + torch.tensor([0.02, 0.0, 0.0])
    right_fk = raw - torch.tensor([0.02, 0.0, 0.0])
    # The formal published refusion has an analytic result when w_I=w_S=0.
    published = 0.5 * (palms[:, 0] + left_fk - stage1_palms[:, 0])
    published = published + 0.5 * (palms[:, 1] + right_fk - stage1_palms[:, 1])
    rotation = _z_rotation(0.15 + 0.015 * t)
    human_imu = torch.zeros(length, 6, 9)
    human_imu[..., 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    human_imu[..., 0] = (0.1 * t)[:, None]
    human_imu[:, 0, 3:] = matrix_to_rotation_6d(_z_rotation(0.25 + 0.01 * t))
    obj_imu = torch.cat((torch.stack((sensor_shift + 0.2 * t, 0.1 * t, 0.05 * t), -1),
                         matrix_to_rotation_6d(rotation)), -1)
    target_obj = published + torch.einsum("tij,j->ti", rotation, torch.tensor([0.03, -0.01, 0.005]))
    canonical_palms = palms + torch.tensor([[0.002, 0.003, 0.0], [-0.004, 0.0, 0.002]])
    return {
        "seq_file": str(path.resolve()), "seq_name": name, "obj_name": "synthetic_box",
        "seq_len": length, "publish_len": length, "has_object": True,
        "gt_human_imu": human_imu, "gt_obj_imu": obj_imu,
        "pred_obj_trans": raw, "pred_obj_trans_published": published,
        "pred_palm_glb_pos": palms, "stage1_pred_palm_glb_pos": stage1_palms,
        "root_trans_pred": root,
        "pred_hand_contact_prob": torch.ones(length, 3),
        "pred_lhand_obj_trans": left_fk, "pred_rhand_obj_trans": right_fk,
        "pred_imu_obj_trans": raw.clone(), "pred_static_obj_trans": raw.clone(),
        "gating_weights": torch.tensor([0.5, 0.5, 0.0, 0.0]).expand(length, 4).clone(),
        "gt_obj_trans": target_obj,
        "gt_palm_pos_eval": canonical_palms,
        # A valid-looking but deliberately wrong legacy palm definition.
        "gt_palm_pos": canonical_palms + torch.tensor([1.0, 0.0, 0.0]),
        "gt_obj_rot": matrix_to_rotation_6d(torch.eye(3).expand(length, 3, 3)),
        "gt_lhand_contact": (t.long() % 3) != 2,
        "gt_rhand_contact": (t.long() % 2) == 0,
        "gt_obj_contact": torch.ones(length, dtype=torch.bool),
    }


@pytest.fixture
def development(tmp_path):
    source = tmp_path / "dataset" / "train"
    source.mkdir(parents=True)
    groups = {"train": [], "validation": []}
    seed = 42
    for subject in range(100):
        group = "subject%03d_box" % subject
        hashed = hashlib.sha256(("%d:%s" % (seed, group)).encode()).hexdigest()
        partition = "validation" if int(hashed[:8], 16) % 5 == 0 else "train"
        if len(groups[partition]) < (2 if partition == "train" else 1):
            groups[partition].append(group)
        if len(groups["train"]) == 2 and len(groups["validation"]) == 1:
            break
    records = []
    for group, lengths, shift in zip(
        groups["train"] + groups["validation"], ((4, 9), (6, 12), (5, 11)), (0.0, 0.4, 20.0)
    ):
        for trial, length in enumerate(lengths):
            path = source / ("%04d.pt" % len(records))
            name = "%s_%02d" % (group, trial)
            record = make_online_record(path, name, length, sensor_shift=shift)
            torch.save({"seq_name": name, "human_imu": record["gt_human_imu"], "obj_name": "synthetic_box"}, path)
            records.append(record)
    split_dir = tmp_path / "split"
    runner.prepare(SimpleNamespace(data_path=str(source), output=str(split_dir), seed=seed,
                                   min_frames=3, train_sequences=0, validation_sequences=0))
    manifest_path = split_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = {"mode": "online", "data_path": str(source.resolve()), "records": records,
               "online_window": 60, "online_lookahead": 10, "simulate_imu_noise": True,
               "synthetic_integration_fixture": True}
    dump_path = tmp_path / "development_online.pt"
    torch.save(payload, dump_path)
    return SimpleNamespace(root=tmp_path, source=source, records=records, payload=payload,
                           dump=dump_path, manifest_path=manifest_path, manifest=manifest)


def training_args(data, architecture="pointwise", **overrides):
    values = dict(dump=str(data.dump), split_manifest=str(data.manifest_path),
                  output=str(data.root / ("run_" + architecture)), device="cpu", cpu_threads=1,
                  architecture=architecture, hidden_dim=8, max_correction=0.20,
                  batch_size=3, epochs=2, patience=5, lr=0.003, weight_decay=0.0,
                  object_weight=0.1, smooth_weight=0.01, residual_weight=0.01,
                  object_guard_cm=100.0, jitter_guard=1000.0, seed=42)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_prepare_creates_group_disjoint_manifest_from_train_sources(development):
    data = development
    rows = data.manifest["rows"]
    assert data.manifest["source"] == str(data.source.resolve())
    assert data.manifest["selected_sequences"] == len(data.records) == 6
    partitions = {group: set() for group in (row["group"] for row in rows)}
    for row in rows:
        partitions[row["group"]].add(row["partition"])
        assert Path(row["path"]).resolve() == (data.source / row["file"]).resolve()
        assert row["group"] == row["seq_name"].rsplit("_", 1)[0]
    assert all(len(values) == 1 for values in partitions.values())
    assert {row["group"] for row in rows if row["partition"] == "train"}.isdisjoint(
        {row["group"] for row in rows if row["partition"] == "validation"}
    )
    for partition in ("train", "validation"):
        listed = (data.manifest_path.parent / (partition + ".txt")).read_text().splitlines()
        assert set(listed) == {row["path"] for row in rows if row["partition"] == partition}
    with pytest.raises(ValueError, match="train"):
        runner.prepare(SimpleNamespace(data_path=str(data.source.parent / "test"),
                                       output=str(data.root / "bad_split")))


@pytest.mark.parametrize("architecture", ["pointwise", "gru"])
def test_real_train_evaluate_reload_and_train_only_normalization(development, monkeypatch, architecture):
    data = development
    args = training_args(data, architecture)
    if architecture == "pointwise":
        # The original dumper legitimately stores only the basename in seq_file.
        payload = deepcopy(data.payload)
        for record in payload["records"]:
            record["seq_file"] = Path(record["seq_file"]).name
        torch.save(payload, data.dump)
    updates = []
    original_step = torch.optim.AdamW.step

    def counted_real_step(optimizer, *positional, **keywords):
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        before = [parameter.detach().clone() for parameter in parameters]
        result = original_step(optimizer, *positional, **keywords)
        updates.append(sum(float((parameter.detach() - old).abs().sum()) for parameter, old in zip(parameters, before)))
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_real_step)
    runner.train(args)
    assert len(updates) == 4  # Four train sequences, batch_size=3, two full epochs.
    assert all(update > 0 for update in updates)
    checkpoint_path = Path(args.output) / "best.pt"
    checkpoint = runner.load_pt(checkpoint_path)
    report = json.loads((Path(args.output) / "training_report.json").read_text())
    assert [item["epoch"] for item in report["history"]] == [0, 1, 2]
    assert all(torch.isfinite(torch.tensor(item["loss"])) for item in report["history"][1:])
    assert report["test_used"] is False
    assert checkpoint["train_sequences"] == 4 and checkpoint["validation_sequences"] == 2
    assert checkpoint["settings"]["architecture"] == architecture
    assert checkpoint["settings"]["feature_dim"] == 99
    assert checkpoint["gain"] in (0.0, 0.25, 0.5, 0.75, 1.0)
    assert checkpoint["split_sha256"] == hashlib.sha256(data.manifest_path.read_bytes()).hexdigest()
    assert all(value.device.type == "cpu" for value in checkpoint["model_state_dict"].values())

    membership = {row["file"]: row["partition"] for row in data.manifest["rows"]}
    prepared = [runner.prepare_record(record) for record in data.records]
    train_features = torch.cat([record["features"] for record in prepared if membership[record["file"]] == "train"])
    all_features = torch.cat([record["features"] for record in prepared])
    expected_mean = train_features.mean(0)
    expected_std = train_features.std(0).clamp_min(0.03)
    torch.testing.assert_close(checkpoint["feature_mean"], expected_mean)
    torch.testing.assert_close(checkpoint["feature_std"], expected_std)
    assert not torch.allclose(expected_mean, all_features.mean(0))  # Validation is deliberately shifted.

    model = runner.model_from_settings(checkpoint["settings"]).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    second_path = data.root / (architecture + "_roundtrip.pt")
    runner.save_checkpoint(second_path, model, checkpoint["settings"], expected_mean, expected_std,
                           {"gain": checkpoint["gain"], "epoch": checkpoint["epoch"]})
    reloaded = runner.load_pt(second_path)
    clone = runner.model_from_settings(reloaded["settings"]).eval()
    clone.load_state_dict(reloaded["model_state_dict"], strict=True)
    assert reloaded["settings"] == checkpoint["settings"]
    for key, value in checkpoint["model_state_dict"].items():
        assert torch.equal(reloaded["model_state_dict"][key], value)
    batch = runner.collate(prepared[:3], torch.device("cpu"), expected_mean, expected_std)
    with torch.no_grad():
        expected_prediction, _ = runner.corrected_object(model, batch, checkpoint["gain"])
        reloaded_prediction, _ = runner.corrected_object(clone, batch, reloaded["gain"])
    torch.testing.assert_close(reloaded_prediction, expected_prediction, rtol=0, atol=0)

    validation_raw = [record for record in data.records if membership[Path(record["seq_file"]).name] == "validation"]
    validation = [runner.prepare_record(record) for record in validation_raw]
    eval_dump = data.root / (architecture + "_evaluation_online.pt")
    torch.save(dict(data.payload, records=validation_raw), eval_dump)
    eval_output = data.root / (architecture + "_evaluation.json")
    locked_checkpoint_bytes = checkpoint_path.read_bytes()
    runner.evaluate(SimpleNamespace(dump=str(eval_dump), checkpoint=str(checkpoint_path),
                                    output=str(eval_output), device="cpu", batch_size=2, cpu_threads=1))
    evaluation = json.loads(eval_output.read_text())
    reference = runner.score(model, validation, torch.device("cpu"), expected_mean, expected_std,
                             batch_size=1, gain=checkpoint["gain"], per_sequence=True)
    assert checkpoint_path.read_bytes() == locked_checkpoint_bytes
    assert evaluation["checkpoint_sha256"] == hashlib.sha256(locked_checkpoint_bytes).hexdigest()
    assert evaluation["sequences"] == 2 and evaluation["contact_sequences"] == 2
    for metric in ("hoi_cm", "baseline_hoi_cm", "obj_cm", "baseline_obj_cm", "correction_cm"):
        assert evaluation[metric] == pytest.approx(reference[metric], rel=1e-5, abs=1e-5)
    for row, record in zip(evaluation["per_sequence"], validation_raw):
        formal = compute_hoi_error_at_gt_contact(
            record["pred_obj_trans_published"], record["gt_obj_trans"], record["pred_palm_glb_pos"],
            record["gt_palm_pos_eval"], record["gt_lhand_contact"], record["gt_rhand_contact"]
        )
        assert row["baseline_hoi_cm"] == pytest.approx(formal, abs=1e-5)
    assert evaluation["gain"] == checkpoint["gain"]
    assert evaluation["paired_bootstrap_95_ci_cm"][0] <= evaluation["paired_bootstrap_95_ci_cm"][1]


@pytest.mark.parametrize("missing", ["gt_palm_pos_eval", "pred_obj_trans_published"])
@pytest.mark.parametrize("command", ["train", "evaluate"])
def test_train_and_evaluate_refuse_missing_canonical_fields(development, missing, command):
    payload = deepcopy(development.payload)
    del payload["records"][0][missing]
    assert "gt_palm_pos" in payload["records"][0] and "pred_obj_trans" in payload["records"][0]
    bad_dump = development.root / ("missing_" + missing + ".pt")
    torch.save(payload, bad_dump)
    args = training_args(development, dump=str(bad_dump), epochs=1)
    if command == "evaluate":
        settings = dict(architecture="pointwise", feature_dim=99, hidden_dim=8, max_correction=0.2)
        checkpoint_path = development.root / "canonical_guard_checkpoint.pt"
        runner.save_checkpoint(checkpoint_path, runner.model_from_settings(settings), settings,
                               torch.zeros(99), torch.ones(99), {"gain": 0.0, "epoch": 0})
        args = SimpleNamespace(dump=str(bad_dump), checkpoint=str(checkpoint_path), device="cpu",
                               cpu_threads=1, batch_size=2, output=str(development.root / "invalid_evaluation.json"))
    with pytest.raises(ValueError, match="canonical|published|regenerate"):
        getattr(runner, command)(args)
    assert not Path(args.output).exists()


@pytest.mark.parametrize("source_kind", ["top_level_test", "per_record_test"])
def test_numeric_test_source_collision_is_rejected(development, source_kind):
    data = development
    test_dir = data.source.parent / "test"
    test_dir.mkdir()
    collision = test_dir / Path(data.records[0]["seq_file"]).name
    torch.save({"human_imu": data.records[0]["gt_human_imu"]}, collision)
    assert collision.name in {row["file"] for row in data.manifest["rows"]}
    assert (data.source / collision.name).exists()
    payload = deepcopy(data.payload)
    payload["records"][0]["seq_file"] = str(collision.resolve())
    if source_kind == "top_level_test":
        payload["data_path"] = str(test_dir.resolve())
    bad_dump = data.root / (source_kind + ".pt")
    torch.save(payload, bad_dump)
    args = training_args(data, dump=str(bad_dump), epochs=1)
    with pytest.raises(ValueError, match="(?i)(source|path|train|split|origin|provenance|cache)"):
        runner.train(args)
    assert not (Path(args.output) / "best.pt").exists()


def test_training_refuses_offline_predictions(development):
    payload = dict(development.payload, mode="offline")
    bad_dump = development.root / "offline.pt"
    torch.save(payload, bad_dump)
    with pytest.raises(ValueError, match="online"):
        runner.train(training_args(development, dump=str(bad_dump), epochs=1))


def make_nonzero_model(prepared, architecture):
    settings = dict(architecture=architecture, feature_dim=99, hidden_dim=8, max_correction=0.2)
    torch.manual_seed(18)
    model = runner.model_from_settings(settings)
    batch = runner.collate(prepared, torch.device("cpu"), torch.zeros(99), torch.ones(99))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    # Exercise only the public forward contract; do not reach into a private head.
    for _ in range(3):
        optimizer.zero_grad()
        delta, _ = model(batch["features"])
        (delta - delta.new_tensor([0.03, -0.01, 0.005])).square().mean().backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, correction = runner.corrected_object(model, batch, gain=1.0)
    assert correction.abs().sum() > 0
    return model, batch


@pytest.mark.parametrize("architecture", ["pointwise", "gru"])
def test_gain_zero_is_exact_formal_published_baseline(development, architecture):
    raw = development.records[:3]
    prepared = [runner.prepare_record(record) for record in raw]
    model, batch = make_nonzero_model(prepared, architecture)
    with torch.no_grad():
        prediction, correction = runner.corrected_object(model, batch, gain=0.0)
    assert torch.equal(prediction, batch["base_obj"])
    assert torch.count_nonzero(correction) == 0
    scores = runner.score(model, prepared, torch.device("cpu"), torch.zeros(99), torch.ones(99),
                          batch_size=3, gain=0.0, per_sequence=True)
    for row, record, processed in zip(scores["per_sequence"], raw, prepared):
        assert torch.equal(processed["target_palm"], record["gt_palm_pos_eval"])
        expected = compute_hoi_error_at_gt_contact(
            record["pred_obj_trans_published"], record["gt_obj_trans"], record["pred_palm_glb_pos"],
            record["gt_palm_pos_eval"], record["gt_lhand_contact"], record["gt_rhand_contact"]
        )
        wrong_raw = compute_hoi_error_at_gt_contact(
            record["pred_obj_trans"], record["gt_obj_trans"], record["pred_palm_glb_pos"],
            record["gt_palm_pos_eval"], record["gt_lhand_contact"], record["gt_rhand_contact"]
        )
        assert abs(wrong_raw - expected) > 1.0
        assert row["hoi_cm"] == pytest.approx(expected, abs=1e-5)
        assert row["hoi_cm"] == row["baseline_hoi_cm"]
        assert row["obj_cm"] == row["baseline_obj_cm"]
        assert row["jitter"] == row["baseline_jitter"]
        assert row["correction_cm"] == 0.0


@pytest.mark.parametrize("architecture", ["pointwise", "gru"])
def test_unequal_padding_is_excluded_from_metrics_and_cannot_change_valid_predictions(development, architecture):
    records = [runner.prepare_record(development.records[index]) for index in (0, 3, 4)]
    assert [len(record["features"]) for record in records] == [4, 12, 5]
    model, batch = make_nonzero_model(records, architecture)
    assert batch["features"].shape == (3, 12, 99)
    assert torch.equal(batch["valid"].sum(1), torch.tensor([4, 12, 5]))
    with torch.no_grad():
        predicted, delta = runner.corrected_object(model, batch)
    assert torch.count_nonzero(delta[:, 0]) == 0  # Preserve each sequence's initialized first frame.
    expected = runner.sequence_losses(predicted, batch)
    poisoned = {key: value.clone() for key, value in batch.items()}
    for key in ("target_obj", "target_palm", "pred_palm"):
        poisoned[key][~batch["valid"]] = 1000.0
    poisoned["contact"][~batch["valid"]] = True
    observed = runner.sequence_losses(predicted, poisoned)
    for old, new in zip(expected, observed):
        torch.testing.assert_close(old, new)
    single = runner.score(model, records, torch.device("cpu"), torch.zeros(99), torch.ones(99),
                          batch_size=1, gain=0.75, per_sequence=True)
    padded = runner.score(model, records, torch.device("cpu"), torch.zeros(99), torch.ones(99),
                          batch_size=3, gain=0.75, per_sequence=True)
    for alone, together in zip(single["per_sequence"], padded["per_sequence"]):
        assert alone["frames"] == together["frames"]
        for key in ("hoi_cm", "obj_cm", "baseline_hoi_cm", "baseline_obj_cm", "jitter", "correction_cm"):
            assert together[key] == pytest.approx(alone[key], rel=1e-5, abs=1e-5)
    for index, record in enumerate(records):
        one = runner.collate([record], torch.device("cpu"), torch.zeros(99), torch.ones(99))
        with torch.no_grad():
            isolated, _ = runner.corrected_object(model, one)
        torch.testing.assert_close(predicted[index, :len(record["features"])], isolated[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("architecture", ["pointwise", "gru"])
def test_saved_nonzero_checkpoint_runtime_matches_training_full_and_contiguous_chunks(development, architecture):
    """Audit the runtime boundary with real checkpoint IO and nontrivial scales."""
    from model.rnn.contact_residual_runtime import ContactResidualPredictor

    data = development
    record = make_online_record(data.source / "runtime_only.pt", "runtime_only_00", 137, sensor_shift=0.6)
    prepared = runner.prepare_record(record)
    model, _ = make_nonzero_model([prepared], architecture)
    membership = {row["file"]: row["partition"] for row in data.manifest["rows"]}
    train_features = torch.cat([
        runner.prepare_record(item)["features"] for item in data.records
        if membership[Path(item["seq_file"]).name] == "train"
    ])
    mean = train_features.mean(0)
    std = train_features.std(0).clamp_min(0.03)
    assert mean.abs().sum() > 0 and not torch.equal(std, torch.ones_like(std))
    settings = dict(architecture=architecture, feature_dim=99, hidden_dim=8, max_correction=0.2)
    checkpoint_path = data.root / (architecture + "_runtime_parity.pt")
    runner.save_checkpoint(checkpoint_path, model, settings, mean, std, {"gain": 0.75, "epoch": 1})
    batch = runner.collate([prepared], torch.device("cpu"), mean, std)
    with torch.no_grad():
        expected, correction = runner.corrected_object(model, batch, gain=0.75)
    assert correction[0, 60].abs().sum() > 0  # A chunk boundary must not be a new initialization.

    # Runtime input has observations only, no canonical labels or GT contacts.
    observations = {
        key: value for key, value in record.items()
        if key in ("gt_human_imu", "gt_obj_imu", "root_trans_pred", "gating_weights")
        or key.startswith(("pred_", "stage1_"))
    }
    untouched = {key: value.clone() for key, value in observations.items()}
    full_runtime = ContactResidualPredictor(checkpoint_path, device="cpu")
    whole = full_runtime.predict(observations)
    torch.testing.assert_close(whole, expected[0], rtol=1e-5, atol=1e-6)
    stream = ContactResidualPredictor(checkpoint_path, device="cpu")
    outputs = [stream.predict({key: value[start:end] for key, value in observations.items()})
               for start, end in ((0, 60), (60, 61), (61, 120), (120, 137))]
    torch.testing.assert_close(torch.cat(outputs), whole, rtol=1e-5, atol=1e-6)
    stream.reset()
    torch.testing.assert_close(stream.predict(observations), whole, rtol=0, atol=0)
    for key in observations:
        assert torch.equal(observations[key], untouched[key])
