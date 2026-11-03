"""The per-sequence IMU noise key must not depend on where the project is checked out."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils.utils as U  # noqa: E402

REF = U.NOISE_KEY_REFERENCE_ROOT
BEHAVE_FILE = "process/IMUHOI++cano/processed_split_data_BEHAVE/test/Date07_Sub08_backpack_hug.pt"
# Seed of that sequence in every published BEHAVE run (simulate_imu_noise_seed 42, reference checkout).
PUBLISHED_SEED = 5689844386628324063


@pytest.fixture
def off_reference(monkeypatch):
    monkeypatch.setattr(U, "_PROJECT_ROOT_REAL", "/workspace/IMUHOI")
    monkeypatch.delenv("IMUHOI_NOISE_KEY_MODE", raising=False)


def test_reference_checkout_keeps_real_path(monkeypatch, tmp_path):
    monkeypatch.setattr(U, "_PROJECT_ROOT_REAL", REF)
    f = tmp_path / "process" / "x.pt"
    assert U.sequence_noise_key(str(f)) == os.path.realpath(str(f))
    assert U.sequence_noise_key(f"{REF}/{BEHAVE_FILE}") == f"{REF}/{BEHAVE_FILE}"


def test_other_checkout_is_rerooted_to_published_seed(off_reference):
    key = U.sequence_noise_key(f"/workspace/IMUHOI/{BEHAVE_FILE}")
    assert key == f"{REF}/{BEHAVE_FILE}"
    assert U._stable_sequence_seed(42, key) == PUBLISHED_SEED
    # The old behaviour (server real path) drew a different realization.
    assert U._stable_sequence_seed(42, f"/workspace/IMUHOI/{BEHAVE_FILE}") != PUBLISHED_SEED


def test_symlinked_corpus_resolves_before_rerooting(off_reference, tmp_path):
    corpus = tmp_path / "main" / "process" / "IMUHOI++cano" / "test"
    corpus.mkdir(parents=True)
    (corpus / "a.pt").write_bytes(b"0")
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "process").symlink_to(tmp_path / "main" / "process")
    key = U.sequence_noise_key(str(dev / "process" / "IMUHOI++cano" / "test" / "a.pt"))
    assert key == f"{REF}/process/IMUHOI++cano/test/a.pt"


def test_path_without_anchor_and_native_mode_keep_real_path(off_reference, monkeypatch, tmp_path):
    f = tmp_path / "elsewhere" / "a.pt"
    assert U.sequence_noise_key(str(f)) == os.path.realpath(str(f))
    monkeypatch.setenv("IMUHOI_NOISE_KEY_MODE", "native")
    p = f"/workspace/IMUHOI/{BEHAVE_FILE}"
    assert U.sequence_noise_key(p) == p


def test_dataset_noise_is_checkout_independent(off_reference, monkeypatch, tmp_path):
    """Same file content under two checkouts -> bit-identical noisy IMU."""
    import torch
    import dataset.dataset_IMUHOI as D

    imu = torch.randn(50, 6, 9)
    outs = []
    for root in ("ckpt_a", "ckpt_b"):
        path = tmp_path / root / "process" / "IMUHOI++cano" / "s.pt"
        path.parent.mkdir(parents=True)
        ds = D.IMUDataset.__new__(D.IMUDataset)
        ds.simulate_imu_noise = True
        ds.simulate_imu_noise_seed = 42
        ds.imu_noise_cfg = D.merge_noise_cfg(None)
        seq = {"human_imu": imu.clone(), "obj_imu": imu[:, :1].clone()}
        ds._apply_sequence_noise(seq, str(path))
        outs.append(seq["human_imu"])
    assert torch.equal(outs[0], outs[1])
    assert not torch.equal(outs[0], imu)
