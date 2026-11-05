"""External human-pose replay for the HP plug-in experiment (model/rnn/external_human_pose.py)."""
import glob
import json
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict as edict
from pytorch3d.transforms import matrix_to_rotation_6d, random_rotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs import load_config  # noqa: E402
from model.rnn.external_human_pose import (  # noqa: E402
    ExternalHumanPoseModule,
    global_to_local_rotations,
    local_to_global_rotations,
    maybe_build_external_human_pose,
)
from utils.external_hp_cache import (  # noqa: E402
    CACHE_FORMAT,
    CACHE_VERSION,
    attach_external_hp_inputs,
    external_hp_cache_key,
    write_cache_entry,
)


def _write_manifest(cache_dir: Path, method="tip", latency=0):
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(json.dumps({
        "format": CACHE_FORMAT, "version": CACHE_VERSION, "status": "complete",
        "method": method, "dataset": "omomo", "latency_frames": latency,
    }))


def _fake_sequence(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "data" / "process" / "IMUHOI++cano" / "split" / "train" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _random_motion(frames: int, seed: int = 0):
    """Plausible local rotations (small joint angles, large root orientation) and a root path."""
    generator = torch.Generator().manual_seed(seed)
    scale = torch.full((24, 1), 0.2)
    scale[0] = 2.0
    local = torch.linalg.matrix_exp(_skew(scale * torch.randn(frames, 24, 3, generator=generator)))
    root = torch.cumsum(0.01 * torch.randn(frames, 3, generator=generator), dim=0)
    return local, root


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (torch.stack((zero, -z, y), -1), torch.stack((z, zero, -x), -1), torch.stack((-y, x, zero), -1)), -2
    )


def _cfg(cache_dir: Path, method=None):
    cfg = load_config(str(PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml"))
    cfg.external_hp_cache_dir = str(cache_dir)
    if method:
        cfg.external_hp_method = method
    return cfg


@pytest.fixture
def project_cwd(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)  # body model paths in the config are repository-relative


def test_cache_key_is_relative_to_process(tmp_path):
    path = _fake_sequence(tmp_path, "a.pt")
    assert external_hp_cache_key(path) == "IMUHOI++cano/split/train/a.pt"
    link = tmp_path / "elsewhere"
    link.symlink_to(tmp_path / "data" / "process")
    assert external_hp_cache_key(link / "IMUHOI++cano/split/train/a.pt") == "IMUHOI++cano/split/train/a.pt"
    with pytest.raises(ValueError):
        external_hp_cache_key(tmp_path / "no_anchor.pt")


def test_attach_slices_windows_and_repeat_pads(tmp_path):
    cache_dir = tmp_path / "cache"
    _write_manifest(cache_dir)
    seq = _fake_sequence(tmp_path, "a.pt")
    frames = 8
    rot6d = torch.randn(frames, 24, 6)
    root = torch.cumsum(torch.randn(frames, 3), dim=0)
    write_cache_entry(cache_dir, seq, rot6d, root)
    batch = {
        "seq_path": [str(seq), str(seq)],
        "window_start": torch.tensor([1, 0]),
        "window_end": torch.tensor([6, 3]),
    }
    out = attach_external_hp_inputs(batch, edict(external_hp_cache_dir=str(cache_dir)), "cpu", torch.float32, 2, 5)
    assert out["ext_hp_local_rot6d"].shape == (2, 5, 24, 6)
    assert torch.equal(out["ext_hp_local_rot6d"][0], rot6d[1:6])
    disp = root[1:] - root[:-1]
    assert torch.allclose(out["ext_hp_root_disp"][0], disp[0:5])
    # Second sample: frames 0..2, then repeat padding of frame 2.
    assert torch.equal(out["ext_hp_local_rot6d"][1, :3], rot6d[:3])
    assert torch.equal(out["ext_hp_local_rot6d"][1, 3:], rot6d[2:3].expand(2, -1, -1))
    assert torch.allclose(out["ext_hp_root_disp"][1, 0], disp[0])  # d[0] repeats d[1]
    with pytest.raises(KeyError):
        attach_external_hp_inputs({**batch, "seq_path": [str(_fake_sequence(tmp_path, "b.pt"))] * 2},
                                  edict(external_hp_cache_dir=str(cache_dir)), "cpu", torch.float32, 2, 5)
    assert attach_external_hp_inputs(batch, edict(), "cpu", torch.float32, 2, 5) == {}


def test_rotation_chain_round_trip():
    local = random_rotations(5 * 24).reshape(5, 24, 3, 3)
    assert torch.allclose(global_to_local_rotations(local_to_global_rotations(local)), local, atol=1e-5)


@pytest.mark.parametrize(
    "pattern",
    [
        "process/IMUHOI++cano/processed_split_data_OMOMO_full/test/*.pt",  # SMPL-X
        "process/IMUHOI++cano/processed_split_data_BEHAVE/test/*.pt",      # SMPL-H
    ],
)
def test_ground_truth_pose_reproduces_ground_truth_joints(tmp_path, project_cwd, pattern):
    """GT local pose + GT root through the replay must give the corpus' GT joints and palms."""
    files = sorted(glob.glob(str(PROJECT_ROOT / pattern)))
    if not files:
        pytest.skip(f"no data for {pattern}")
    data = torch.load(files[0], map_location="cpu")
    frames = min(90, int(data["trans"].shape[0]))
    global_rot = torch.as_tensor(data["rotation_global"][:frames, :24]).float()
    trans = torch.as_tensor(data["trans"][:frames]).float().reshape(frames, 3)
    joints_gt = torch.as_tensor(data["position_global_full_gt_world"][:frames, :24]).float()
    cache_dir = tmp_path / "cache"
    _write_manifest(cache_dir)
    local = global_to_local_rotations(global_rot)
    module = ExternalHumanPoseModule(_cfg(cache_dir), torch.device("cpu"))
    out = module({
        "ext_hp_local_rot6d": matrix_to_rotation_6d(local).unsqueeze(0),
        "ext_hp_root_disp": torch.cat((trans[1:2] - trans[:1], trans[1:] - trans[:-1])).unsqueeze(0),
        "trans_init": trans[:1],
        "body_model_type": [str(data.get("body_model_type") or "")],
    })
    assert torch.allclose(out["root_trans_pred"][0], trans, atol=1e-5)
    error = (out["pred_joints_global"][0] - joints_gt).norm(dim=-1)
    assert error.max().item() < 1e-3, f"max joint error {error.max().item():.2e} m"
    assert torch.allclose(out["pred_full_pose_rotmat"][0], global_rot, atol=1e-5)


def test_online_finalize_matches_window_root(tmp_path, project_cwd):
    cache_dir = tmp_path / "cache"
    _write_manifest(cache_dir)
    module = ExternalHumanPoseModule(_cfg(cache_dir), torch.device("cpu"))
    local, root = _random_motion(12)
    disp = torch.cat((root[1:2] - root[:1], root[1:] - root[:-1]))
    window = module({
        "ext_hp_local_rot6d": matrix_to_rotation_6d(local).unsqueeze(0),
        "ext_hp_root_disp": disp.unsqueeze(0),
        "trans_init": torch.tensor([[0.3, 0.9, -0.2]]),
    })
    assert torch.allclose(window["root_trans_pred"][0] - window["root_trans_pred"][0, :1], root - root[:1], atol=1e-5)
    history = {"root_trans_pred": window["root_trans_pred"][:, :7]}
    frame = {key: value[:, 7:8] for key, value in window.items()}
    frame["root_trans_pred"] = frame["root_trans_pred"] + 5.0  # finalize must not trust the window root
    final = module.finalize_online_frame(frame, history)
    assert torch.allclose(final["root_trans_pred"], window["root_trans_pred"][:, 7:8], atol=1e-5)
    assert torch.allclose(final["pred_palm_glb_pos"], window["pred_palm_glb_pos"][:, 7:8], atol=1e-5)


def test_latency_holds_the_unavailable_tail(tmp_path, project_cwd):
    cache_dir = tmp_path / "cache"
    _write_manifest(cache_dir, method="transpose", latency=5)
    module = ExternalHumanPoseModule(_cfg(cache_dir, method="transpose"), torch.device("cpu"))
    local, root = _random_motion(20, seed=1)
    disp = torch.cat((root[1:2] - root[:1], root[1:] - root[:-1]))
    out = module({
        "ext_hp_local_rot6d": matrix_to_rotation_6d(local).unsqueeze(0),
        "ext_hp_root_disp": disp.unsqueeze(0),
        "trans_init": root[:1],
    })
    last = 20 - 1 - 5
    assert torch.allclose(out["pred_full_pose_6d"][0, last + 1:], out["pred_full_pose_6d"][0, last:last + 1].expand(5, -1, -1))
    assert torch.allclose(out["root_trans_pred"][0, : last + 1], root[: last + 1], atol=1e-5)
    step = root[last] - root[last - 1]
    expected_tail = root[last] + step * torch.arange(1, 6).unsqueeze(-1)
    assert torch.allclose(out["root_trans_pred"][0, last + 1:], expected_tail, atol=1e-5)


def test_refine_derivation_is_identity_without_correction(tmp_path, project_cwd):
    cache_dir = tmp_path / "cache"
    _write_manifest(cache_dir)
    module = ExternalHumanPoseModule(_cfg(cache_dir), torch.device("cpu"))
    local, root = _random_motion(10, seed=2)
    disp = torch.cat((root[1:2] - root[:1], root[1:] - root[:-1]))
    out = module({
        "ext_hp_local_rot6d": matrix_to_rotation_6d(local).unsqueeze(0),
        "ext_hp_root_disp": disp.unsqueeze(0),
        "trans_init": root[:1],
    })
    results = dict(out)
    results["stage1_pred_full_pose_rotmat"] = out["pred_full_pose_rotmat"]
    module.derive_refined_human_outputs(results, {"body_model_type": None})
    for key in ("pred_joints_global", "pred_palm_glb_pos", "pred_full_pose_6d"):
        assert torch.allclose(results[key], out[key], atol=1e-5), key
    # A refined root shifts every joint by the same offset.
    results["root_trans_pred"] = out["root_trans_pred"] + torch.tensor([0.1, 0.0, 0.0])
    module.derive_refined_human_outputs(results, {"body_model_type": None})
    assert torch.allclose(results["pred_palm_glb_pos"] - out["pred_palm_glb_pos"],
                          torch.tensor([0.1, 0.0, 0.0]).expand_as(out["pred_palm_glb_pos"]), atol=1e-5)


def test_builder_requires_matching_cache_and_checkpoint(tmp_path, project_cwd):
    cache_dir = tmp_path / "cache"
    _write_manifest(cache_dir)
    cfg = _cfg(cache_dir)
    assert isinstance(maybe_build_external_human_pose(cfg, "cpu", cache_dir / "manifest.json"), ExternalHumanPoseModule)
    with pytest.raises(ValueError):
        maybe_build_external_human_pose(cfg, "cpu", tmp_path / "human_pose.pt")
    other = tmp_path / "other"
    _write_manifest(other)
    with pytest.raises(ValueError):
        maybe_build_external_human_pose(cfg, "cpu", other / "manifest.json")
    plain = load_config(str(PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml"))
    assert maybe_build_external_human_pose(plain, "cpu", tmp_path / "human_pose.pt") is None
    with pytest.raises(ValueError):
        maybe_build_external_human_pose(plain, "cpu", cache_dir / "manifest.json")
    with pytest.raises(ValueError):
        maybe_build_external_human_pose(_cfg(cache_dir, method="dip18"), "cpu", cache_dir / "manifest.json")
