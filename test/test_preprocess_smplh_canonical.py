import ast
import importlib
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch


MODULE_NAMES = (
    "process.preprocess_BEHAVE",
    "process.preprocess_IMHD",
    "process.preprocess_HODOME",
)


def _module_source(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return Path(module.__file__).read_text(encoding="utf-8")


def _output_dict_keys(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "output" for target in node.targets):
            continue
        if isinstance(node.value, ast.Dict):
            yield {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_body_model_loader_is_one_fixed_neutral_smplh_instance(
    module_name, tmp_path, monkeypatch
):
    module = importlib.import_module(module_name)
    model_path = tmp_path / "smplh" / "neutral" / "model.npz"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"mock")

    created = []

    class FakeParameter:
        def __init__(self):
            self.requires_grad = True

        def requires_grad_(self, value):
            self.requires_grad = bool(value)
            return self

    class FakeBodyModel:
        def __init__(self, bm_fname, num_betas):
            self.bm_fname = os.path.normpath(str(bm_fname))
            self.num_betas = int(num_betas)
            self.parameter = FakeParameter()
            self.eval_called = False
            self.device = None
            created.append(self)

        def eval(self):
            self.eval_called = True
            return self

        def parameters(self):
            return [self.parameter]

        def to(self, device):
            self.device = torch.device(device)
            return self

    monkeypatch.setattr(module, "BodyModel", FakeBodyModel)
    loader = module.BodyModelLoader(str(tmp_path), torch.device("cpu"), num_betas=16)

    expected_path = os.path.normpath(str(model_path))
    assert len(created) == 1
    assert created[0].bm_fname == expected_path
    assert created[0].eval_called
    assert created[0].parameter.requires_grad is False
    assert created[0].device == torch.device("cpu")

    # Source metadata must not select another body-model instance.
    assert loader.get(gender="female", model_type="smplh") is created[0]
    resolved, model_type = loader.resolve(
        {"gender": "male", "betas": np.ones(16, dtype=np.float32)},
        gender="male",
        model_type="smplh",
    )
    assert resolved is created[0]
    assert model_type == "smplh"
    assert len(created) == 1


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_preprocessor_has_zero_shape_metadata_and_no_saved_betas(module_name):
    module = importlib.import_module(module_name)
    assert module.canonical_body_model_metadata() == {
        "canonical_body_model_gender": "neutral",
        "canonical_betas": "zero",
        "canonical_skeleton_version": 1,
    }

    source = _module_source(module_name)
    tree = ast.parse(source)
    output_dicts = list(_output_dict_keys(tree))
    assert output_dicts
    assert all("betas" not in keys for keys in output_dicts)

    output_nodes = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "output"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    ]
    assert output_nodes
    assert all(
        any(
            key is None
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "canonical_body_model_metadata"
            for key, value in zip(output_node.keys, output_node.values)
        )
        for output_node in output_nodes
    )
    assert "GenderedBodyModelLoader" not in source
    assert "smplShape" not in source

    # No BodyModel forward is allowed to receive a shape override. Omitting
    # betas makes human_body_prior use the model's zero-valued init_betas.
    bm_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bm"
    ]
    assert bm_calls
    assert all(
        all(keyword.arg != "betas" for keyword in call.keywords)
        for call in bm_calls
    )


@pytest.mark.parametrize(
    "module_name",
    ("process.preprocess_BEHAVE", "process.preprocess_IMHD"),
)
def test_smpl_input_builder_never_includes_betas(module_name):
    module = importlib.import_module(module_name)
    root_orient = np.zeros((2, 3), dtype=np.float32)
    pose_body = np.zeros((2, 63), dtype=np.float32)
    trans = np.zeros((2, 3), dtype=np.float32)
    smpl_input = module.build_smpl_input(
        root_orient, pose_body, trans, torch.device("cpu")
    )
    assert set(smpl_input) == {"root_orient", "pose_body", "trans"}


def test_imhd_source_shape_is_not_required(tmp_path):
    module = importlib.import_module("process.preprocess_IMHD")
    record_path = tmp_path / "gt_0_0_3.pkl"
    record = {
        "objectRot": np.zeros((4, 3), dtype=np.float32),
        "objectTrans": np.zeros((4, 3), dtype=np.float32),
        "smplPose": np.zeros((4, 66), dtype=np.float32),
        "smplTrans": np.zeros((4, 3), dtype=np.float32),
    }
    with record_path.open("wb") as handle:
        pickle.dump(record, handle)

    data, human_imu, object_imu = module.load_imhd_sequence([str(record_path)])
    assert "smplShape" not in data
    assert data["smplPose"].shape == (4, 66)
    assert human_imu is None
    assert object_imu is None


def test_hodome_source_gender_and_betas_are_not_required(tmp_path):
    module = importlib.import_module("process.preprocess_HODOME")
    human_path = tmp_path / "subject_object_human.npz"
    object_path = tmp_path / "subject_object_object.npz"

    poses = np.arange(4 * 156, dtype=np.float32).reshape(4, 156)
    np.savez(
        human_path,
        poses=poses,
        Rh=np.zeros((4, 1, 3), dtype=np.float32),
        Th=np.zeros((4, 1, 3), dtype=np.float32),
        mocap_frame_rate=np.array(60),
    )
    np.savez(
        object_path,
        object_R=np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0),
        object_T=np.zeros((4, 1, 3), dtype=np.float32),
        mocap_frame_rate=np.array(60),
    )

    sequence = module.load_hodome_npz_pair(
        "subject_object",
        human_path,
        object_path,
        target_fps=30,
    )
    assert sequence["gender"] == "neutral"
    assert "betas" not in sequence
    np.testing.assert_array_equal(sequence["source_frame_indices"], np.array([0, 2]))
    np.testing.assert_array_equal(sequence["pose_body"], poses[[0, 2], 3:66])


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_canonical_joints_feed_floor_contact_and_derived_features(module_name):
    source = _module_source(module_name)
    tree = ast.parse(source)
    derived_feature_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "precompute_sequence_features"
    ]
    assert any(
        call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "position_global"
        for call in derived_feature_calls
    )
    assert "compute_foot_contact_labels" in source
    assert "compute_improved_contact_labels" in source
    assert "position_global_full_gt_world" in source
