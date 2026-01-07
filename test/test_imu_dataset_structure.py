"""
Tests the structure of the raw IMU pickle file provided in the dataset.
It validates the presence of the expected keys and the shapes/dtypes of the
stored numpy arrays.
"""

import pickle

import numpy as np
from pathlib import Path
from typing import List


_RELATIVE_DATASET_PARTS = [
    "IMHD",
    "IMHD Dataset",
    "imu_preprocessed",
    "20230825",
    "20230825_songzn_bat",
    "bat_holdhandle_hit",
    "imu_0_330_-1.pkl",
]


def _candidate_paths() -> List[Path]:
    """Yield possible file locations to handle the datasets symlink on Windows."""
    test_file = Path(__file__).resolve()
    candidates = []

    # 1) Typical layout: repository root / datasets / ...
    repo_root = test_file.parents[1]
    candidates.append(repo_root / "datasets" / Path(*_RELATIVE_DATASET_PARTS))

    # 2) Walk up the tree to find a top-level datasets directory (the repo uses a symlink).
    for parent in test_file.parents:
        candidate = parent / "datasets" / Path(*_RELATIVE_DATASET_PARTS)
        if candidate not in candidates:
            candidates.append(candidate)

    return candidates


def _resolve_dataset_path() -> Path:
    """Return the first accessible dataset path."""
    for candidate in _candidate_paths():
        try:
            if candidate.exists():
                return candidate
        except OSError:
            # Some Windows symlinks may raise on exists(); skip to the next candidate.
            continue
    raise FileNotFoundError("Cannot locate imu_0_330_-1.pkl via known dataset paths.")


def test_imu_pickle_structure():
    imu_path = _resolve_dataset_path()
    with open(imu_path, "rb") as f:
        data = pickle.load(f)

    assert isinstance(data, dict), "IMU pickle should contain a dict."

    expected = {
        "objectImuOri": ((3677, 3), np.float64),
        "objectImuAcc": ((3677, 3), np.float64),
    }

    assert set(data.keys()) == set(expected.keys())

    for key, (shape, dtype) in expected.items():
        assert key in data, f"Missing key: {key}"
        value = data[key]
        assert isinstance(value, np.ndarray), f"{key} should be a numpy array."
        assert value.shape == shape, f"{key} shape mismatch: {value.shape} != {shape}"
        assert value.dtype == dtype, f"{key} dtype mismatch: {value.dtype} != {dtype}"


if __name__ == "__main__":
    # Simple inspection printout for manual runs.
    path = _resolve_dataset_path()
    print(f"Found file: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"type: {type(data)}")
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"{k}: type={type(v)}, shape={getattr(v, 'shape', None)}")
    else:
        print("Unexpected data type; expected dict.")
