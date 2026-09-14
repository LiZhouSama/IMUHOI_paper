"""Lightweight split/path selection for OT dumps; deliberately no ML imports."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _resolve_path(value: Any, project_root: Path) -> Path:
    # Match utils.dataset_config.resolve_dataset_path without importing torch.
    if value is None or not str(value).strip():
        raise ValueError("Dataset path is not configured.")
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _train_path(config: Mapping[str, Any]) -> Any:
    if config.get("full_sequence_train_path"):
        return config["full_sequence_train_path"]
    full_test = config.get("full_sequence_test_path")
    if full_test and Path(str(full_test)).name == "test":
        # A declared original-sequence test directory identifies its train peer.
        return Path(str(full_test)).parent / "train"
    return config.get("train_path")


def resolve_dump_data_path(
    dataset_configs: Mapping[str, Mapping[str, Any]],
    dataset: str,
    project_root: Path,
    *,
    split: str = "test",
    data_path: str | Path | None = None,
) -> Path:
    """Resolve a full-sequence split, never falling back from train to test.

    OMOMO training prefers the configured ``omomo_full`` corpus. Test retains
    the evaluator's full_sequence_test_path > test_path > data_dir precedence.
    Overrides must be split-specific: known test roots cannot enter training,
    including through symlinks or an override containing both train and test.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split!r}")
    if dataset not in dataset_configs:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(dataset_configs)}")
    project_root = Path(project_root)
    config = dataset_configs[dataset]
    if data_path is not None:
        value = data_path
    elif split == "test":
        value = (
            config.get("full_sequence_test_path")
            or config.get("test_path")
            or config.get("data_dir")
        )
    else:
        # Do not use the segmented/window-training config for this cache.
        train_config = dataset_configs.get("omomo_full", config) if dataset == "omomo" else config
        value = _train_path(train_config)
        if not value:
            raise ValueError(f"No full-sequence train path configured for {dataset!r}.")
        if "processed_seg_data_" in str(value):
            raise ValueError("Segmented train data is not a full-sequence cache; set full_sequence_train_path or --data_path.")
    path = _resolve_path(value, project_root)
    if split == "train":
        for other_config in dataset_configs.values():
            for key in ("full_sequence_test_path", "test_path", "data_dir"):
                if not other_config.get(key):
                    continue
                test_path = _resolve_path(other_config[key], project_root)
                if path == test_path or test_path in path.parents or path in test_path.parents:
                    raise ValueError(f"Train path {path} overlaps configured test split {test_path}.")
    return path


def read_sequence_list(path: str | Path) -> list[str]:
    """Read UTF-8 filenames, one per line (blank/# lines ignored), or a JSON list.

    Relative entries are relative to the chosen split directory, NOT the list
    file's directory. Absolute paths are accepted only within that split.
    """
    path = Path(path).expanduser()
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        entries = json.loads(text)
        if not isinstance(entries, list) or any(not isinstance(item, str) for item in entries):
            raise ValueError("A JSON sequence list must be an array of file path strings.")
        entries = [item.strip() for item in entries]
    else:
        entries = [line.strip() for line in text.splitlines()]
        entries = [line for line in entries if line and not line.startswith("#")]
    if not entries or any(not entry for entry in entries):
        raise ValueError("Sequence list must contain at least one non-empty .pt file path.")
    return entries


def select_sequence_paths(
    data_path: str | Path,
    sequence_list: str | Path | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Validate selection before limiting it; never load tensors or randomize.

    Explicit lists retain their order; otherwise filenames are lexicographic.
    Every entry is validated, even beyond limit, to catch split leakage early.
    The caller keeps the legacy no-options test discovery path unchanged.
    """
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    root = Path(data_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    entries = (
        read_sequence_list(sequence_list)
        if sequence_list is not None
        else [path.name for path in sorted(root.glob("*.pt"))]
    )
    selected = []
    seen = set()
    for entry in entries:
        path = Path(entry).expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        if root not in path.parents:
            raise ValueError(f"Sequence {entry!r} is outside selected split {root}.")
        if path.suffix != ".pt" or not path.is_file():
            raise ValueError(f"Sequence is not an existing .pt file: {path}")
        if path in seen:
            raise ValueError(f"Duplicate sequence in selection: {path}")
        seen.add(path)
        selected.append(path)
    if not selected:
        raise ValueError(f"No .pt sequences selected from {root}.")
    return selected if limit is None else selected[:limit]
