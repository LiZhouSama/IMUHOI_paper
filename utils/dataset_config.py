"""Helpers for resolving dataset-specific evaluation paths from YAML config."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict


def _as_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"Expected a mapping, got {type(value)!r}")


def get_dataset_configs(config: Any) -> Dict[str, Dict[str, Any]]:
    """Return normalized dataset configs with ``data_dir`` aliased to ``test_path``.

    Dataset names are the keys under the YAML ``datasets`` section.  The
    ``data_dir`` alias keeps older evaluation helpers compatible while the
    source of truth remains the YAML ``test_path`` field.
    """
    if isinstance(config, Mapping):
        datasets = config.get("datasets")
    else:
        datasets = getattr(config, "datasets", None)

    dataset_mapping = _as_mapping(datasets)
    if not dataset_mapping:
        raise ValueError("Config must define at least one dataset under 'datasets'.")

    normalized: Dict[str, Dict[str, Any]] = {}
    for name, raw_config in dataset_mapping.items():
        dataset_config = _as_mapping(raw_config)
        test_path = dataset_config.get("test_path") or dataset_config.get("data_dir")
        if test_path:
            dataset_config["test_path"] = test_path
            dataset_config["data_dir"] = test_path
        normalized[str(name)] = dataset_config
    return normalized


def resolve_dataset_path(path_like: Any, project_root: Path) -> Path:
    """Resolve a YAML path relative to the repository root."""
    if path_like is None or str(path_like).strip() == "":
        raise ValueError("Dataset path is not configured.")

    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
