"""Helpers for resolving dataset-specific evaluation paths from YAML config."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from utils.human_pose import is_neutral_body_model_path, normalize_body_model_type


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


def get_full_sequence_test_path(dataset_config: Any) -> Any:
    """Select the original-sequence test split, with legacy fallback."""
    dataset_config = _as_mapping(dataset_config)
    return (
        dataset_config.get("full_sequence_test_path")
        or dataset_config.get("test_path")
        or dataset_config.get("data_dir")
    )


def resolve_dataset_path(path_like: Any, project_root: Path) -> Path:
    """Resolve a YAML path relative to the repository root."""
    if path_like is None or str(path_like).strip() == "":
        raise ValueError("Dataset path is not configured.")

    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def configure_canonical_body_model_for_datasets(
    config: Any,
    dataset_names: Iterable[str],
) -> Tuple[Optional[str], Tuple[str, ...]]:
    """Select the neutral model family used by the active training datasets.

    RNN supports mixed SMPL-H/SMPL-X batches through ``body_model_paths``. The
    other model families currently own one BodyModel, so mixed-family training
    is rejected instead of silently using the wrong skeleton.
    """
    datasets = get_dataset_configs(config)
    names = [str(name) for name in dataset_names]
    families = []
    for name in names:
        if name not in datasets:
            raise ValueError(f"Unknown training dataset {name!r}")
        family = normalize_body_model_type(datasets[name].get("body_model_type"))
        if family is None:
            raise ValueError(
                f"Dataset {name!r} must declare body_model_type for canonical training"
            )
        families.append(family)
    unique = tuple(sorted(set(families)))
    if not unique:
        raise ValueError("At least one training dataset is required")

    model_arch = str(
        config.get("model_arch", "rnn")
        if isinstance(config, Mapping)
        else getattr(config, "model_arch", "rnn")
    ).strip().lower()
    if len(unique) > 1 and model_arch != "rnn":
        raise ValueError(
            f"{model_arch} currently owns one BodyModel and cannot train mixed "
            f"canonical families {list(unique)}; train one family at a time."
        )
    if len(unique) > 1:
        return None, unique

    mapping = _as_mapping(
        config.get("body_model_paths")
        if isinstance(config, Mapping)
        else getattr(config, "body_model_paths", None)
    )
    normalized_paths = {
        normalize_body_model_type(raw_type): path
        for raw_type, path in mapping.items()
        if normalize_body_model_type(raw_type) is not None and path
    }
    family = unique[0]
    path = normalized_paths.get(family)
    if not path:
        raise ValueError(f"No canonical body_model_paths entry for {family!r}")
    if bool(
        config.get("require_canonical_skeleton", False)
        if isinstance(config, Mapping)
        else getattr(config, "require_canonical_skeleton", False)
    ) and not is_neutral_body_model_path(path):
        raise ValueError(f"Canonical body model must be neutral: {path}")

    if isinstance(config, Mapping):
        config["body_model_path"] = str(path)
    else:
        setattr(config, "body_model_path", str(path))
    return str(path), unique

def resolve_evaluation_body_model_path(
    config: Any,
    dataset_config: Any,
    sequence_info: Optional[Iterable[Dict[str, Any]]] = None,
    explicit_path: Optional[str] = None,
) -> Tuple[str, Optional[str], str]:
    """Resolve one homogeneous evaluation dataset's body-model path."""
    if explicit_path:
        return str(explicit_path), None, "cli"

    dataset_config = _as_mapping(dataset_config)
    dataset_path = dataset_config.get("body_model_path")
    if dataset_path:
        return str(dataset_path), normalize_body_model_type(
            dataset_config.get("body_model_type")
        ), "dataset.body_model_path"

    declared_raw = dataset_config.get("body_model_type")
    declared_type = normalize_body_model_type(declared_raw)
    if declared_raw and declared_type is None:
        raise ValueError(
            f"Unsupported dataset body_model_type: {declared_raw!r}"
        )

    sample_types = set()
    for info in sequence_info or ():
        raw_type = info.get("body_model_type")
        if not raw_type:
            continue
        model_type = normalize_body_model_type(raw_type)
        if model_type is None:
            raise ValueError(f"Unsupported sample body_model_type: {raw_type!r}")
        sample_types.add(model_type)
    if len(sample_types) > 1:
        raise ValueError(
            "A single evaluation run cannot mix body-model families: "
            f"{sorted(sample_types)}"
        )
    sample_type = next(iter(sample_types), None)
    if declared_type and sample_type and declared_type != sample_type:
        raise ValueError(
            "Dataset body_model_type does not match its samples: "
            f"{declared_type!r} != {sample_type!r}"
        )

    model_type = declared_type or sample_type
    mapping = _as_mapping(
        config.get("body_model_paths")
        if isinstance(config, Mapping)
        else getattr(config, "body_model_paths", None)
    )
    normalized_paths = {
        normalize_body_model_type(raw_type): path
        for raw_type, path in mapping.items()
        if normalize_body_model_type(raw_type) is not None and path
    }
    if model_type is not None:
        model_path = normalized_paths.get(model_type)
        if model_path:
            source = (
                "dataset.body_model_type"
                if declared_type is not None
                else "sample.body_model_type"
            )
            return str(model_path), model_type, source
        raise ValueError(
            f"No body_model_paths entry is configured for {model_type!r}"
        )

    fallback_path = (
        config.get("body_model_path")
        if isinstance(config, Mapping)
        else getattr(config, "body_model_path", None)
    )
    if not fallback_path:
        raise ValueError(
            "Cannot resolve a body model: set dataset.body_model_type, "
            "sample body_model_type, or config.body_model_path"
        )
    return str(fallback_path), None, "config.body_model_path"
