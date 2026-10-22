"""One variant directory = one self-contained visualization model specification."""
from dataclasses import dataclass
import json
from pathlib import Path

from utils.ot_experiment import resolve_pipeline_files


@dataclass(frozen=True)
class VisModelSpec:
    label: str
    config_path: Path
    checkpoints: dict
    object_trans_arch: str
    velocity_contact_arch: str


def resolve_vis_models(directories, project_root):
    """Resolve repeated ``--model`` directories using their parent manifest."""
    root = Path(project_root)
    result = []
    seen = set()
    for directory in directories:
        folder = Path(directory).expanduser()
        folder = (folder if folder.is_absolute() else root / folder).resolve()
        label = folder.name
        if label in seen:
            raise ValueError(f"Duplicate model label {label!r}; model directory names must differ")
        manifest_path = folder.parent / "experiment_manifest.json"
        if not (folder / "config.yaml").is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Model requires {folder / 'config.yaml'} and {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if label not in manifest.get("pipelines", {}):
            raise ValueError(f"{label!r} is not a pipeline in {manifest_path}")
        config_path, checkpoints, pipeline = resolve_pipeline_files(manifest, label, "final", root)
        if config_path.resolve() != (folder / "config.yaml").resolve():
            raise ValueError(f"{label}: manifest config {config_path} does not match {folder / 'config.yaml'}")
        for name, checkpoint in checkpoints.items():
            if not checkpoint.is_file():
                raise FileNotFoundError(f"{label}: {name} checkpoint not found: {checkpoint}")
        result.append(VisModelSpec(
            label, config_path, checkpoints,
            str(pipeline.get("object_trans_arch") or manifest.get("object_trans_arch") or "current"),
            str(pipeline.get("velocity_contact_arch") or manifest.get("velocity_contact_arch") or "v4"),
        ))
        seen.add(label)
    return result
