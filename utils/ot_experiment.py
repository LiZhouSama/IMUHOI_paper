"""Resolve modular HP/VC/OT checkpoints from a full-pipeline experiment manifest."""
from pathlib import Path


def resolve_pipeline_files(manifest, variant, tag, project_root):
    """Return config path and module paths using the evaluator's stage fallbacks.

    A joint run may fine-tune HP (IMUHOI) or VC (v2/current). Prefer the
    checkpoint written by that run and otherwise use the manifest's reused
    module. The OT checkpoint always belongs to the selected variant.
    """
    if tag not in ("best", "final"):
        raise ValueError(f"Invalid checkpoint tag: {tag}")
    pipeline = manifest["pipelines"][variant]
    root = Path(project_root)

    def path(value):
        value = Path(value).expanduser()
        return value if value.is_absolute() else root / value

    stage = path(pipeline["stage_dir"]) if pipeline.get("stage_dir") else None

    def stage_or(file_name, fallback):
        candidate = stage / file_name if stage is not None else None
        return candidate if candidate is not None and candidate.is_file() else path(fallback)

    reused = manifest["reused_checkpoints"]
    paths = {
        "human_pose": stage_or(f"{tag}_human_pose.pt", reused["hp_ckpt"]),
        "velocity_contact": stage_or(
            f"{tag}_velocity_contact.pt",
            pipeline["joint_vc_ckpt"] if tag == "best" else reused["vc_ckpt"],
        ),
        "object_trans": stage_or(f"{tag}_object_trans.pt", pipeline["joint_ot_ckpt"]),
    }
    return path(pipeline["config"]), paths, pipeline
