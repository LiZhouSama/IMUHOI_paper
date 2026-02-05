"""Quick utility to report frame counts for each `.pt` file in the test splits."""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


# Absolute roots provided by the user.
DATASET_ROOTS: Dict[str, Path] = {
    "BEHAVE": Path(r"D:\a_WORK\Projects\PhD\tasks\EgoIMU\process\processed_split_data_BEHAVE"),
    "OMOMO": Path(r"D:\a_WORK\Projects\PhD\tasks\EgoIMU\process\processed_split_data_OMOMO"),
    "IMHD": Path(r"D:\a_WORK\Projects\PhD\tasks\EgoIMU\process\processed_split_data_IMHD"),
}

# Prefer keys where the first dimension corresponds to frames.
FRAME_KEYS: Tuple[str, ...] = (
    "rotation_local_full_gt_list",
    "position_global_full_gt_world",
    "rotation_global",
    "trans",
)


def _first_dim(value) -> int | None:
    """Return the leading dimension if the object exposes a shape."""
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    try:
        return int(shape[0])
    except Exception:
        return None


def frame_count_for_pt(pt_path: Path) -> int:
    """Load a .pt file and infer its frame count."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    for key in FRAME_KEYS:
        if key in data:
            dim0 = _first_dim(data[key])
            if dim0 is not None:
                return dim0

    # Fallback: pick the first array-like value with a shape.
    for value in data.values():
        dim0 = _first_dim(value)
        if dim0 is not None:
            return dim0

    raise ValueError(f"Could not determine frame count for {pt_path}")


def summarize_dataset(root: Path) -> List[Tuple[str, int]]:
    """Return (filename, frame_count) pairs for every .pt in root/test."""
    test_dir = root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    counts: List[Tuple[str, int]] = []
    for pt_file in sorted(test_dir.glob("*.pt")):
        frames = frame_count_for_pt(pt_file)
        counts.append((pt_file.name, frames))
    return counts


def main() -> None:
    for name, root in DATASET_ROOTS.items():
        try:
            counts = summarize_dataset(root)
        except FileNotFoundError as exc:
            print(f"{name}: {exc}")
            continue

        total_frames = sum(frames for _, frames in counts)
        print(f"{name} ({root})")
        for filename, frames in counts:
            print(f"  {filename}: {frames} frames")
        print(f"  Total frames: {total_frames}\n")


if __name__ == "__main__":
    main()
