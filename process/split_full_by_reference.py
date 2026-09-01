"""Extend an existing train/test split with newly recovered sequences."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def sequence_name_key(name: str) -> Tuple[int, int, str]:
    stem = Path(name).stem
    return (0, int(stem), "") if stem.isdigit() else (1, 0, name)


def _load_reference_split(path: Path) -> Tuple[List[str], List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        split_info = json.load(handle)

    train_files = list(split_info.get("train_files", []))
    test_files = list(split_info.get("test_files", []))
    all_files = train_files + test_files
    declared_total = int(split_info.get("total_files", len(all_files)))
    if len(all_files) != declared_total:
        raise ValueError(
            f"Reference split declares {declared_total} files but lists {len(all_files)}"
        )
    if len(set(all_files)) != len(all_files):
        raise ValueError("Reference train/test lists contain duplicates")
    return train_files, test_files


def _prepare_target_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing_files = list(path.glob("*.pt"))
    if existing_files and not overwrite:
        raise FileExistsError(
            f"{path} already contains {len(existing_files)} .pt files; use --overwrite"
        )
    for file_path in existing_files:
        file_path.unlink()


def _materialize_files(
    source_dir: Path,
    target_dir: Path,
    names: Sequence[str],
    hardlink: bool,
) -> None:
    for name in names:
        source = source_dir / name
        target = target_dir / name
        if hardlink:
            os.link(source, target)
        else:
            shutil.copy2(source, target)


def extend_reference_split(
    unified_dir: str,
    reference_split_path: str,
    output_root: str,
    recovered_train_ratio: float = 0.8,
    seed: int = 42,
    hardlink: bool = False,
    overwrite: bool = False,
) -> Dict[str, object]:
    if not 0.0 <= recovered_train_ratio <= 1.0:
        raise ValueError("recovered_train_ratio must be in [0, 1]")

    unified_path = Path(unified_dir).resolve()
    reference_path = Path(reference_split_path).resolve()
    output_path = Path(output_root).resolve()
    if not unified_path.is_dir():
        raise FileNotFoundError(f"Unified directory does not exist: {unified_path}")

    reference_train, reference_test = _load_reference_split(reference_path)
    reference_names = set(reference_train) | set(reference_test)
    full_names = {
        path.name for path in unified_path.iterdir() if path.is_file() and path.suffix == ".pt"
    }
    if not full_names:
        raise FileNotFoundError(f"No .pt files found under {unified_path}")

    missing_reference = sorted(reference_names - full_names, key=sequence_name_key)
    if missing_reference:
        raise ValueError(
            f"Full dataset is missing {len(missing_reference)} reference files: "
            f"{missing_reference[:10]}"
        )

    recovered_files = sorted(full_names - reference_names, key=sequence_name_key)
    shuffled_recovered = list(recovered_files)
    random.Random(seed).shuffle(shuffled_recovered)
    recovered_train_count = int(round(len(shuffled_recovered) * recovered_train_ratio))
    recovered_train = sorted(
        shuffled_recovered[:recovered_train_count], key=sequence_name_key
    )
    recovered_test = sorted(
        shuffled_recovered[recovered_train_count:], key=sequence_name_key
    )

    train_files = sorted(reference_train + recovered_train, key=sequence_name_key)
    test_files = sorted(reference_test + recovered_test, key=sequence_name_key)
    if set(train_files) & set(test_files):
        raise AssertionError("Generated train/test lists overlap")
    if set(train_files) | set(test_files) != full_names:
        raise AssertionError("Generated train/test lists do not cover the full dataset")

    train_dir = output_path / "train"
    test_dir = output_path / "test"
    _prepare_target_dir(train_dir, overwrite=overwrite)
    _prepare_target_dir(test_dir, overwrite=overwrite)
    _materialize_files(unified_path, train_dir, train_files, hardlink=hardlink)
    _materialize_files(unified_path, test_dir, test_files, hardlink=hardlink)

    total_files = len(full_names)
    split_info: Dict[str, object] = {
        "train_ratio": len(train_files) / total_files,
        "test_ratio": len(test_files) / total_files,
        "total_files": total_files,
        "split_method": "reference_plus_random_recovered",
        "source_dir": str(unified_path),
        "source_split_info": str(reference_path),
        "reference_total_files": len(reference_names),
        "reference_train_files_count": len(reference_train),
        "reference_test_files_count": len(reference_test),
        "recovered_total_files": len(recovered_files),
        "requested_recovered_train_ratio": recovered_train_ratio,
        "recovered_train_ratio": (
            len(recovered_train) / len(recovered_files) if recovered_files else 0.0
        ),
        "seed": seed,
        "skipped_files": [],
        "recovered_train_files": recovered_train,
        "recovered_test_files": recovered_test,
        "train_files": train_files,
        "test_files": test_files,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "split_info.json").open("w", encoding="utf-8") as handle:
        json.dump(split_info, handle, ensure_ascii=False, indent=2)
    shutil.copy2(reference_path, output_path / "split_info_reference.json")
    skip_manifest_path = unified_path / "skipped_sequences.json"
    if skip_manifest_path.is_file():
        shutil.copy2(skip_manifest_path, output_path / "skipped_sequences.json")

    return split_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-dir", required=True)
    parser.add_argument("--reference-split", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--recovered-train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Hard-link train/test files to unified/ instead of copying them",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing .pt files in output-root/train and output-root/test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_info = extend_reference_split(
        unified_dir=args.unified_dir,
        reference_split_path=args.reference_split,
        output_root=args.output_root,
        recovered_train_ratio=args.recovered_train_ratio,
        seed=args.seed,
        hardlink=args.hardlink,
        overwrite=args.overwrite,
    )
    print(
        f"train {len(split_info['train_files'])}, "
        f"test {len(split_info['test_files'])}, "
        f"recovered {split_info['recovered_total_files']}, "
        f"seed {split_info['seed']}"
    )


if __name__ == "__main__":
    main()
