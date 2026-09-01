"""Randomly re-split a unified preprocessing directory into train/ and test/.

The reference-split path in ``preprocess.py`` reproduces an existing split's
membership exactly, which leaves any newly recovered sequence unassigned.  This
script instead uses every sequence in ``unified/`` and draws a fresh random
split at a given ratio, so nothing is discarded.

The previous ``split_info.json`` is kept as ``split_info_reference.json`` so the
reference-based membership stays recoverable.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil


def numeric_key(name: str):
    """Sort 0.pt..N.pt numerically, everything else lexicographically."""
    stem = os.path.splitext(name)[0]
    return (0, int(stem), "") if stem.isdigit() else (1, 0, name)


def resplit(
    root: str,
    train_ratio: float,
    seed: int,
    unified_dir: str = "unified",
    copy_files: bool = True,
) -> dict:
    unified = os.path.join(root, unified_dir)
    if not os.path.isdir(unified):
        raise FileNotFoundError(f"No unified directory at {unified}")

    names = sorted(
        (name for name in os.listdir(unified) if name.endswith(".pt")), key=numeric_key
    )
    if not names:
        raise FileNotFoundError(f"No .pt files under {unified}")

    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    train_count = int(round(len(shuffled) * float(train_ratio)))
    train_count = max(0, min(len(shuffled), train_count))
    train_files = sorted(shuffled[:train_count], key=numeric_key)
    test_files = sorted(shuffled[train_count:], key=numeric_key)

    previous_path = os.path.join(root, "split_info.json")
    if os.path.isfile(previous_path):
        reference_path = os.path.join(root, "split_info_reference.json")
        if not os.path.isfile(reference_path):
            shutil.copy2(previous_path, reference_path)

    operation = shutil.copy2 if copy_files else shutil.move
    for sub, files in (("train", train_files), ("test", test_files)):
        target = os.path.join(root, sub)
        shutil.rmtree(target, ignore_errors=True)
        os.makedirs(target, exist_ok=True)
        for name in files:
            operation(os.path.join(unified, name), os.path.join(target, name))

    total = len(train_files) + len(test_files)
    info = {
        "train_ratio": len(train_files) / total if total else 0.0,
        "test_ratio": len(test_files) / total if total else 0.0,
        "total_files": total,
        "requested_train_ratio": float(train_ratio),
        "seed": int(seed),
        "split_method": "random",
        "source_dir": os.path.abspath(unified),
        "train_files": train_files,
        "test_files": test_files,
    }
    with open(os.path.join(root, "split_info.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Dataset roots containing unified/")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unified-dir", type=str, default="unified")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move instead of copy; the unified directory is no longer a full copy.",
    )
    args = parser.parse_args()

    for root in args.roots:
        info = resplit(
            root,
            train_ratio=args.train_ratio,
            seed=args.seed,
            unified_dir=args.unified_dir,
            copy_files=not args.move,
        )
        print(
            f"{root}: train {len(info['train_files'])} / test {len(info['test_files'])} "
            f"(ratio {info['train_ratio']:.4f}, seed {info['seed']})"
        )


if __name__ == "__main__":
    main()
