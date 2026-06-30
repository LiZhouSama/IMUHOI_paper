"""Generate TIP SBP sidecar caches for existing processed IMUHOI datasets."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

from tqdm import tqdm

from Comparisons.tip.sbp_cache import generate_cache_for_file, iter_processed_files


DEFAULT_DATA_ROOTS = [
    "process/processed_seg_data_BEHAVE_bps",
    "process/processed_seg_data_HODOME_bps",
    "process/processed_seg_data_IMHD_bps",
    "process/processed_seg_data_PAHOI_bps",
    "process/processed_split_data_OMOMO_bps",
]
DEFAULT_CACHE_ROOT = "Comparisons/tip/sbp_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TIP SBP sidecar cache files")
    parser.add_argument("--data_roots", nargs="*", default=DEFAULT_DATA_ROOTS)
    parser.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def _run_one(args_tuple):
    return generate_cache_for_file(*args_tuple)


def main() -> None:
    args = parse_args()
    files = iter_processed_files(args.data_roots)
    cache_root = Path(args.cache_root)
    print(f"source files: {len(files)}")
    print(f"cache root: {cache_root}")
    if not files:
        return

    start = time.perf_counter()
    counts = {"write": 0, "skip": 0, "error": 0}
    frames = 0
    worker_args = [(str(path), str(cache_root), bool(args.overwrite), float(args.fps)) for path in files]

    if args.workers <= 1:
        iterator = map(_run_one, worker_args)
        for status, num_frames, _ in tqdm(iterator, total=len(worker_args)):
            counts[status] = counts.get(status, 0) + 1
            frames += max(0, int(num_frames))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_one, item) for item in worker_args]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    status, num_frames, _ = future.result()
                except Exception as exc:
                    counts["error"] += 1
                    print(f"[error] {exc}")
                    continue
                counts[status] = counts.get(status, 0) + 1
                frames += max(0, int(num_frames))

    elapsed = time.perf_counter() - start
    print(
        f"done: {counts}, frames={frames}, elapsed={elapsed:.1f}s, "
        f"throughput={frames / max(elapsed, 1e-6):.1f} frames/s"
    )


if __name__ == "__main__":
    main()

