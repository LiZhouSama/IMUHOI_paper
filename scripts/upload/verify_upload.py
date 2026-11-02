#!/usr/bin/env python3
"""Check an uploaded copy of the repo against scripts/upload/upload_manifest.tsv (written by make_manifest.py on
the source machine): every file must exist with the same size, and checkpoints / configs / code must have the
same sha256. Run from the repo root on the server; exit code 1 lists everything that has to be copied again.

    python scripts/upload/verify_upload.py                 # sizes + hashes
    python scripts/upload/verify_upload.py --kind data     # one group (code, data, ckpt, ckpt_imuhoi3b)
    python scripts/upload/verify_upload.py --no-hash       # sizes only (fast)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "scripts/upload/upload_manifest.tsv")
    ap.add_argument("--kind", action="append", default=None)
    ap.add_argument("--no-hash", action="store_true")
    args = ap.parse_args()
    with open(args.manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if args.kind is None or r["kind"] in args.kind]
    missing, wrong_size, wrong_hash, ok = [], [], [], Counter()
    for r in rows:
        path = ROOT / r["path"]
        if not path.is_file():
            missing.append(r)
        elif path.stat().st_size != int(r["bytes"]):
            wrong_size.append(r)
        elif r["sha256"] and not args.no_hash and sha256(path) != r["sha256"]:
            wrong_hash.append(r)
        else:
            ok[r["kind"]] += 1
    total = Counter(r["kind"] for r in rows)
    for kind in sorted(total):
        print(f"{kind:14s} {ok[kind]:6d} / {total[kind]:6d} ok")
    for label, bad in (("MISSING", missing), ("WRONG SIZE", wrong_size), ("WRONG SHA256", wrong_hash)):
        if bad:
            print(f"\n{label} ({len(bad)}):")
            # Group by top-level directory of the upload entry to keep the report short.
            for r in bad[:200]:
                print(f"  [{r['kind']}] {r['path']}")
            if len(bad) > 200:
                print(f"  ... {len(bad) - 200} more")
    problems = len(missing) + len(wrong_size) + len(wrong_hash)
    print("\nALL OK" if not problems else f"\n{problems} problem(s): copy these paths again")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
