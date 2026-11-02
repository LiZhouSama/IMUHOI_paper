#!/usr/bin/env python3
"""Write scripts/upload/upload_manifest.tsv: every file behind the upload lists, with its size and, for
checkpoints / configs / code, its sha256 (data files: size only; ~15k .pt files, a size mismatch already catches
a truncated copy). Run on the source machine from the repo root; check the copy with verify_upload.py.

    python scripts/upload/make_manifest.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LISTS = {  # list file -> (kind, hash?)
    "filelist_code.txt": ("code", True),
    "filelist_data.txt": ("data", False),
    "filelist_ckpt.txt": ("ckpt", True),
    "filelist_ckpt_imuhoi3b.txt": ("ckpt_imuhoi3b", True),
}
SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def entries(list_path):
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def files_of(entry):
    path = ROOT / entry
    if path.is_file():
        yield Path(entry)
        return
    for p in sorted(path.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_file() and not (SKIP_PARTS & set(rel.parts)) and p.suffix != ".pyc":
            yield rel


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    out = ROOT / "scripts/upload/upload_manifest.tsv"
    rows, seen = [], set()
    for name, (kind, do_hash) in LISTS.items():
        for entry in entries(ROOT / "scripts/upload" / name):
            for rel in files_of(entry):
                if rel in seen or rel.as_posix() == "scripts/upload/upload_manifest.tsv":
                    continue
                seen.add(rel)
                full = ROOT / rel
                rows.append((kind, rel.as_posix(), str(full.stat().st_size), sha256(full) if do_hash else ""))
    out.write_text("kind\tpath\tbytes\tsha256\n" + "".join("\t".join(r) + "\n" for r in rows))
    for kind in {r[0] for r in rows}:
        sel = [r for r in rows if r[0] == kind]
        print(f"{kind}: {len(sel)} files, {sum(int(r[2]) for r in sel) / 2**30:.2f} GiB")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
