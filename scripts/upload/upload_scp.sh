#!/bin/bash
# Copy the repo subset listed in scripts/upload/filelist_*.txt to a server with the same relative layout.
# Every entry (file or directory, relative to the repo root) goes to DEST/<same path>: the parent directories are
# created first in one ssh call, then each entry is copied with `scp -r` into its parent. One SSH connection is
# reused for all calls (one password prompt). Symlinks are followed (datasets/ -> ../../datasets becomes real
# files on the server).
#
#   bash scripts/upload/upload_scp.sh user@host /remote/path/IMUHOI                      # all four lists
#   bash scripts/upload/upload_scp.sh user@host /remote/path/IMUHOI scripts/upload/filelist_code.txt
#   DRY_RUN=1 bash scripts/upload/upload_scp.sh user@host /remote/path/IMUHOI            # print the commands only
set -euo pipefail
cd "$(dirname "$0")/../.."
REMOTE=${1:?usage: upload_scp.sh user@host /remote/dest [filelist ...]}
DEST=${2:?usage: upload_scp.sh user@host /remote/dest [filelist ...]}
shift 2
LISTS=("$@")
[ ${#LISTS[@]} -eq 0 ] && LISTS=(scripts/upload/filelist_code.txt scripts/upload/filelist_data.txt
                                 scripts/upload/filelist_ckpt.txt scripts/upload/filelist_vc_ablation.txt)
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" -o ControlPersist=30m)

mapfile -t ENTRIES < <(cat "${LISTS[@]}" | grep -vE '^[[:space:]]*(#|$)')
for e in "${ENTRIES[@]}"; do
  [ -e "$e" ] || { echo "missing: $e" >&2; exit 1; }
done
PARENTS=$(for e in "${ENTRIES[@]}"; do dirname "$e"; done | sort -u)

run() { if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "$@"; echo; else "$@"; fi; }
run ssh "${SSH_OPTS[@]}" "$REMOTE" "cd '$DEST' 2>/dev/null || mkdir -p '$DEST'; cd '$DEST' && mkdir -p $(echo $PARENTS)"
for e in "${ENTRIES[@]}"; do
  run scp "${SSH_OPTS[@]}" -rp "$e" "$REMOTE:$DEST/$(dirname "$e")/"
done
echo "done: ${#ENTRIES[@]} entries -> $REMOTE:$DEST"
