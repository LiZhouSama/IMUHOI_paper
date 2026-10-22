#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_dir="outputs/IMUHOI++/paper_recipe_aug_seed10_behave_20261019"
test -f "$run_dir/protocol.json"
test -f "$run_dir/preflight.json"
test -f "$run_dir/recipe_audit.json"
if [[ -f "$run_dir/launcher.pid" ]] && kill -0 "$(cat "$run_dir/launcher.pid")" 2>/dev/null; then
  echo "Existing BEHAVE runner is active"
  exit 0
fi
exec bash scripts/run_behave_paper_aug_20261019.sh run
