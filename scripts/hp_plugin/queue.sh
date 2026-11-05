#!/bin/bash
# Run the HP plug-in jobs on several GPUs, one job per GPU at a time (never two trainings on one GPU).
#   GPUS="0 1 2 3" bash scripts/hp_plugin/queue.sh [jobs file]       (default scripts/hp_plugin/jobs.txt)
# Each lane takes the next unclaimed "<dataset> <method>" line under a lock and runs
# scripts/hp_plugin/run_plugin.sh on its GPU.  Finished jobs go to $ROOT/queue/done.txt, failed ones
# to failed.txt; starting the queue again skips done jobs and retries the rest (each job resumes).
# Env: GPUS (required), IMUHOI_PYTHON, HP_PLUGIN_ROOT, PHYSICS_WORKERS (passed to run_plugin.sh).
cd "$(dirname "$0")/../.." || exit 1
JOBS=${1:-scripts/hp_plugin/jobs.txt}
[ -n "$GPUS" ] || { echo "set GPUS, e.g. GPUS=\"0 1 2 3\""; exit 2; }
ROOT=${HP_PLUGIN_ROOT:-outputs/IMUHOI++/RNN_cano/hp_plugin_20261105}
Q=$ROOT/queue
mkdir -p "$Q"
touch "$Q/done.txt" "$Q/failed.txt"
cp "$Q/done.txt" "$Q/claimed.txt"   # unfinished claims of an earlier run are released
: > "$Q/failed.txt"

claim() {  # print the next unclaimed job and mark it claimed (under the queue lock)
  exec 8>"$Q/.lock"
  flock 8
  local job=""
  while read -r line; do
    line=${line%%#*}
    line=$(echo $line)
    [ -z "$line" ] && continue
    if ! grep -qxF "$line" "$Q/claimed.txt"; then
      job=$line
      echo "$job" >> "$Q/claimed.txt"
      break
    fi
  done < "$JOBS"
  flock -u 8
  exec 8>&-
  echo "$job"
}

lane() {
  local gpu=$1 job ds m
  while true; do
    job=$(claim)
    [ -z "$job" ] && break
    read -r ds m <<< "$job"
    echo "[$(date +%FT%T)] gpu$gpu start $job" | tee -a "$Q/queue.log"
    if bash scripts/hp_plugin/run_plugin.sh "$ds" "$m" "$gpu" >> "$Q/lane_gpu${gpu}.log" 2>&1; then
      echo "$job" >> "$Q/done.txt"
      echo "[$(date +%FT%T)] gpu$gpu done $job" | tee -a "$Q/queue.log"
    else
      echo "$job" >> "$Q/failed.txt"
      echo "[$(date +%FT%T)] gpu$gpu FAILED $job" | tee -a "$Q/queue.log"
    fi
  done
  echo "[$(date +%FT%T)] gpu$gpu lane finished" | tee -a "$Q/queue.log"
}

for gpu in $GPUS; do
  lane "$gpu" &
  sleep 5
done
wait
echo "[$(date +%FT%T)] queue finished: $(wc -l < "$Q/done.txt") done, $(wc -l < "$Q/failed.txt") failed" | tee -a "$Q/queue.log"
