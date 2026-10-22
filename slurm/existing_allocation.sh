#!/usr/bin/env bash
# Reuse a live allocation; never submit or allocate a new job.
set -euo pipefail
job_id="${1:?Existing job ID required}"
experiment="${2:?Experiment name required}"
stage="${3:-all}"
epochs="${4:-10}"
[[ "$job_id" =~ ^[0-9]+$ && "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && "$epochs" =~ ^[1-9][0-9]*$ ]] || exit 2
case "$stage" in all|caption|prepare|train) ;; *) exit 2;; esac
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$repo_root"
exp_root="artifacts/mvwd/experiments/$experiment"
[[ -f "$exp_root/manifest.json" ]] || { echo 'Initialize the experiment first'; exit 2; }
attempt="$(date -u +%Y%m%dT%H%M%S%N)"
mkdir -p -- "$exp_root/logs" "$exp_root/source/slurm"
cp -- "${BASH_SOURCE[0]}" "$exp_root/source/slurm/existing-${job_id}-${attempt}.sh"
log="$exp_root/logs/driver-${job_id}-${attempt}.log"
exec > >(tee -- "$log") 2>&1
trap 'code=$?; printf "\nDriver exit_code=%s at %s\n" "$code" "$(date -u +%FT%TZ)"' EXIT
printf 'Existing allocation: %s; experiment: %s; stage: %s; epochs: %s\n' "$job_id" "$experiment" "$stage" "$epochs"
date -u
srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 \
  --gres=gpu:4 --immediate=15 --kill-on-bad-exit=1 --chdir="$repo_root" \
  ./mvwd full "$experiment" --stage "$stage" --processes 4 --epochs "$epochs"
