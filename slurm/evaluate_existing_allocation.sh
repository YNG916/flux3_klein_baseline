#!/usr/bin/env bash
# Explicit evaluation launch on an existing allocation; no sbatch/salloc.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
workspace_root="${MULTIVIEW_STORAGE_ROOT:-$(cd "$repo_root/../.." && pwd -P)}"
export MULTIVIEW_STORAGE_ROOT="$workspace_root"
export PYTHONPATH="$repo_root/vendor/flux3/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd -- "$repo_root"
case " $* " in *' --plan-only '*) exec "${FLUX_PYTHON:-$workspace_root/envs/flux3-klein/bin/python}" slurm/evaluate_existing_allocation.py "$@";; esac
job_id="${1:?Existing job ID required}"
experiment="${2:?Experiment name required}"
[[ "$job_id" =~ ^[0-9]+$ && "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || exit 2
exp_root="artifacts/mvwd/experiments/$experiment"
[[ -f "$exp_root/manifest.json" ]] || exit 2
attempt="$(date -u +%Y%m%dT%H%M%S%N)"
mkdir -p -- "$exp_root/logs" "$exp_root/source/evaluation"
cp -- "${BASH_SOURCE[0]}" "$exp_root/source/evaluation/eval-${job_id}-${attempt}.sh"
exec > >(tee -- "$exp_root/logs/driver-eval-${job_id}-${attempt}.log") 2>&1
trap 'code=$?; printf "\nEvaluation driver exit_code=%s at %s\n" "$code" "$(date -u +%FT%TZ)"' EXIT
"${FLUX_PYTHON:-$workspace_root/envs/flux3-klein/bin/python}" slurm/evaluate_existing_allocation.py "$@"
