#!/usr/bin/env bash
#SBATCH --account=hk-project-p0029489
#SBATCH --partition=gpu-h100
#SBATCH --job-name=cw-resume-budget
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@2700
#SBATCH --no-requeue
#SBATCH --output=artifacts/cw/slurm-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
repo_root="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd -- "$repo_root"
experiment="${1:?Experiment name required}"
[[ "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || exit 2
# Optional submission-time pins reject changed launch code while the job waited.
if [[ -n "${2:-}" ]]; then
  read -r actual_runtime_sha _ < <(sha256sum tools/cw_batch_runtime.py)
  [[ "$actual_runtime_sha" == "$2" ]] || { echo 'Runtime changed since submission' >&2; exit 2; }
fi
if [[ -n "${3:-}" ]]; then
  read -r actual_batch_sha _ < <(sha256sum slurm/cw_resume.sbatch)
  [[ "$actual_batch_sha" == "$3" ]] || { echo 'Batch template changed since submission' >&2; exit 2; }
fi
workspace_root="${MULTIVIEW_STORAGE_ROOT:-$(cd "$repo_root/../.." && pwd -P)}"
export MULTIVIEW_STORAGE_ROOT="$workspace_root"
export HF_HOME="$workspace_root/cache/huggingface" TORCH_HOME="$workspace_root/cache/torch"
export XDG_CACHE_HOME="$workspace_root/cache" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CW_PYTHON="$workspace_root/envs/cw-diffusion/bin/python"
export CUDA_HOME="${CW_CUDA_HOME:-/software/easybuild/x86_64/software/CUDA/13.0.2}"
export PATH="$(dirname -- "$CW_PYTHON"):$CUDA_HOME/bin:$PATH"
if [[ -f "$CUDA_HOME/targets/x86_64-linux/include/cuda_runtime.h" ]]; then
  export CUDA_INC_PATH="$CUDA_HOME/targets/x86_64-linux/include"
  export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
export TORCH_EXTENSIONS_DIR="$workspace_root/cache/torch_extensions/cw-diffusion"
export CUDA_CACHE_PATH="$workspace_root/cache/cuda/cw-diffusion"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS=8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$repo_root/vendor/flux3/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export SLURM_EXPORT_ENV=ALL
run_root="$repo_root/artifacts/cw/experiments/$experiment"
[[ -d "$run_root/reports" ]] || exit 2
# One batch writer per training trajectory. Never overlap two resumes.
exec 9> "$run_root/.batch-training.lock"
flock -n 9 || { echo 'Another batch writer holds this experiment' >&2; exit 2; }
attempt="joint_sbatch_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}"
batch_root="$run_root/reports/$attempt"
mkdir -- "$batch_root"
stop_file="$batch_root/STOP_REQUESTED"
request_stop() { touch -- "$stop_file"; }
trap request_stop USR1 TERM INT
date -u
"$CW_PYTHON" -m tools.cw_batch_runtime plan "$experiment" "$attempt" --output "$batch_root/plan.json"
srun --ntasks=1 --cpu-bind=none --cpus-per-task=64 --gres=gpu:4 --kill-on-bad-exit=1 \
  "$CW_PYTHON" -m accelerate.commands.launch --num_machines 1 --num_processes 4 --multi_gpu \
  --mixed_precision bf16 --dynamo_backend no --module tools.cw_batch_runtime run \
  --plan "$batch_root/plan.json" --stop-file "$stop_file" &
child_pid=$!
# A warning interrupts bash wait; re-wait until the actual step has exited.
while true; do
  set +e
  wait "$child_pid"
  child_status=$?
  set -e
  if kill -0 "$child_pid" 2>/dev/null; then continue; fi
  break
done
printf '%s\n' "$child_status" > "$batch_root/step_exit_code"
[[ "$child_status" -eq 0 ]] || exit "$child_status"
[[ -f "$batch_root/completion.json" ]] || { echo 'Missing verified completion receipt' >&2; exit 3; }
date -u
