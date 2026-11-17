#!/usr/bin/env bash
# Explicitly use a user-supplied allocation. This file never invokes sbatch/salloc.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
workspace_root="${MULTIVIEW_STORAGE_ROOT:-$(cd "$repo_root/../.." && pwd -P)}"
job_id="${1:?Existing allocation ID required}"; shift
[[ "$job_id" =~ ^[0-9]+$ ]] || exit 2
export MULTIVIEW_STORAGE_ROOT="$workspace_root"
export HF_HOME="$workspace_root/cache/huggingface"
export TORCH_HOME="$workspace_root/cache/torch"
export XDG_CACHE_HOME="$workspace_root/cache"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME="${CW_CUDA_HOME:-/software/easybuild/x86_64/software/CUDA/13.0.2}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$workspace_root/cache/torch_extensions/cw-diffusion"
export CUDA_CACHE_PATH="$workspace_root/cache/cuda/cw-diffusion"
export CW_PYTHON="$workspace_root/envs/cw-diffusion/bin/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-8}"
command="${1:?Expected validate-gpu, prepare, train, infer, or eval}"
case "$command" in
  validate-gpu|eval)
    exec srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --immediate=15 --kill-on-bad-exit=1 --chdir="$repo_root" ./mvwd cw "$@" ;;
  validate-gpu-ddp)
    shift
    export PYTHONPATH="$repo_root/vendor/flux3/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
    exec srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 --gres=gpu:4 --immediate=15 --kill-on-bad-exit=1 --chdir="$repo_root" \
      "$CW_PYTHON" -m accelerate.commands.launch --num_processes 4 --multi_gpu --mixed_precision bf16 --module cw_diffusion.cli validate-gpu --ddp "$@" ;;
  prepare|train|infer)
    export PYTHONPATH="$repo_root/vendor/flux3/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
    exec srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 --gres=gpu:4 --immediate=15 --kill-on-bad-exit=1 --chdir="$repo_root" \
      "$CW_PYTHON" -m accelerate.commands.launch --num_processes 4 --multi_gpu --mixed_precision bf16 --module cw_diffusion.cli "$@" ;;
  *) echo "Unsupported GPU command: $command" >&2; exit 2 ;;
esac
