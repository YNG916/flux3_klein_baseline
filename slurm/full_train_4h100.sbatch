#!/usr/bin/env bash
#SBATCH --account=hk-project-test-p0029577
#SBATCH --partition=gpu-h100
#SBATCH --job-name=flux3-mvwd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=artifacts/slurm-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Run using explicit ./mvwd batch or sbatch}"
repo_root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd -- "$repo_root"
experiment="${1:?Experiment name required}"
stage="${2:-train}"
epochs="${3:-10}"
[[ "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || exit 2
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL
exp_root="artifacts/mvwd/experiments/$experiment"
mkdir -p -- "$exp_root/logs"
receipt="$exp_root/logs/slurm-${SLURM_JOB_ID}-execution.log"
date -u > "$receipt"
env | sort | sed -n '/^SLURM_/p' >> "$receipt"
srun --unbuffered --ntasks=1 --kill-on-bad-exit=1 ./mvwd full "$experiment" --stage "$stage" --processes 4 --epochs "$epochs"
