#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
workspace_root="${MULTIVIEW_STORAGE_ROOT:-$(cd "$repo_root/../.." && pwd -P)}"
task_python="${PYTHON_BOOTSTRAP:-python3.11}"
export PIP_CACHE_DIR="$workspace_root/cache/pip"
[[ -f "$repo_root/vendor/flux3/src/flux3/dit.py" ]] || { echo 'Missing pinned vendor source'; exit 2; }
"$task_python" -m venv --copies "$workspace_root/envs/flux3-klein"
"$workspace_root/envs/flux3-klein/bin/python" -m pip install -r "$repo_root/requirements.lock.txt"
"$repo_root/mvwd" doctor
