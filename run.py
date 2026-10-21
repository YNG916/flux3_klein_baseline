"""Immutable experiment config and unique, logged subprocess steps."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from flux_mvwd.common import file_hash, load_config, read_json, write_json

ROOT = Path(__file__).resolve().parent
KINDS = {"select": "selections", "inspect": "diagnostics", "prepare": "features", "train": "training", "infer": "predictions", "evaluate": "evaluations", "caption": "captions", "review": "reports", "profile": "diagnostics", "camera_plan": "diagnostics"}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create")
    create.add_argument("experiment")
    create.add_argument("--config", required=True)
    create.add_argument("--purpose", required=True)
    run = sub.add_parser("run")
    run.add_argument("experiment")
    run.add_argument("--name", required=True)
    run.add_argument("--kind", choices=KINDS, required=True)
    run.add_argument("--processes", type=int, default=1)
    run.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:]
    boundary = argv.index("--") if "--" in argv else len(argv)
    args = p.parse_args(argv[:boundary])
    arguments = argv[boundary + 1:]
    if Path(args.experiment).name != args.experiment or args.experiment in (".", ".."):
        p.error("Experiment must be a simple directory name")
    exp = ROOT / "artifacts/mvwd/experiments" / args.experiment
    if args.action == "create":
        config = load_config(args.config)
        exp.mkdir(parents=True, exist_ok=False)
        for name in ("steps", "logs", "reports", *set(KINDS.values())):
            (exp / name).mkdir(exist_ok=True)
        write_json(exp / "config.json", config)
        from flux_mvwd.runtime import check_upstream
        check_upstream(config)
        sources = {str(path.relative_to(ROOT)): file_hash(path) for path in [ROOT / "run.py", ROOT / "mvwd", ROOT / "UPSTREAM.lock.json", ROOT / "requirements.lock.txt", ROOT / "pyproject.toml", *sorted((ROOT / "flux_mvwd").glob("*.py")), *sorted((ROOT / "tools").glob("*.py"))]}
        snapshot = exp / "source"
        snapshot.mkdir()
        shutil.copy2(ROOT / "run.py", snapshot / "run.py")
        shutil.copytree(ROOT / "flux_mvwd", snapshot / "flux_mvwd", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "tools", snapshot / "tools", ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("mvwd", "UPSTREAM.lock.json", "requirements.lock.txt"):
            shutil.copy2(ROOT / name, snapshot / name)
        shutil.copytree(ROOT / "vendor/flux3/src", snapshot / "vendor/flux3/src", ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("LICENSE", "pyproject.toml"):
            shutil.copy2(ROOT / "vendor/flux3" / name, snapshot / "vendor/flux3" / name)
        shutil.copy2(ROOT / "pyproject.toml", snapshot / "pyproject.toml")
        write_json(exp / "manifest.json", {"purpose": args.purpose, "created_utc": utc(), "source_sha256": sources,
            "python": sys.executable, "config_sha256": file_hash(exp / "config.json")})
        print(exp)
        return
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        p.error("Step name must be a simple directory name")
    manifest = read_json(exp / "manifest.json")
    if file_hash(exp / "config.json") != manifest["config_sha256"]:
        raise ValueError("Experiment config was modified; create a new experiment")
    for name, expected in manifest["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("Implementation changed; create a new experiment")
    if any(a.split("=", 1)[0] in ("--config", "--output") for a in arguments):
        p.error("Runner owns --config and --output")
    if args.processes < 1 or (args.processes != 1 and args.kind not in ("train", "prepare", "caption", "infer")):
        p.error("Multiple processes are supported for train/prepare/caption/infer")
    command = [sys.executable]
    if args.kind in ("train", "prepare", "caption", "infer"):
        command += ["-m", "accelerate.commands.launch", "--num_processes", str(args.processes), "--mixed_precision", "bf16"]
        if args.processes > 1:
            command += ["--multi_gpu"]
        command += ["--module"]
    else:
        command += ["-m"]
    output = exp / KINDS[args.kind] / args.name
    command += ["flux_mvwd.cli", args.kind, "--config", str(exp / "config.json"), "--output", str(output), *arguments]
    if args.dry_run:
        import shlex
        print(shlex.join(command))
        return
    receipt_path = exp / "steps" / (args.name + ".json")
    # Exclusive creation reserves a step; failed attempts remain visible.
    with receipt_path.open("x") as stream:
        stream.write("{}\n")
    receipt = {"argv": command, "cwd": str(ROOT), "python": sys.executable, "started_utc": utc(),
        "environment": {k: v for k, v in os.environ.items() if k.startswith("SLURM_") or k in ("CUDA_VISIBLE_DEVICES", "CONDA_PREFIX")},
        "status": "running", "config_sha256": manifest["config_sha256"]}
    write_json(receipt_path, receipt)
    code = 1
    try:
        with (exp / "logs" / (args.name + ".log")).open("x") as log:
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=child_env)
            try:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
    finally:
        receipt.update(finished_utc=utc(), exit_code=code, status="completed" if code == 0 else "failed")
        write_json(receipt_path, receipt)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
