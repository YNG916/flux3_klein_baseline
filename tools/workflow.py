"""Human commands and explicit Slurm submission; no implicit allocations."""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
from flux_mvwd.common import read_json, write_json, file_hash

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "artifacts/mvwd/experiments"
KINDS = {"import-captions": "import_captions", "select": "select", "inspect": "inspect", "caption": "caption", "prepare": "prepare", "train": "train",
         "infer": "infer", "eval": "evaluate", "review": "review", "profile": "profile", "diagnose": "camera_plan"}


def execute(argv):
    print(shlex.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=ROOT, check=True)


def run_step(experiment, name, kind, arguments, processes=1, dry_run=False):
    task_python = sys.executable
    if kind == "evaluate":
        workspace = Path(os.environ.get("MULTIVIEW_STORAGE_ROOT", ROOT.parent.parent))
        task_python = os.environ.get("MVWD_METRICS_PYTHON", str(workspace / "envs/mvwd-metrics/bin/python"))
    command = [task_python, ROOT / "run.py", "run", experiment, "--name", name,
               "--kind", kind, "--processes", processes]
    if dry_run: command += ["--dry-run"]
    execute([*command, "--", *arguments])


def status(experiment=None):
    folders = [EXPERIMENTS / experiment] if experiment else sorted(EXPERIMENTS.glob("*"))
    reports = []
    for exp in folders:
        if not (exp / "manifest.json").exists(): continue
        rows = []
        for path in sorted((exp / "steps").glob("*.json")):
            row = read_json(path)
            rows.append(f"| {path.stem} | {row.get('status', '?')} | {row.get('exit_code', '')} |")
        text = f"# {exp.name}\n\n{read_json(exp / 'manifest.json')['purpose']}\n\n| Step | Status | Exit |\n|---|---|---|\n" + "\n".join(rows) + "\n"
        checkpoints = sorted((exp / "training").glob("*/*/metadata.json"))
        for path in checkpoints:
            meta = read_json(path)
            if meta.get("complete"):
                text += f"\n- checkpoint: [{path.parent.relative_to(exp)}](../{path.parent.relative_to(exp)}) / step {meta['step']}\n"
        for path in sorted((exp / "predictions").glob("*/metadata.json")):
            meta = read_json(path)
            text += f"\n- predictions: [{path.parent.name}](../predictions/{path.parent.name}/index.html), complete={meta.get('complete')}, views={meta.get('view_count', 0)}\n"
        for path in sorted((exp / "evaluations").glob("*/metrics.json")):
            meta = read_json(path)
            scores = ", ".join(f"{key}={value.get('mean')}" for key, value in meta["metrics"].items())
            text += f"\n- metrics: [{path.parent.name}](../evaluations/{path.parent.name}/METRICS.md), {scores}\n"
        write_json(exp / "reports/status.json", {"steps": len(rows), "complete_checkpoints": len(checkpoints)})
        (exp / "reports/OVERVIEW.md").write_text(text)
        reports.append(f"- [{exp.name}](experiments/{exp.name}/reports/OVERVIEW.md)")
        print(text)
    if EXPERIMENTS.exists():
        reports = [f"- [{exp.name}](experiments/{exp.name}/reports/OVERVIEW.md)"
                   for exp in sorted(EXPERIMENTS.iterdir()) if (exp / "manifest.json").exists()]
        (EXPERIMENTS.parent / "EXPERIMENTS.md").write_text("# FLUX MVWD experiments\n\n" + "\n".join(reports) + "\n")


def batch(args):
    exp = EXPERIMENTS / args.experiment
    if not (exp / "manifest.json").exists():
        raise ValueError("Initialize the experiment first")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    snapshot = exp / "source/slurm" / f"submit_{stamp}.sbatch"
    command = ["sbatch", "--parsable", "--chdir", str(ROOT),
        "--output", str(exp / "logs/slurm-%j.log")]
    if args.after: command += ["--dependency", "afterany:" + args.after]
    command += [str(snapshot), args.experiment, args.stage, str(args.epochs)]
    if args.dry_run:
        command[-4] = str(ROOT / "slurm/full_train_4h100.sbatch")
        print(shlex.join(command)); return
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "slurm/full_train_4h100.sbatch", snapshot)
    receipt_path = exp / "steps" / f"submission_{stamp}.json"
    receipt = {"argv": command, "script_sha256": file_hash(snapshot), "status": "submitting"}
    write_json(receipt_path, receipt)
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        receipt.update(status="submitted", job_id=result.stdout.strip(), stderr=result.stderr)
        print("Slurm job:", result.stdout.strip())
    except subprocess.CalledProcessError as error:
        receipt.update(status="failed", stderr=error.stderr, exit_code=error.returncode)
        raise
    finally:
        write_json(receipt_path, receipt)


def main():
    p = argparse.ArgumentParser(description="FLUX 3 × MVWD: one entry for every baseline operation")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Freeze config and source for one independent training trajectory")
    init.add_argument("experiment"); init.add_argument("--config", required=True)
    init.add_argument("--purpose", default="MVWD Level 1 FLUX 3 Base baseline")
    st = sub.add_parser("status"); st.add_argument("experiment", nargs="?")
    for name in KINDS:
        q = sub.add_parser(name, help=f"Logged {name} step; model arguments follow --")
        q.add_argument("experiment"); q.add_argument("name")
        q.add_argument("--processes", type=int, default=1); q.add_argument("--dry-run", action="store_true")
        if name == "infer": q.add_argument("--no-review", action="store_true")
    full = sub.add_parser("full", help="Resumable full-data training workflow")
    full.add_argument("experiment"); full.add_argument("--stage", choices=("plan", "caption", "prepare", "train", "all"), default="plan")
    full.add_argument("--processes", type=int, default=4); full.add_argument("--epochs", type=int, default=10)
    full.add_argument("--dry-run", action="store_true")
    job = sub.add_parser("batch", help="Explicitly submit a four-H100 job")
    job.add_argument("experiment"); job.add_argument("--stage", choices=("caption", "prepare", "train", "all"), default="train")
    job.add_argument("--epochs", type=int, default=10); job.add_argument("--after"); job.add_argument("--dry-run", action="store_true")
    sub.add_parser("doctor", help="CPU-only environment, source, paths and model readiness")
    download = sub.add_parser("download", help="Download pinned weights; never starts GPU work")
    download.add_argument("--config", default="configs/level1.local.json")
    download.add_argument("--only", choices=("base", "pup", "all"), default="all")
    argv = sys.argv[1:]; boundary = argv.index("--") if "--" in argv else len(argv)
    args = p.parse_args(argv[:boundary]); extra = argv[boundary+1:]
    if hasattr(args, "experiment") and args.experiment is not None:
        name = args.experiment
        if Path(name).name != name or name in (".", ".."):
            p.error("Experiment must be a simple directory name")
    if args.command == "init":
        execute([sys.executable, ROOT / "run.py", "create", args.experiment, "--config", args.config, "--purpose", args.purpose])
    elif args.command in KINDS:
        run_step(args.experiment, args.name, KINDS[args.command], extra, args.processes, args.dry_run)
        if args.command == "infer" and not args.no_review:
            prediction = EXPERIMENTS / args.experiment / "predictions" / args.name
            run_step(args.experiment, args.name + "_review", "review", ["--predictions", str(prediction)], dry_run=args.dry_run)
    elif args.command == "status": status(args.experiment)
    elif args.command == "batch": batch(args)
    elif args.command == "full":
        from .full_training import full_training
        full_training(args)
    elif args.command == "doctor":
        from .setup import doctor
        doctor()
    elif args.command == "download":
        from .setup import download
        download(args)


if __name__ == "__main__": main()
