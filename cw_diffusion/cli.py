"""Human entry: ./mvwd cw COMMAND. GPU operations are always explicit."""

from pathlib import Path
import argparse, datetime, json, os, sys, subprocess
from flux_mvwd.common import read_json, write_json
from .workflow import ROOT, config_at


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="CPU label availability audit; no GPU")
    a.add_argument("--dataset", required=True)
    a.add_argument("--output", required=True)
    a = sub.add_parser("init", help="Freeze one independent canonical-world experiment")
    a.add_argument("experiment")
    a.add_argument("--config", default="configs/cw/full.json")
    a = sub.add_parser("doctor", help="CPU environment/config readiness; no model load")
    a = sub.add_parser(
        "validate-gpu",
        help="Explicit real pretrained CUDA, equivalence, gradient and memory acceptance; no optimizer steps",
    )
    a.add_argument("--config", default="configs/cw/full.json")
    a.add_argument("--output", required=True)
    a.add_argument(
        "--ddp",
        action="store_true",
        help="Four-rank gradient synchronization acceptance; no optimizer updates",
    )
    a = sub.add_parser(
        "prepare", help="Explicit DINO supervision cache, reuse baseline VAE encodings"
    )
    a.add_argument("experiment")
    a.add_argument("--split", choices=["train", "val"], required=True)
    a.add_argument("--name", help="Unique cache-preparation attempt name")
    a.add_argument("--max-episodes", type=int)
    a = sub.add_parser(
        "train",
        help="Explicit native pretrained joint training, no allocation submission",
    )
    a.add_argument("experiment")
    a.add_argument("name")
    a.add_argument("--resume")
    a.add_argument("--initialize-from")
    a.add_argument("--max-updates", type=int)
    a = sub.add_parser(
        "infer", help="Explicit synchronous three-camera generation; GT-free"
    )
    a.add_argument("experiment")
    a.add_argument("name")
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--split", choices=["train", "val", "test"], default="test")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument(
        "--resume-predictions",
        help="Existing same-experiment generation directory; keep completed synchronous frames",
    )
    a.add_argument("--max-episodes", type=int)
    a = sub.add_parser(
        "review", help="CPU GT/BEV comparisons, after completed generation"
    )
    a.add_argument("experiment")
    a.add_argument("name")
    a.add_argument("--predictions", required=True)
    a = sub.add_parser(
        "eval",
        help="Existing four metric mathematics in the isolated metrics environment",
    )
    a.add_argument("experiment")
    a.add_argument("name")
    a.add_argument("--predictions", nargs="+", required=True)
    a.add_argument("--preview", action="store_true")
    a = sub.add_parser("status")
    a.add_argument("experiment")
    args = p.parse_args()
    if args.command == "doctor":
        from .checkpoint import environment_identity
        import importlib.util

        print(
            json.dumps(
                {
                    "python": sys.executable,
                    "environment": environment_identity(),
                    "gsplat": bool(importlib.util.find_spec("gsplat")),
                    "cuda_compiler": __import__("shutil").which("nvcc"),
                    "warning": "CPU readiness only; CUDA compilation and pretrained GPU equivalence/gradient/memory measurements remain required",
                },
                indent=2,
            )
        )
        return
    if args.command == "validate-gpu":
        from .gpu_validation import validate

        validate(args.config, args.output, args.ddp)
        return
    if args.command == "audit":
        from .audit import audit

        audit(args.dataset, args.output)
        return
    if args.command == "init":
        from .workflow import initialize

        initialize(args.experiment, args.config)
        return
    exp, c = config_at(args.experiment)
    if args.command == "status":
        for f in sorted((exp / "steps").glob("*.json")):
            x = read_json(f)
            print(f.stem, x.get("status"), x.get("exit_code"))
        return
    rank = int(os.environ.get("RANK", "0"))
    name = getattr(args, "name", None) or args.command + "_" + getattr(
        args, "split", "all"
    )
    if Path(name).name != name or name in (".", ".."):
        p.error("Simple step name required")
    receipt = exp / "steps" / f"{name}.json"
    if rank == 0:
        if receipt.exists():
            raise FileExistsError("Step exists; use a new name for retry/resume")
        state = {
            "argv": sys.argv,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "running",
            "config_sha256": __import__(
                "flux_mvwd.common", fromlist=["file_hash"]
            ).file_hash(exp / "config.json"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "python": sys.executable,
        }
        with receipt.open("x") as stream:
            json.dump(state, stream)
    else:
        state = None

    class Tee:
        def __init__(self, original, log):
            self.original, self.log = original, log

        def write(self, text):
            self.original.write(text)
            self.log.write(text)
            self.log.flush()

        def __getattr__(self, name):
            return getattr(self.original, name)

        def flush(self):
            self.original.flush()
            self.log.flush()

    log_path = exp / "logs" / f"{name}_rank_{rank:04d}.log"
    log = log_path.open("x")
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
    try:
        if args.command == "prepare":
            from .workflow import prepare

            prepare(args.experiment, args.split, args.max_episodes)
        elif args.command == "train":
            from .training import train

            train(
                args.experiment,
                args.name,
                args.resume,
                args.initialize_from,
                args.max_updates,
            )
        elif args.command == "infer":
            from .inference import infer

            infer(
                args.experiment,
                args.name,
                args.checkpoint,
                args.split,
                args.seed,
                args.max_episodes,
                args.resume_predictions,
            )
        elif args.command == "review":
            from tools.visualizations import export_comparisons

            result = export_comparisons(args.predictions)
            write_json(exp / "reports" / f"{name}.json", result)
        elif args.command == "eval":
            for d in args.predictions:
                m = read_json(Path(d) / "metadata.json")
                if (
                    not m.get("complete")
                    or m.get("ground_truth_loaded") is not False
                    or m.get("method") != "BEV-Grounded-Canonical-World-Diffusion"
                ):
                    raise ValueError("Require completed GT-free canonical predictions")
            python = ROOT.parent.parent / "envs/mvwd-metrics/bin/python"
            command = [
                str(python),
                "-m",
                "flux_mvwd.cli",
                "evaluate",
                "--config",
                str(exp / "baseline_config.json"),
                "--output",
                str(exp / "evaluations" / name),
                "--predictions",
                *args.predictions,
                "--metrics-repo",
                str(ROOT.parent / "top2pano"),
                "--weights-dir",
                str(ROOT.parent / "top2pano/models/metrics"),
                "--device",
                "cuda",
            ]
            if args.preview:
                command.append("--preview")
            subprocess.run(command, check=True, cwd=ROOT)
        if state is not None:
            state.update(status="completed", exit_code=0)
    except BaseException as exc:
        import traceback

        traceback.print_exc(file=log)
        if state is not None:
            state.update(
                status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}"
            )
        raise
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        log.close()
        if state is not None:
            state["finished_utc"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            write_json(receipt, state)


if __name__ == "__main__":
    main()
