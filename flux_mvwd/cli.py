"""Internal command interface; human entry is ./mvwd."""
import argparse
from .common import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "inspect", "caption", "prepare", "train", "infer", "evaluate", "review", "profile", "camera_plan"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        if name in ("select", "inspect", "caption", "prepare", "infer", "camera_plan"):
            p.add_argument("--split", choices=("train", "val", "test"), default="train")
            p.add_argument("--scene", nargs="+")
            p.add_argument("--max-episodes", type=int)
            p.add_argument("--selection")
        if name in ("caption", "prepare", "infer"):
            p.add_argument("--max-samples", type=int)
        if name in ("caption", "prepare"):
            p.add_argument("--cache-root", help="Persistent resumable cache; logs still belong to this unique step")
        if name == "select":
            p.add_argument("--episodes-per-scene", type=int, required=True)
            p.add_argument("--scene-count", type=int)
            p.add_argument("--selection-seed", type=int, default=42)
        if name == "inspect":
            p.add_argument("--preview-samples", type=int, default=3)
            p.add_argument("--with-targets", action="store_true")
        if name in ("prepare", "infer"):
            p.add_argument("--captions", required=True)
        if name in ("train", "profile"):
            p.add_argument("--cache", required=True)
        if name == "train":
            p.add_argument("--max-steps", type=int)
            p.add_argument("--epochs", type=int)
            p.add_argument("--resume-from")
            p.add_argument("--validation-cache")
        if name == "infer":
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument("--lora")
            g.add_argument("--zero-shot", action="store_true")
            p.add_argument("--seed", type=int, default=42)
        if name == "profile":
            p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
            p.add_argument("--warmup", type=int, default=2)
            p.add_argument("--repeats", type=int, default=3)
            p.add_argument("--checkpointing", choices=("config", "on", "off", "both"), default="config")
        if name == "evaluate":
            p.add_argument("--predictions", nargs="+", required=True)
            p.add_argument("--metrics-repo", required=True)
            p.add_argument("--weights-dir", required=True)
            p.add_argument("--metrics", nargs="+", default=["psnr", "lpips", "fid", "mv_gtwarp_dino"])
            p.add_argument("--device", default="cuda")
            p.add_argument("--preview", action="store_true")
        if name == "review":
            p.add_argument("--predictions", required=True)
        if name == "camera_plan":
            p.add_argument("--max-groups", type=int, default=1)
            p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "select":
        from .selection import select
        select(config, args)
    elif args.command == "inspect":
        from .inspection import inspect
        inspect(config, args)
    elif args.command in ("caption", "prepare"):
        from . import preparation
        getattr(preparation, args.command)(config, args)
    elif args.command == "train":
        from .training import train
        train(config, args)
    elif args.command == "infer":
        from .inference import infer
        infer(config, args)
    elif args.command == "profile":
        from .profiling import profile
        profile(config, args)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(config, args)
    elif args.command == "review":
        from tools.visualizations import export_comparisons
        from .common import write_json
        write_json(__import__("pathlib").Path(args.output) / "review.json", export_comparisons(args.predictions))
    elif args.command == "camera_plan":
        from .camera_diagnostics import plan
        plan(config, args)


if __name__ == "__main__":
    main()
