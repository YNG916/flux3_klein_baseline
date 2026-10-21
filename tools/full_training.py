"""Frozen full-query plan and per-attempt logged, resumable stages."""
from pathlib import Path
import datetime as dt
import math
import os
import shutil
from flux_mvwd.common import load_config, read_json, digest, write_json
from flux_mvwd.data import Level1Data
from flux_mvwd.runtime import source_hash
from .workflow import EXPERIMENTS, run_step


def latest_checkpoint(experiment):
    candidates = []
    for path in (Path(experiment) / "training").glob("*/*/metadata.json"):
        meta = read_json(path)
        if meta.get("schema") == "flux3-mvwd-training-state-v3" and meta.get("complete"):
            candidates.append((meta["step"], path.stat().st_mtime_ns, path.parent))
    return max(candidates)[2] if candidates else None


def full_training(args):
    if args.processes < 1 or args.epochs < 1:
        raise ValueError("Processes and epochs must be positive")
    exp = EXPERIMENTS / args.experiment
    config = load_config(exp / "config.json")
    settings = config["training"]
    support = {}
    for split in ("train", "val"):
        data = Level1Data(config, split)
        support[split] = {"queries": len(list(data.queries())), "dataset": data.provenance()}
    effective = args.processes * settings["micro_batch_size"] * settings["gradient_accumulation_steps"]
    updates = math.ceil(support["train"]["queries"] / effective)
    plan = {"schema": "flux3-mvwd-full-plan-v1", "config_sha256": digest(config),
        "implementation_sha256": source_hash(), "processes": args.processes, "support": support,
        "effective_batch": effective, "updates_per_epoch": updates,
        "epochs": settings["epochs"], "updates": updates * settings["epochs"],
        "padding_queries_per_epoch": updates * effective - support["train"]["queries"]}
    h, w = config["target_hw"]
    per_query = (h * w * (3 if config["mode"] == "joint" else 1) // 256 + 2 * config["bev_size"] ** 2 // 256) * (128 * 2 + 4 * 4)
    plan["latent_tensor_upper_bound_bytes_before_dedup"] = {
        split: value["queries"] * per_query for split, value in support.items()}
    plan["storage_note"] = "Tensor payload estimate only; excludes prompt/zip/manifest overhead. Cached text mode additionally stores text_tokens*20480*2 bytes per query. Measure pilot manifest before full prepare."
    if args.epochs > plan["epochs"]:
        raise ValueError("Requested epochs exceed the frozen experiment budget")
    plan_path = exp / "plans/full_training.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Full plan changed; use the original config/process count or a new experiment")
    if not args.dry_run:
        write_json(plan_path, plan)
    print({k: v for k, v in plan.items() if k != "support"}, flush=True)
    if args.stage == "plan": return
    attempt = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    if args.stage in ("all", "caption"):
        for split in ("train", "val"):
            run_step(args.experiment, f"caption_{split}_{attempt}", "caption",
                ["--split", split, "--cache-root", str(exp / "captions" / split)], args.processes, args.dry_run)
    if args.stage in ("all", "prepare"):
        for split in ("train", "val"):
            run_step(args.experiment, f"prepare_{split}_{attempt}", "prepare",
                ["--split", split, "--captions", str(exp / "captions" / split),
                 "--cache-root", str(exp / "features" / split)], args.processes, args.dry_run)
    if args.stage in ("all", "train"):
        checkpoint = latest_checkpoint(exp)
        if checkpoint:
            step = read_json(checkpoint / "metadata.json")["step"]
            if step >= updates * args.epochs:
                print(f"Already complete through {args.epochs} epochs: {checkpoint}")
                return
        extra = ["--cache", str(exp / "features/train"), "--validation-cache", str(exp / "features/val"),
                 "--epochs", str(args.epochs)]
        if checkpoint: extra += ["--resume-from", str(checkpoint)]
        run_step(args.experiment, f"train_{attempt}", "train", extra, args.processes, args.dry_run)
