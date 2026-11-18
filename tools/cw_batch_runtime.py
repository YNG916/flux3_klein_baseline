"""Audited short-batch run control around the unchanged, frozen CW trainer.

No model/optimizer/data/sampler/RNG math is replaced. At optimizer boundaries an
extra all-reduce agrees on a soft-stop request. The trainer's existing handler
then saves its complete state. DataLoader workers receive no process signals.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path
import inspect
import math
import os
import signal
import sys

from flux_mvwd.common import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def verify_checkpoint(path, contract):
    path = Path(path)
    m = read_json(path / "metadata.json")
    if not m.get("complete") or m.get("contract") != contract:
        raise ValueError("Incomplete checkpoint or different frozen training contract")
    expected = {
        "model.pt",
        "optimizer.pt",
        *[f"rank_{i:04d}.pt" for i in range(contract["world_size"])],
    }
    if set(m["files"]) != expected:
        raise ValueError("Missing model, optimizer or rank RNG state")
    for name, sha in m["files"].items():
        if file_hash(path / name) != sha:
            raise ValueError("Checkpoint checksum differs: " + name)
    progress = m["progress"]
    local = contract["padded_groups"] // contract["world_size"]
    accum = contract["accumulation"]
    if not (0 <= progress["cursor"] < local) or progress["cursor"] % accum:
        raise ValueError("Invalid checkpoint sampler cursor")
    if (
        progress["step"]
        != progress["epoch"] * (local // accum) + progress["cursor"] // accum
    ):
        raise ValueError("Checkpoint is not an optimizer boundary")
    if (
        progress["global_groups_seen"]
        != progress["step"] * accum * contract["world_size"]
    ):
        raise ValueError("Checkpoint sample count differs")
    return m


def create_plan(experiment, attempt, output):
    from cw_diffusion.workflow import config_at
    from cw_diffusion.checkpoint import source_identity, environment_identity

    if Path(attempt).name != attempt or attempt in (".", ".."):
        raise ValueError("Simple unique attempt name required")
    p, c = config_at(experiment)
    caches = {s: read_json(p / "cache" / s / "manifest.json") for s in ("train", "val")}
    if any(not m.get("complete") or not m.get("full_split") for m in caches.values()):
        raise ValueError("Full verified train/val caches required")
    settings = c["training"]
    world, accum = settings["world_size"], settings["accumulation"]
    groups = caches["train"]["groups"]
    padded = math.ceil(groups / (world * accum)) * world * accum
    budget = settings["epochs"] * (padded // (world * accum))
    contract = dict(
        sources=source_identity(),
        environment=environment_identity(),
        config_sha256=file_hash(p / "config.json"),
        train_cache_sha256=file_hash(p / "cache/train/manifest.json"),
        val_cache_sha256=file_hash(p / "cache/val/manifest.json"),
        world_size=world,
        accumulation=accum,
        groups=groups,
        padded_groups=padded,
        target_updates=budget,
    )
    candidates = []
    for f in (p / "training").glob("*/*/metadata.json"):
        m = read_json(f)
        if m.get("complete"):
            if m.get("contract") != contract:
                raise ValueError(
                    "Foreign complete checkpoint within experiment: " + str(f)
                )
            candidates.append((m["progress"]["step"], f.stat().st_mtime_ns, f.parent))
    if not candidates:
        raise ValueError("No complete same-trajectory checkpoint available")
    step, _, checkpoint = max(candidates)
    verify_checkpoint(checkpoint, contract)
    if (p / "steps" / f"{attempt}.json").exists() or (
        p / "training" / attempt
    ).exists():
        raise FileExistsError("Attempt exists; never overwrite an earlier run")
    plan = dict(
        schema="cw-short-batch-v1",
        experiment=experiment,
        attempt=attempt,
        experiment_path=str(p),
        resume=str(checkpoint.resolve()),
        start_step=step,
        stop_after_checkpoint=budget,
        target_updates=budget,
        contract=contract,
        training_complete=step >= budget,
        runtime_sha256=file_hash(Path(__file__)),
        batch_script_sha256=file_hash(ROOT / "slurm/cw_resume.sbatch"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Plan already exists")
    write_json(output, plan)
    print(
        f"resume_step={step} stop_checkpoint={plan['stop_after_checkpoint']} full_budget={budget}",
        flush=True,
    )
    return plan


class StopController:
    def __init__(self, stop_file, target_step):
        self.stop_file = Path(stop_file)
        self.target_step = target_step
        self.reason = None

    def request_stop(self, reason):
        if self.reason is None:
            handler = signal.getsignal(signal.SIGTERM)
            if not callable(handler):
                raise RuntimeError(
                    "Trainer's cooperative SIGTERM handler is not installed"
                )
            self.reason = reason
            handler(signal.SIGTERM, None)

    def optimizer_boundary(self, accelerator, value, reduce_impl):
        flag = value.new_tensor(int(self.stop_file.exists()))
        request = reduce_impl(accelerator, flag, reduction="sum")
        if request.item() > 0:
            self.request_stop("slurm_advance_warning")

    def checkpoint_complete(self, progress):
        if progress["step"] >= self.target_step:
            self.request_stop("next_complete_periodic_checkpoint")


@contextmanager
def run_control(controller):
    from accelerate import Accelerator
    import cw_diffusion.training as training

    original_reduce = Accelerator.reduce
    original_save = training.save_checkpoint
    training_file = Path(training.__file__).resolve()

    def reduce(accelerator, tensor, *args, **kwargs):
        result = original_reduce(accelerator, tensor, *args, **kwargs)
        caller = inspect.currentframe().f_back
        # The scalar loss reduction occurs after an optimizer step and sampler update.
        boundary = (
            getattr(tensor, "ndim", -1) == 0
            and caller.f_code.co_name == "train"
            and Path(caller.f_code.co_filename).resolve() == training_file
        )
        del caller
        if boundary:
            controller.optimizer_boundary(accelerator, tensor, original_reduce)
        return result

    def save(destination, model, optimizer, progress, contract, accelerator, generator):
        original_save(
            destination, model, optimizer, progress, contract, accelerator, generator
        )
        # All ranks have completed the native checkpoint barriers before requesting stop.
        controller.checkpoint_complete(progress)

    Accelerator.reduce, training.save_checkpoint = reduce, save
    try:
        yield
    finally:
        Accelerator.reduce, training.save_checkpoint = original_reduce, original_save


def run(plan_path, stop_file):
    from cw_diffusion.workflow import config_at
    from cw_diffusion.checkpoint import source_identity, environment_identity
    from cw_diffusion.cli import main as native_main

    plan = read_json(plan_path)
    p, _ = config_at(plan["experiment"])
    if (
        plan["runtime_sha256"] != file_hash(Path(__file__))
        or plan["batch_script_sha256"] != file_hash(ROOT / "slurm/cw_resume.sbatch")
        or plan["contract"]["sources"] != source_identity()
        or plan["contract"]["environment"] != environment_identity()
    ):
        raise ValueError("Runtime or frozen experiment changed after preflight")
    if plan["training_complete"]:
        print("Declared 10-epoch trajectory is already complete", flush=True)
        if int(os.environ.get("RANK", "0")) == 0:
            metadata = verify_checkpoint(plan["resume"], plan["contract"])
            write_json(
                Path(plan_path).parent / "completion.json",
                dict(
                    complete=True,
                    checkpoint=plan["resume"],
                    progress=metadata["progress"],
                    stop_reason="already_complete",
                    full_training_complete=True,
                    original_model_training_source_unchanged=True,
                    plan_sha256=file_hash(plan_path),
                ),
            )
        return
    controller = StopController(stop_file, plan["stop_after_checkpoint"])
    argv = sys.argv
    try:
        sys.argv = [
            "mvwd cw",
            "train",
            plan["experiment"],
            plan["attempt"],
            "--resume",
            plan["resume"],
        ]
        with run_control(controller):
            native_main()
    finally:
        sys.argv = argv
    if int(os.environ.get("RANK", "0")) == 0:
        output = p / "training" / plan["attempt"]
        paths = [
            output / name
            for name in ("final", "interrupted")
            if (output / name / "metadata.json").exists()
        ]
        if len(paths) != 1:
            raise RuntimeError("No unique final complete batch checkpoint")
        metadata = verify_checkpoint(paths[0], plan["contract"])
        write_json(
            Path(plan_path).parent / "completion.json",
            dict(
                complete=True,
                checkpoint=str(paths[0]),
                progress=metadata["progress"],
                stop_reason=controller.reason,
                full_training_complete=metadata["progress"]["step"]
                >= plan["target_updates"],
                original_model_training_source_unchanged=True,
                plan_sha256=file_hash(plan_path),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("plan")
    a.add_argument("experiment")
    a.add_argument("attempt")
    a.add_argument("--output", required=True)
    a = sub.add_parser("run")
    a.add_argument("--plan", required=True)
    a.add_argument("--stop-file", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        create_plan(args.experiment, args.attempt, args.output)
    else:
        run(args.plan, args.stop_file)


if __name__ == "__main__":
    main()
