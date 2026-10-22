"""DDP LoRA training with bounded I/O, fixed validation and full-state resume."""
from pathlib import Path
import json
import math
import signal
import time
import torch

from .common import digest, file_hash, read_json, write_json
from .runtime import require_gpu, source_hash, environment_signature
from .training_state import capture_rng, restore_rng, epoch_indices, checkpoint_receipt
from .cache import FeatureDataset, collate
from .backend import flow_loss, load_dit, load_text_encoder
from .lora import inject, adapter_state, load_adapter, gradient_checkpointing


def training_order(size, settings, epoch, world):
    return epoch_indices(size, settings["seed"], epoch, world,
                         settings["gradient_accumulation_steps"], settings["micro_batch_size"],
                         ["native-variable-length"] * size)


def train(config, args):
    from accelerate import Accelerator
    from accelerate.utils import set_seed, DistributedDataParallelKwargs, InitProcessGroupKwargs
    from datetime import timedelta
    from safetensors.torch import load_file, save_file
    require_gpu()
    settings = config["training"]
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False),
                         InitProcessGroupKwargs(timeout=timedelta(hours=4))])
    world, rank, device = accelerator.num_processes, accelerator.process_index, accelerator.device
    output, root = Path(args.output), Path(args.cache)
    impl = source_hash()

    def read_cache(path, split):
        info = read_json(Path(path) / "manifest.json")
        if (info.get("schema") != "flux3-mvwd-features-v1" or not info.get("complete")
                or not info.get("samples") or info["dataset"]["split"] != split
                or info["config_sha256"] != digest(config) or info["implementation_sha256"] != impl):
            raise ValueError(f"Requires a complete compatible {split} cache")
        if info["versions"] != environment_signature():
            raise ValueError("Feature encoding environment differs from training runtime")
        return info

    receipt = read_cache(root, "train")
    val = read_cache(args.validation_cache, "val") if args.validation_cache else None
    if val and any(val["dataset"][k] != receipt["dataset"][k] for k in ("release_id", "robot_assets")):
        raise ValueError("Train/val release or assets differ")
    accumulation, micro = settings["gradient_accumulation_steps"], settings["micro_batch_size"]
    size = len(receipt["samples"])
    padded = len(training_order(size, settings, 0, world))
    updates_per_epoch = padded // (world * micro * accumulation)
    epochs = args.epochs or settings.get("epochs", 10)
    steps = args.max_steps if args.max_steps is not None else epochs * updates_per_epoch
    if steps < 1 or epochs < 1:
        raise ValueError("Training budget must be positive")
    contract = {"config_sha256": digest(config), "implementation_sha256": impl,
        "versions": environment_signature(),
        "cache_sha256": digest(receipt), "sample_count": size, "world_size": world,
        "micro_batch_size": micro, "accumulation": accumulation, "padded_epoch_samples": padded,
        "sampler": "native-packed-shuffle-pad-full-effective-batch-v1",
        "validation": None if val is None else {"cache_sha256": digest(val),
            "every": settings["validate_every"], "seed": 137, "cfg_dropout": False}}
    resumed = checkpoint_receipt(args.resume_from, contract) if args.resume_from else None
    if resumed and steps <= resumed["step"]:
        raise ValueError("The requested total update budget is already completed")
    set_seed(settings["seed"])
    model = load_dit(config, device)
    modules = inject(model, settings["rank"], settings["alpha"])
    if settings["gradient_checkpointing"]:
        gradient_checkpointing(model)
    if resumed:
        load_adapter(model, load_file(str(Path(args.resume_from) / "lora.safetensors")))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings.get("weight_decay", 0.01),
        fused=settings.get("fused_optimizer", True))
    model, optimizer = accelerator.prepare(model, optimizer)
    text_encoder = load_text_encoder(config, device) if receipt.get("text_mode") == "online" else None
    generator = torch.Generator(device=device).manual_seed(settings["seed"] + rank)
    progress = {"step": 0, "epoch": 0, "cursor": 0, "global_samples_seen": 0}
    if resumed:
        state = torch.load(Path(args.resume_from) / f"state_rank_{rank:04d}.pt", map_location="cpu", weights_only=True)
        if state["progress"] != resumed["progress"]:
            raise ValueError("Checkpoint rank progress differs")
        optimizer.load_state_dict(state["optimizer"])
        if accelerator.scaler is not None:
            accelerator.scaler.load_state_dict(state["scaler"])
        progress = dict(state["progress"])
        restore_rng(state["rng"], device, generator)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "training_metadata.json", {"config": config, "resume_contract": contract,
            "dataset": receipt["dataset"], "starting_progress": progress,
            "resumed_from": str(args.resume_from) if resumed else None,
            "trainable_parameters": sum(p.numel() for p in parameters), "lora_modules": modules,
            "updates_per_epoch": updates_per_epoch, "target_updates": steps,
            "effective_batch": world * micro * accumulation, "padded_epoch_samples": padded,
            "validation": "Fixed per-query noise/time, target-only flow MSE; select final quality using generated val views"})
    accelerator.wait_for_everyone()
    train_data = FeatureDataset(root, receipt["samples"])
    val_data = FeatureDataset(args.validation_cache, val["samples"]) if val else None
    model.train()
    stop = {"requested": False}
    handlers = {}
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        handlers[sig] = signal.signal(sig, lambda *_: stop.update(requested=True))

    def save(name):
        accelerator.wait_for_everyone()
        dest = output / name
        if accelerator.is_main_process:
            dest.mkdir(exist_ok=False)
            save_file(adapter_state(accelerator.unwrap_model(model)), str(dest / "lora.safetensors"))
        accelerator.wait_for_everyone()
        torch.save({"optimizer": optimizer.state_dict(), "progress": dict(progress),
                    "rng": capture_rng(device, generator),
                    "scaler": accelerator.scaler.state_dict() if accelerator.scaler is not None else None},
                   dest / f"state_rank_{rank:04d}.pt")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            files = ["lora.safetensors", *[f"state_rank_{r:04d}.pt" for r in range(world)]]
            write_json(dest / "metadata.json", {"schema": "flux3-mvwd-training-state-v3", "complete": True,
                **progress, "progress": dict(progress), "resume_contract": contract,
                "state_files": {name: file_hash(dest / name) for name in files},
                "config_sha256": digest(config), "dataset": receipt["dataset"], "config": config})
        accelerator.wait_for_everyone()

    def validate():
        rng = capture_rng(device, generator)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        try:
            with torch.no_grad(), accelerator.autocast():
                for i in range(rank, len(val_data), world):
                    seed = int(digest([137, val["samples"][i]["query"]])[:16], 16)
                    probe = torch.Generator(device=device).manual_seed(seed)
                    loss = flow_loss(unwrapped, [val_data[i]], settings, device, generator=probe, training=False, text_encoder=text_encoder)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite validation loss")
                    totals[0] += loss.double(); totals[1] += 1
            totals = accelerator.reduce(totals, reduction="sum")
            row = {"step": progress["step"], "flow_mse": (totals[0] / totals[1]).item(), "queries": int(totals[1])}
            if accelerator.is_main_process:
                with (output / "validation.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print({"validation": row}, flush=True)
        finally:
            unwrapped.train()
            restore_rng(rng, device, generator)

    if val:
        validate()
    started, start_step = time.monotonic(), progress["step"]
    torch.cuda.reset_peak_memory_stats(device)
    loss_sum = torch.zeros((), device=device)
    optimizer.zero_grad(set_to_none=True)
    finished = False
    class BatchSampler:
        batches = []
        def __iter__(self): return iter(self.batches)
        def __len__(self): return len(self.batches)
    sampler = BatchSampler()
    workers = settings.get("workers", 4)
    loader_generator = torch.Generator().manual_seed(settings["seed"])
    loader = torch.utils.data.DataLoader(train_data, batch_sampler=sampler, collate_fn=collate,
        num_workers=workers, pin_memory=True, generator=loader_generator,
        **({"prefetch_factor": settings.get("prefetch_factor", 2), "persistent_workers": True} if workers else {}))
    while progress["step"] < steps and not finished:
        order = training_order(size, settings, progress["epoch"], world)
        per_epoch = len(order) // (world * micro)
        sampler.batches = [order[(c * world + rank) * micro:(c * world + rank + 1) * micro]
                   for c in range(progress["cursor"], per_epoch)]
        for samples in loader:
            if progress["cursor"] % accumulation == 0:
                update_started = time.monotonic()
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss = flow_loss(model, samples, settings, device, generator=generator, text_encoder=text_encoder)
                finite = accelerator.reduce(torch.isfinite(loss).int(), reduction="sum")
                if finite.item() != world:
                    raise FloatingPointError("Nonfinite training loss on at least one rank")
                accelerator.backward(loss)
                loss_sum += loss.detach()
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(parameters, settings.get("max_grad_norm", 1.0))
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite adapter gradients")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            progress["cursor"] += 1
            progress["global_samples_seen"] += world * micro
            if not accelerator.sync_gradients:
                continue
            progress["step"] += 1
            mean_loss = accelerator.reduce(loss_sum / accumulation, reduction="mean").item()
            loss_sum.zero_()
            if progress["cursor"] == per_epoch:
                progress["epoch"] += 1; progress["cursor"] = 0
            update_seconds = time.monotonic() - update_started
            row = {**progress, "loss": mean_loss, "update_compute_seconds": update_seconds,
                   "update_samples_per_second": world * micro * accumulation / update_seconds, "elapsed_seconds": time.monotonic() - started,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                   "samples_per_second": (progress["step"] - start_step) * world * micro * accumulation / (time.monotonic() - started)}
            if accelerator.is_main_process:
                with (output / "loss.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(row, flush=True)
            if val and (progress["step"] % settings["validate_every"] == 0 or progress["step"] == steps):
                validate()
            if progress["step"] % settings["checkpoint_every"] == 0 or progress["cursor"] == 0:
                save(f"step_{progress['step']:08d}")
            requested = accelerator.reduce(torch.tensor(int(stop["requested"]), device=device), reduction="sum").item()
            if requested:
                finished = True
            if finished or progress["step"] >= steps or progress["cursor"] == 0:
                break
    del loader
    save("interrupted" if finished else "final")
    for sig, handler in handlers.items():
        signal.signal(sig, handler)
    accelerator.end_training()
