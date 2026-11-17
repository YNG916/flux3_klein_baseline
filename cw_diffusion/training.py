"""Four-GPU coupled training with exact frame sampling, accumulation and full-state recovery."""

import math, json, time, signal
from pathlib import Path
import torch
from flux_mvwd.common import read_json, write_json, digest, file_hash
from flux_mvwd.training_state import epoch_indices
from .config import WorldConfig
from .system import CanonicalSystem
from .workflow import config_at, GroupCache, move
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_model,
    source_identity,
    environment_identity,
)


def build_system(p, c, device):
    from flux_mvwd.backend import load_dit
    from flux_mvwd.lora import inject, load_adapter
    from safetensors.torch import load_file

    base = read_json(p / "baseline_config.json")
    backbone = load_dit(base, device)
    inject(backbone, base["training"]["rank"], base["training"]["alpha"])
    load_adapter(
        backbone, load_file(str(Path(c["baseline_checkpoint"]) / "lora.safetensors"))
    )
    if c["training"]["stage"] != "joint":
        backbone.requires_grad_(False)
    model = CanonicalSystem(backbone, WorldConfig(**c["world"])).to(device)
    if c["training"]["stage"] == "fusion":
        model.world.requires_grad_(False)
    return model


def optimizer_groups(model, settings):
    groups = {"new": [], "backbone": []}
    for name, param in model.named_parameters():
        if param.requires_grad:
            groups["backbone" if name.startswith("dit.backbone.") else "new"].append(
                param
            )
    result = []
    for kind, params in groups.items():
        if params:
            result.append(
                {
                    "params": params,
                    "lr": settings["backbone_learning_rate"]
                    if kind == "backbone"
                    else settings["learning_rate"],
                    "group_kind": kind,
                }
            )
    if not result:
        raise ValueError("No trainable method parameters")
    return result


def train(experiment, name, resume=None, initialize_from=None, max_updates=None):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from flux_mvwd.backend import load_text_encoder

    p, c = config_at(experiment)
    if c["world"]["render_backend"] != "gsplat":
        raise ValueError(
            "Research training requires production gsplat; reference renderer is unit-test only"
        )
    settings = c["training"]
    accum = settings["accumulation"]
    acc = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=accum,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if acc.device.type != "cuda":
        raise ValueError(
            "Real pretrained training requires an explicitly authorized GPU allocation"
        )
    if acc.num_processes != settings.get("world_size", 4):
        raise ValueError(
            "Training rank count differs from declared plan; use the existing-allocation launcher"
        )
    set_seed(settings["seed"])
    device = acc.device
    dataset = GroupCache(p / "cache/train", c["baseline_experiment"])
    val = GroupCache(p / "cache/val", c["baseline_experiment"])
    if not dataset.meta.get("full_split") or not val.meta.get("full_split"):
        raise ValueError(
            "Formal method training requires complete train and val caches"
        )
    per_epoch = (
        math.ceil(len(dataset) / (acc.num_processes * accum))
        * acc.num_processes
        * accum
    )
    updates = per_epoch // (acc.num_processes * accum)
    budget = max_updates or settings["epochs"] * updates
    contract = {
        "sources": source_identity(),
        "environment": environment_identity(),
        "config_sha256": file_hash(p / "config.json"),
        "train_cache_sha256": file_hash(p / "cache/train/manifest.json"),
        "val_cache_sha256": file_hash(p / "cache/val/manifest.json"),
        "world_size": acc.num_processes,
        "accumulation": accum,
        "groups": len(dataset),
        "padded_groups": per_epoch,
        "target_updates": budget,
    }
    output = p / "training" / name
    if acc.is_main_process:
        output.mkdir(exist_ok=False)
    acc.wait_for_everyone()
    model = build_system(p, c, device)
    text = (
        None
        if settings["stage"] == "world"
        else load_text_encoder(read_json(p / "baseline_config.json"), device)
    )
    if initialize_from:
        load_model(initialize_from, model)
    params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, settings), weight_decay=0.01, fused=True
    )
    generator = torch.Generator(device=device).manual_seed(
        settings["seed"] + acc.process_index
    )
    progress = {"step": 0, "epoch": 0, "cursor": 0, "global_groups_seen": 0}
    if resume:
        progress = load_checkpoint(
            resume, model, optimizer, contract, acc.process_index, device, generator
        )
    model, optimizer = acc.prepare(model, optimizer)
    write_json(output / f"contract_rank_{acc.process_index:04d}.json", contract)
    stop = [False]

    def interrupt(*_):
        stop[0] = True

    old = {s: signal.signal(s, interrupt) for s in [signal.SIGINT, signal.SIGTERM]}
    started = time.monotonic()
    last_saved = -1

    def save(label):
        nonlocal last_saved
        save_checkpoint(
            output / label, model, optimizer, progress, contract, acc, generator
        )
        last_saved = progress["step"]

    def validate():
        raw = acc.unwrap_model(model)
        raw.eval()
        values = []
        # Fixed per-group RNG makes the validation probe independent of resume and rank order.
        with torch.no_grad():
            for i in range(acc.process_index, len(val), acc.num_processes):
                sample = move(val[i], device)
                g = torch.Generator(device=device).manual_seed(
                    int(digest([137, i])[:15], 16)
                )
                with acc.autocast():
                    loss, _ = raw(
                        sample["condition"],
                        sample["samples"],
                        sample["supervision"],
                        text,
                        g,
                        settings["stage"],
                    )
                values.append(loss.detach())
        result = torch.tensor(
            [sum(float(x) for x in values), len(values)],
            device=device,
            dtype=torch.float64,
        )
        result = acc.reduce(result, reduction="sum")
        if acc.is_main_process:
            with (output / "validation.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": progress["step"],
                            "loss": (result[0] / result[1]).item(),
                            "groups": int(result[1]),
                            "fixed_seed": 137,
                        }
                    )
                    + "\n"
                )
        raw.train()

    while progress["step"] < budget and not stop[0]:
        order = epoch_indices(
            len(dataset), settings["seed"], progress["epoch"], acc.num_processes, accum
        )
        local = order[acc.process_index :: acc.num_processes]
        cursor = progress["cursor"]

        class Ordered(torch.utils.data.Dataset):
            def __len__(self):
                return len(local) - cursor

            def __getitem__(self, i):
                return dataset[local[cursor + i]]

        loader = torch.utils.data.DataLoader(
            Ordered(),
            batch_size=1,
            shuffle=False,
            num_workers=settings["workers"],
            pin_memory=True,
            collate_fn=lambda x: x,
            **({"prefetch_factor": 2} if settings["workers"] else {}),
        )
        for batch in loader:
            sample = move(batch[0], device)
            for param_group in optimizer.param_groups:
                if param_group["group_kind"] == "backbone":
                    warmup = max(1, settings.get("backbone_warmup_updates", 1))
                    param_group["lr"] = settings["backbone_learning_rate"] * min(
                        1.0, progress["step"] / warmup
                    )
            with acc.accumulate(model):
                with acc.autocast():
                    loss, parts = model(
                        sample["condition"],
                        sample["samples"],
                        sample["supervision"],
                        text,
                        generator,
                        settings["stage"],
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite method loss")
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            progress["cursor"] += 1
            if acc.sync_gradients:
                progress["step"] += 1
                progress["global_groups_seen"] += acc.num_processes * accum
                if progress["cursor"] == len(local):
                    progress["epoch"] += 1
                    progress["cursor"] = 0
                reduced = acc.reduce(loss.detach(), reduction="mean")
                if acc.is_main_process:
                    row = {
                        **progress,
                        "loss": float(reduced),
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                        / 2**30,
                        "losses": {k: float(v.detach()) for k, v in parts.items()},
                    }
                    with (output / "loss.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(row, flush=True)
                if progress["step"] % settings["checkpoint_every"] == 0:
                    save(f"step_{progress['step']:08d}")
                if progress["cursor"] == 0 and settings.get(
                    "validate_each_epoch", True
                ):
                    validate()
                if stop[0] or progress["step"] >= budget:
                    break
        del loader
    save("final" if progress["step"] >= budget else "interrupted")
    for sig, handler in old.items():
        signal.signal(sig, handler)
    acc.end_training()
