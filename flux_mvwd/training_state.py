"""Small resumable state; never serialize the frozen FLUX backbone."""
from __future__ import annotations

import math
import random
from pathlib import Path

from .common import digest, file_hash, read_json, within

SCHEMA = "flux3-mvwd-training-state-v3"


def epoch_indices(size, seed, epoch, world_size, accumulation, micro_batch=1, layouts=None):
    import torch
    if min(size, world_size, accumulation, micro_batch) < 1 or epoch < 0:
        raise ValueError("Invalid sampler dimensions")
    generator = torch.Generator().manual_seed(int(digest([seed, "epoch", epoch])[:16], 16))
    order = torch.randperm(size, generator=generator).tolist()
    if micro_batch > 1:
        if layouts is None or len(layouts) != size:
            raise ValueError("Microbatching requires one exact layout key per sample")
        from collections import defaultdict
        buckets = defaultdict(list)
        for index in order:
            buckets[layouts[index]].append(index)
        batches = []
        for bucket in buckets.values():
            count = math.ceil(len(bucket) / micro_batch) * micro_batch
            padded = (bucket * math.ceil(count / len(bucket)))[:count]
            batches.extend([padded[i:i + micro_batch] for i in range(0, len(padded), micro_batch)])
        shuffled = torch.randperm(len(batches), generator=generator).tolist()
        order = [i for b in shuffled for i in batches[b]]
    total = math.ceil(len(order) / (world_size * accumulation * micro_batch)) * world_size * accumulation * micro_batch
    return (order * math.ceil(total / len(order)))[:total]


def capture_rng(device, generator):
    import numpy as np
    import torch
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
            "torch": torch.get_rng_state(), "noise_generator": generator.get_state(),
            "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}


def restore_rng(state, device, generator):
    import numpy as np
    import torch
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["noise_generator"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def check_progress(progress, size, world_size, accumulation, micro_batch=1, padded_size=None):
    step, epoch, cursor = (progress[k] for k in ("step", "epoch", "cursor"))
    if any(type(v) is not int or v < 0 for v in (step, epoch, cursor)):
        raise ValueError("Invalid checkpoint progress")
    per_epoch = (padded_size // (world_size * micro_batch)) if padded_size is not None else math.ceil(size / (world_size * accumulation * micro_batch)) * accumulation
    if cursor >= per_epoch or cursor % accumulation or step != epoch * (per_epoch // accumulation) + cursor // accumulation:
        raise ValueError("Checkpoint progress is not an optimizer boundary")
    if progress["global_samples_seen"] != step * accumulation * world_size * micro_batch:
        raise ValueError("Checkpoint sample count mismatch")


def checkpoint_receipt(directory, contract):
    directory = Path(directory)
    meta = read_json(directory / "metadata.json")
    if meta.get("schema") != SCHEMA or not meta.get("complete"):
        raise ValueError("Checkpoint has no complete optimizer/RNG resume state; old LoRA-only checkpoints cannot resume")
    if meta["resume_contract"] != contract:
        raise ValueError("Resume requires identical config, feature caches, implementation, world size and validation protocol")
    expected = {"lora.safetensors", *[f"state_rank_{i:04d}.pt" for i in range(contract["world_size"]) ]}
    if set(meta["state_files"]) != expected:
        raise ValueError("Checkpoint state file list mismatch")
    for name, checksum in meta["state_files"].items():
        if file_hash(within(directory, name)) != checksum:
            raise ValueError("Checkpoint state checksum mismatch: " + name)
    check_progress(meta["progress"], contract["sample_count"], contract["world_size"], contract["accumulation"],
                   contract.get("micro_batch_size", 1), contract.get("padded_epoch_samples"))
    return meta
