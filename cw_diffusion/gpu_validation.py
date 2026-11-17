"""Explicit pretrained GPU acceptance; no optimizer training trajectory is created."""

from pathlib import Path
import json, time
import torch
from .config import WorldConfig
from .data import CanonicalDataset
from .system import CanonicalSystem
from .workflow import move, load_teacher
from .checkpoint import source_identity, environment_identity
from flux_mvwd.common import read_json, file_hash, write_json


def validate(config_path, output, ddp=False):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    c = read_json(config_path)
    base = Path(c["baseline_experiment"])
    config = read_json(base / "config.json")
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    acc = (
        Accelerator(
            mixed_precision="bf16",
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ],
        )
        if ddp
        else None
    )
    rank = acc.process_index if acc else 0
    world_size = acc.num_processes if acc else 1
    if ddp and world_size != 4:
        raise ValueError("DDP acceptance requires four ranks")
    device = acc.device if acc else torch.device("cuda")
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if acc:
        acc.wait_for_everyone()
    report = {
        "complete": False,
        "sources": source_identity(),
        "environment": environment_identity(),
        "config_sha256": file_hash(config_path),
        "gpu": torch.cuda.get_device_name(),
        "torch_cuda": torch.version.cuda,
        "training_run_started": False,
        "rank": rank,
        "world_size": world_size,
    }
    if rank == 0:
        write_json(output / "validation.json", report)
    start = time.monotonic()
    from flux_mvwd.backend import load_dit, load_text_encoder, checked_text
    from flux_mvwd.lora import inject, load_adapter
    from flux_mvwd.cache import FeatureDataset
    from safetensors.torch import load_file
    from flux3.pipeline import pack

    data = CanonicalDataset(config, "train", max_episodes=1)
    group_index = rank % len(data.groups)
    condition = data.conditions(group_index)
    condition.pop("native_conditions")
    target = data.supervision(group_index)
    root = base / "features/train"
    manifest = read_json(root / "manifest.json")
    cache = FeatureDataset(root, manifest["samples"])
    q = data.groups[group_index]
    indices = []
    for i, row in enumerate(manifest["samples"]):
        query = row["query"]
        if (
            query["episode_id"] == q["episode_id"]
            and query["frame_index"] == q["frame_index"]
        ):
            indices.append(i)
    indices.sort(key=lambda i: manifest["samples"][i]["query"]["robots"][0])
    assert len(indices) == 3
    samples = [cache[i] for i in indices]
    teacher, teacher_identity = load_teacher(device)
    with torch.no_grad():
        rgb = target["rgb"].to(device)
        mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        target["teacher"] = teacher.get_intermediate_layers((rgb - mean) / std, n=1)[0][
            :, 1:
        ].cpu()
    del teacher
    torch.cuda.empty_cache()
    target.update(rgb_K=condition["K"], rgb_c2w=condition["c2w"])
    condition, samples, target = (
        move(condition, device),
        move(samples, device),
        move(target, device),
    )
    backbone = load_dit(config, device)
    inject(backbone, config["training"]["rank"], config["training"]["alpha"])
    load_adapter(
        backbone, load_file(str(Path(c["baseline_checkpoint"]) / "lora.safetensors"))
    )
    model = CanonicalSystem(backbone, WorldConfig(**c["world"])).to(device)
    text = load_text_encoder(config, device)
    model.eval()
    entries = []
    with torch.no_grad():
        for sample in samples:
            ref = []
            ids = []
            for i, (a, b) in enumerate(sample["references"]):
                b = b.clone()
                b[:, 3] = -(i + 1)
                ref.append(a)
                ids.append(b)
            entries.append(
                [
                    checked_text(text, sample["prompt"], cpu=False),
                    (torch.cat(ref), torch.cat(ids)),
                    sample["target"],
                ]
            )
        packed = pack(entries)
        packed.timesteps["image"].fill_(0.5)
        from .native import SceneContext

        initial = model.world(
            condition["bev"],
            condition["world_to_bev"],
            condition["robot_states"],
            generator=torch.Generator(device=device).manual_seed(1),
        )
        context = SceneContext(
            [initial.state],
            condition["K"][None],
            condition["c2w"][None],
            tuple(config["target_hw"]),
            torch.tensor(0.5, device=device),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = backbone(packed)
            actual, states, _ = model.dit(packed, context)
        error = (reference.float() - actual.float()).abs().max().item()
        if error != 0:
            raise AssertionError(
                f"Zero adapter differs from pretrained baseline: {error}"
            )
    del reference, actual, packed, entries, states, initial, context
    torch.cuda.empty_cache()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device=device).manual_seed(9 + rank)
    train_model = acc.prepare(model) if acc else model
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, parts = train_model(condition, samples, target, text, generator, "joint")
    if acc:
        acc.backward(loss)
    else:
        loss.backward()
    torch.cuda.synchronize()
    bad = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    if bad:
        raise FloatingPointError("Nonfinite parameter gradients: " + str(bad))
    norms = {
        n: float(p.grad.norm())
        for n, p in model.named_parameters()
        if p.grad is not None
        and ("feedback" in n and n.endswith("out.weight") or n == "world.field.weight")
    }
    if not norms or not all(x > 0 for x in norms.values()):
        raise AssertionError("Missing active world/adapter gradient")
    if acc:
        vector = torch.tensor([norms[n] for n in sorted(norms)], device=device)
        all_norms = acc.gather(vector).reshape(world_size, -1)
        difference = (all_norms - all_norms[0]).abs().max().item()
        if difference > 1e-6:
            raise AssertionError("DDP gradients are not synchronized")
        report["ddp_max_gradient_norm_difference"] = difference
    report.update(
        complete=True,
        zero_adapter_max_velocity_error=error,
        full_method_loss=float(loss),
        losses={k: float(v.detach()) for k, v in parts.items()},
        gradient_norms=norms,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        seconds=time.monotonic() - start,
        teacher=teacher_identity,
    )
    write_json(output / f"rank_{rank:04d}.json", report)
    if acc:
        acc.wait_for_everyone()
    if rank == 0:
        reports = [read_json(output / f"rank_{i:04d}.json") for i in range(world_size)]
        write_json(
            output / "validation.json",
            {
                **report,
                "ranks": reports,
                "complete": all(r["complete"] for r in reports),
            },
        )
        print(json.dumps(read_json(output / "validation.json"), indent=2))
    if acc:
        acc.end_training()
