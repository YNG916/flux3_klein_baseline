"""Explicit GPU benchmark: test complete optimizer updates, report measured throughput."""
from pathlib import Path
import gc
import time
from .common import read_json, digest, write_json


def profile(config, args):
    import torch
    from .runtime import require_gpu
    from .backend import load_dit, flow_loss, load_text_encoder
    from .lora import inject, gradient_checkpointing
    from .cache import FeatureDataset
    require_gpu()
    receipt = read_json(Path(args.cache) / "manifest.json")
    if not receipt.get("complete") or receipt["config_sha256"] != digest(config) or receipt["dataset"]["split"] != "train":
        raise ValueError("Use a complete training feature cache matching the config")
    if args.repeats < 1 or args.warmup < 1 or any(b < 1 for b in args.batches):
        raise ValueError("Profile counts must be positive")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    settings, results = config["training"], []
    device = torch.device("cuda")
    samples = FeatureDataset(args.cache, receipt["samples"])
    text_encoder = load_text_encoder(config, device) if receipt.get("text_mode") == "online" else None
    for batch in args.batches:
        model = optimizer = None
        row = {"micro_batch": batch, "gradient_checkpointing": settings["gradient_checkpointing"]}
        try:
            model = load_dit(config, device)
            inject(model, settings["rank"], settings["alpha"])
            if settings["gradient_checkpointing"]:
                gradient_checkpointing(model)
            model.train()
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=settings["learning_rate"], fused=True)
            data = [samples[i % len(samples)] for i in range(batch)]
            times = []
            torch.cuda.reset_peak_memory_stats()
            for step in range(args.warmup + args.repeats):
                torch.cuda.synchronize(); started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = flow_loss(model, data, settings, device, text_encoder=text_encoder)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite profile loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                if step >= args.warmup:
                    times.append(time.perf_counter() - started)
            row.update(status="ok", seconds_per_update=sum(times)/len(times),
                samples_per_second=batch * len(times)/sum(times),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        except torch.cuda.OutOfMemoryError:
            row.update(status="oom")
        finally:
            del model, optimizer
            if "params" in locals(): del params
            if "loss" in locals(): del loss
            gc.collect(); torch.cuda.empty_cache()
        results.append(row)
        write_json(output / "profile.json", {"device": torch.cuda.get_device_name(), "config": config,
            "results": results, "scope": "single-GPU complete optimizer updates, not a four-GPU scaling measurement"})
        print(row, flush=True)
