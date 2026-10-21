"""Native Base CFG inference, sharded per query; GT remains inaccessible."""
from pathlib import Path
import itertools
import json
import importlib.metadata
from .common import digest, file_hash, read_json, sample_seed, within, write_json
from .preparation import dataset, queries, input_key, identity


def infer(config, args):
    import torch
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
    from datetime import timedelta
    from safetensors.torch import load_file
    from flux3.pipeline import Pipeline
    from .runtime import require_gpu, source_hash
    from .backend import load_dit, load_encoders
    from .lora import inject, load_adapter
    from .cache import existing_row
    require_gpu()
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=6))])
    data = dataset(config, args)
    selected = queries(data, args)
    caproot = Path(args.captions)
    captions = read_json(caproot / "manifest.json")
    expected = identity(config, data, "captions", selected)
    if not captions.get("complete") or any(captions.get(k) != v for k, v in expected.items()):
        raise ValueError("Caption manifest differs from generation conditions/support")
    if not args.lora and not args.zero_shot:
        raise ValueError("Pass --lora CHECKPOINT or explicitly choose --zero-shot")
    lora_hash = None
    if args.lora:
        checkpoint = Path(args.lora)
        info = read_json(checkpoint / "metadata.json")
        if not info.get("complete") or info["config_sha256"] != digest(config):
            raise ValueError("Incomplete or incompatible LoRA checkpoint")
        if info["resume_contract"]["implementation_sha256"] != source_hash():
            raise ValueError("LoRA implementation differs from generation source")
        for key in ("release_id", "robot_assets"):
            if info["dataset"][key] != data.provenance()[key]:
                raise ValueError("LoRA dataset/assets differ")
        lora_hash = file_hash(checkpoint / "lora.safetensors")
        if info["state_files"]["lora.safetensors"] != lora_hash:
            raise ValueError("LoRA checksum mismatch")
    output = Path(args.output)
    metadata = {"schema": "flux3-mvwd-predictions-v1", "complete": False,
        "method": "FLUX-3-Image-Klein-Base-" + config["mode"], "config": config,
        "dataset": data.provenance(), "base_seed": args.seed, "ground_truth_loaded": False,
        "lora_sha256": lora_hash, "implementation_sha256": source_hash(),
        "caption_manifest_sha256": file_hash(caproot / "manifest.json"),
        "seed_policy": "independent-query-sha256" if config["mode"] == "independent" else "joint-group-sha256",
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "accelerate", "safetensors")}}
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "metadata.json", metadata)
    accelerator.wait_for_everyone()
    model = load_dit(config, accelerator.device)
    if args.lora:
        inject(model, config["training"]["rank"], config["training"]["alpha"])
        load_adapter(model, load_file(str(checkpoint / "lora.safetensors")))
    model.eval()
    vae, text = load_encoders(config, accelerator.device)
    pipe = Pipeline.__new__(Pipeline)
    pipe.device, pipe.offload = accelerator.device, False
    pipe.dits, pipe.autoencoder, pipe.text_encoder = {"base": model}, vae, text
    h, w = config["target_hw"]
    rank_path = output / f"predictions_rank_{accelerator.process_index:04d}.jsonl"
    with rank_path.open("x") as stream:
        for i in range(accelerator.process_index, len(selected), accelerator.num_processes):
            query = selected[i]
            condition = data.conditions(query)
            caprow = existing_row(caproot, i, query)
            if caprow is None:
                raise ValueError("Missing caption")
            cap = read_json(within(caproot, caprow["path"]))
            if cap["input_sha256"] != input_key(condition):
                raise ValueError("Prompt caption belongs to different conditions")
            # Refuse truncation just as in feature preparation.
            from flux3.text_encoder import MAX_LENGTH
            formatted = text.processor.apply_chat_template([{"role": "user", "content": cap["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            if len(text.processor.tokenizer(formatted)["input_ids"]) > MAX_LENGTH:
                raise ValueError("Generation prompt would truncate exact geometry")
            seed = sample_seed(args.seed, data.meta["release_id"], query["episode_id"], query["frame_index"], query["robots"], config["mode"])
            with torch.inference_mode():
                image = pipe.generate("base", cap["prompt"], condition["images"], w * len(query["robots"]), h,
                    config["inference"]["steps"], config["inference"]["guidance"], seed,
                    alpha=config["inference"]["shift"])
            if image.mode != "RGB" or image.size != (w * len(query["robots"]), h):
                raise ValueError("Native decoder output differs from protocol")
            for slot, robot in enumerate(query["robots"]):
                relative = Path(query["episode_id"]) / f"robot_{robot:02d}" / f"frame_{query['frame_index']:03d}.png"
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image.crop((slot*w, 0, (slot+1)*w, h)).save(path)
                row = {"episode_id": query["episode_id"], "robot_id": robot, "frame_index": query["frame_index"],
                    "seed": seed, "path": str(relative), "group_robots": query["robots"], "query_index": i,
                    "sha256": file_hash(path), "input_sha256": cap["input_sha256"], "prompt_sha256": digest(cap["prompt"])}
                write_json(path.with_suffix(".generation.json"), {**row, "prompt": cap["prompt"], "payload": condition["payload"],
                    "ground_truth_loaded": False})
                stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(f"rank={accelerator.process_index} generated query={i}/{len(selected)}", flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        rows = []
        for rank in range(accelerator.num_processes):
            rows.extend(json.loads(line) for line in (output / f"predictions_rank_{rank:04d}.jsonl").read_text().splitlines())
        rows.sort(key=lambda r: (r["query_index"], r["robot_id"]))
        keys = {(r["episode_id"], r["frame_index"], r["robot_id"]) for r in rows}
        count = sum(len(q["robots"]) for q in selected)
        if len(rows) != count or len(keys) != count:
            raise ValueError("Incomplete or duplicate prediction support")
        with (output / "predictions.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        metadata.update(complete=True, view_count=count)
        write_json(output / "metadata.json", metadata)
    accelerator.wait_for_everyone()
    accelerator.end_training()
