"""Separate upsampler and encoder stages so training never holds either model."""
from pathlib import Path
import hashlib
import itertools
from .common import digest, file_hash, read_json, write_json, within
from .data import Level1Data


def dataset(config, args):
    return Level1Data(config, args.split, scenes=args.scene, max_episodes=args.max_episodes,
                      selection=args.selection)


def queries(data, args):
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be positive")
    return list(itertools.islice(data.queries(), args.max_samples)) if args.max_samples else list(data.queries())


def input_key(condition):
    images = [hashlib.sha256(im.tobytes()).hexdigest() for im in condition["images"]]
    return digest({"prompt": condition["prompt"], "images": images})


class PromptUpsampler:
    """Official weights, system prompt, reference policy and scaffold via Transformers.

    Greedy decoding matches official temperature=0. The exact geometry instruction is
    appended unchanged so VLM paraphrasing cannot silently delete camera numbers.
    """
    def __init__(self, config, device):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from .runtime import model_paths
        root = model_paths(config, pup=True)
        self.processor = AutoProcessor.from_pretrained(root, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.text_processor = AutoProcessor.from_pretrained(model_paths(config) / "text_encoder", local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            root, dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
        self.prompts = read_json(root / "system_prompts.json")
        self.device, self.config = device, config

    def __call__(self, condition):
        return self.batch([condition])[0]["prompt"]

    def batch(self, conditions):
        import torch
        from flux3.pup import policy_reference, I2I_SCAFFOLD
        from flux3.bbox import plan
        messages = []
        for condition in conditions:
            images = [policy_reference(im, len(condition["images"])) for im in condition["images"]]
            key, instruction, locked = plan(condition["prompt"], None, len(images))
            if locked is not None:
                raise ValueError("Bounding boxes are outside this baseline")
            messages.append([{"role": "system", "content": self.prompts[key]}, {"role": "user", "content":
                [{"type": "image", "image": im} for im in images] +
                [{"type": "text", "text": I2I_SCAFFOLD[False].format(instruction=instruction)}]}])
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt", return_dict=True, processor_kwargs={"padding": True}).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.config["prompt"]["max_new_tokens"])
        prompts = []
        for tokens, condition in zip(output, conditions):
            generated = tokens[inputs["input_ids"].shape[1]:]
            eos = self.model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            if not any(t in eos for t in generated.tolist()):
                raise ValueError("Upsampler output hit its length budget; increase max_new_tokens in a new experiment")
            caption = self.processor.decode(generated, skip_special_tokens=True).strip()
            if not caption:
                raise ValueError("Empty prompt upsampler output")
            prompts.append(fit_caption(self.text_processor, caption, condition["prompt"]))
        return prompts


def fit_caption(processor, caption, exact):
    """Retain every exact camera constraint; explicitly record any caption shortening."""
    from flux3.text_encoder import MAX_LENGTH
    def tokens(prompt):
        formatted = processor.apply_chat_template([{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        return len(processor.tokenizer(formatted)["input_ids"])
    join = lambda text: text + "\n\nExact task and camera constraints (authoritative):\n" + exact
    if tokens(join("")) >= MAX_LENGTH:
        raise ValueError("Exact camera/robot constraints alone exceed the native text token budget")
    # Leave a small margin for tokenizer boundary effects; binary-search the
    # derived prose only. The original upsampler caption remains in the receipt.
    lo, hi = 0, len(caption)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tokens(join(caption[:mid])) <= MAX_LENGTH - 8: lo = mid
        else: hi = mid - 1
    retained = caption[:lo].rstrip()
    if not retained:
        raise ValueError("No room for an upsampler caption; revise the exact-condition representation")
    prompt = join(retained)
    return {"prompt": prompt, "raw_upsampler_caption": caption,
        "caption_shortened_to_preserve_geometry": lo < len(caption), "text_tokens": tokens(prompt),
        "exact_constraints_sha256": digest(exact)}


def identity(config, data, kind, query_list):
    from .runtime import source_hash, environment_signature
    return {"schema": f"flux3-mvwd-{kind}-v1", "config_sha256": digest(config),
            "implementation_sha256": source_hash(), "dataset": data.provenance(),
            "query_support_sha256": digest(query_list), "ground_truth_loaded": kind == "features",
            "versions": environment_signature()}


def caption(config, args):
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
    from datetime import timedelta
    from .runtime import require_gpu
    from .cache import initialize, existing_row, complete
    require_gpu()
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=6))])
    data = dataset(config, args)
    selected = queries(data, args)
    root = Path(args.cache_root or args.output)
    info = identity(config, data, "captions", selected)
    lock = initialize(root, info, accelerator)
    pending = [i for i, q in enumerate(selected) if i % accelerator.num_processes == accelerator.process_index
               and existing_row(root, i, q) is None]
    upsampler = PromptUpsampler(config, accelerator.device) if pending else None
    batch_size = config["prompt"].get("batch_size", 4)
    for offset in range(0, len(pending), batch_size):
        indices = pending[offset:offset + batch_size]
        conditions = [data.conditions(selected[i]) for i in indices]
        prompts = upsampler.batch(conditions)
        for i, condition, prompt_record in zip(indices, conditions, prompts):
            query = selected[i]
            relative = f"prompts/{i:07d}.json"
            write_json(root / relative, {"query": query, "input_sha256": input_key(condition),
                **prompt_record, "policy": config["prompt"], "ground_truth_loaded": False})
            write_json(root / "rows" / f"{i:07d}.json", {"query_index": i, "query": query,
                "path": relative, "files": {relative: file_hash(root / relative)}})
        print(f"rank={accelerator.process_index} captioned={offset + len(indices)}/{len(pending)}", flush=True)
    complete(root, info, len(selected), accelerator)
    if Path(args.output) != root and accelerator.is_main_process:
        write_json(Path(args.output) / "cache.json", {"cache": str(root), "manifest_sha256": file_hash(root / "manifest.json")})
    accelerator.end_training()
    if lock: lock.close()


def prepare(config, args):
    import torch
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
    from datetime import timedelta
    from .runtime import require_gpu
    from .backend import load_encoders, encode_image, checked_text
    from .cache import initialize, existing_row, complete, save_tensor
    require_gpu()
    if args.split not in ("train", "val"):
        raise ValueError("Training feature caches may use train/val only")
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=6))])
    data = dataset(config, args)
    selected = queries(data, args)
    root = Path(args.cache_root or args.output)
    caproot = Path(args.captions)
    capinfo = read_json(caproot / "manifest.json")
    expected = identity(config, data, "captions", selected)
    if not capinfo["complete"] or any(capinfo.get(k) != v for k, v in expected.items()):
        raise ValueError("Caption support/config/source differs from feature selection")
    info = identity(config, data, "features", selected)
    info["caption_manifest_sha256"] = file_hash(caproot / "manifest.json")
    online = config.get("cache", {}).get("text_mode", "cached") == "online"
    info["text_mode"] = "online" if online else "cached"
    lock = initialize(root, info, accelerator)
    pending = [i for i, q in enumerate(selected) if i % accelerator.num_processes == accelerator.process_index
               and existing_row(root, i, q) is None]
    if pending:
        vae, text = load_encoders(config, accelerator.device, include_text=not online)
        # Avoid concurrent writers: each rank owns an identical unconditional object.
        uncond_path = None if online else f"objects/unconditional_{accelerator.process_index:04d}.pt"
        if uncond_path and not (root / uncond_path).exists():
            save_tensor(root / uncond_path, checked_text(text, ""))
    seen_objects = set()
    for i in pending:
        query = selected[i]
        condition = data.conditions(query)
        caprow = existing_row(caproot, i, query)
        if caprow is None:
            raise ValueError("Missing caption")
        cap = read_json(within(caproot, caprow["path"]))
        if cap["input_sha256"] != input_key(condition):
            raise ValueError("Condition changed since prompt upsampling")
        refs = []
        for image in condition["images"]:
            key = digest([list(image.size), image.mode, hashlib.sha256(image.tobytes()).hexdigest()])
            # Per-rank namespace avoids replace/mmap races; shared clean BEVs are still
            # deduplicated across all that rank's episodes/queries.
            relative = f"objects/rank_{accelerator.process_index:04d}/{key}.pt"
            if relative not in seen_objects:
                if not (root / relative).exists():
                    save_tensor(root / relative, encode_image(vae, image, accelerator.device))
                seen_objects.add(relative)
            refs.append(relative)
        sample = {"target": encode_image(vae, data.target(query), accelerator.device),
                  "text": None if online else checked_text(text, cap["prompt"]),
                  "prompt": cap["prompt"], "reference_paths": refs,
                  "unconditional_path": uncond_path, "query": query}
        relative = f"samples/{i:07d}.pt"
        save_tensor(root / relative, sample)
        files = {p: file_hash(root / p) for p in [relative, *refs, *([uncond_path] if uncond_path else [])]}
        write_json(root / "rows" / f"{i:07d}.json", {"query_index": i, "query": query,
            "path": relative, "files": files, "text_tokens": None if online else sample["text"][0].shape[0]})
        if i % 16 == accelerator.process_index:
            print(f"rank={accelerator.process_index} feature query={i}/{len(selected)}", flush=True)
    complete(root, info, len(selected), accelerator)
    if Path(args.output) != root and accelerator.is_main_process:
        write_json(Path(args.output) / "cache.json", {"cache": str(root), "manifest_sha256": file_hash(root / "manifest.json")})
    accelerator.end_training()
    if lock: lock.close()
