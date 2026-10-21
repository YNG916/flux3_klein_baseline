from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def within(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("Expected relative path")
    result = (root / relative).resolve()
    result.relative_to(root)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def load_config(path):
    config = read_json(path)
    if config["mode"] not in ("independent", "joint"):
        raise ValueError("mode must be independent or joint")
    if any(int(x) != x or x <= 0 or x % 32 for x in config["target_hw"]):
        raise ValueError("Target dimensions must be positive multiples of 32")
    if config["bev_size"] < 256 or config["bev_size"] % 32:
        raise ValueError("BEV size must be a multiple of 32, at least 256")
    if config["mode"] == "joint" and config["robots"] != [0, 1, 2]:
        raise ValueError("Joint layout is fixed to robots [0,1,2]")
    if not config["robots"] or len(set(config["robots"])) != len(config["robots"]) or set(config["robots"]) - {0, 1, 2}:
        raise ValueError("Invalid robots")
    if not config["model"].get("revision") or config["model"]["revision"] == "main":
        raise ValueError("Pin a model revision (not main)")
    camera_variant = config.get("camera_conditioning", "matrix_text")
    if camera_variant not in ("matrix_text", "bev_frustum_text_v1"):
        raise ValueError("Unknown camera_conditioning variant")
    if camera_variant == "bev_frustum_text_v1" and config["mode"] != "independent":
        raise ValueError("BEV camera cues require independent mode")
    settings = config["training"]
    if settings["rank"] < 1 or settings["alpha"] != settings["rank"]:
        raise ValueError("First implementation requires alpha == rank > 0 for portable LoRA scaling")
    if any(settings[k] < 1 for k in ("gradient_accumulation_steps", "max_steps", "checkpoint_every")):
        raise ValueError("Training counts must be positive")
    if type(settings.get("micro_batch_size", 1)) is not int or settings.get("micro_batch_size", 1) < 1:
        raise ValueError("micro_batch_size must be a positive integer")
    if type(settings.get("gradient_checkpointing", True)) is not bool:
        raise ValueError("gradient_checkpointing must be boolean")
    if config["model"]["id"] != "black-forest-labs/flux-3-image-klein-base":
        raise ValueError("Train the Base model, not the distilled flow map")
    for key in ("revision", "upsampler_revision", "upstream_commit"):
        value = config["model"][key]
        if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Pin a full immutable commit: " + key)
    if not 0 <= settings.get("cfg_dropout", 0.1) < 1:
        raise ValueError("CFG dropout must be in [0, 1)")
    if settings.get("shift", 16) <= 0 or settings.get("logit_std", 1) <= 0:
        raise ValueError("Invalid flow timestep distribution")
    if config.get("prompt", {}).get("policy") != "official_pup_plus_exact_geometry_v1":
        raise ValueError("Use the recorded official upsampler policy")
    if config.get("cache", {}).get("text_mode", "cached") not in ("online", "cached"):
        raise ValueError("cache.text_mode must be online or cached")
    if config["prompt"].get("batch_size", 4) < 1:
        raise ValueError("Prompt batch size must be positive")
    if settings.get("timestep_sampling", "logit_normal") not in ("logit_normal", "uniform"):
        raise ValueError("Unknown timestep sampling strategy")
    if settings.get("workers", 4) < 0 or settings.get("epochs", 10) < 1 or settings.get("validate_every", 1000) < 1:
        raise ValueError("Invalid worker/epoch/validation counts")
    return config


def implementation_hash():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


def sample_seed(base, release, episode, frame, robots, mode):
    if mode == "independent":
        material = f"{base}:{release}:{episode}:{robots[0]}:{frame}"
    else:
        material = f"{base}:{release}:{episode}:joint:0,1,2:{frame}"
    return int(hashlib.sha256(material.encode()).hexdigest()[:8], 16)


def frame_selection(count, samples, episode, seed=42):
    if isinstance(samples, bool) or int(samples) != samples or not 1 <= samples <= count:
        raise ValueError("Invalid number of frames")
    result = []
    for slot in range(samples):
        lo, hi = slot * count // samples, (slot + 1) * count // samples
        material = f"{seed}:{episode}:frame-stratum:{slot}"
        result.append(lo + int(hashlib.sha256(material.encode()).hexdigest()[:16], 16) % (hi - lo))
    return result
