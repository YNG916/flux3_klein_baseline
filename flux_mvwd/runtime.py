"""Workspace paths and immutable upstream/model provenance."""
from pathlib import Path
import os

from .common import file_hash, read_json, digest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("MULTIVIEW_STORAGE_ROOT", ROOT.parent.parent))


def require_gpu():
    import torch
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("This GPU action requires an existing Slurm GPU allocation; use ./mvwd batch explicitly to submit")


def source_hash():
    files = [ROOT / "run.py", ROOT / "mvwd", *sorted((ROOT / "flux_mvwd").glob("*.py")),
             *sorted((ROOT / "tools").glob("*.py")), ROOT / "UPSTREAM.lock.json", ROOT / "requirements.lock.txt", ROOT / "pyproject.toml"]
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in files})


def environment_signature():
    import importlib.metadata
    result = {name: importlib.metadata.version(name) for name in
              ("torch", "torchvision", "transformers", "accelerate", "safetensors", "numpy", "pillow")}
    try: result["vllm"] = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError: pass
    return result


def check_upstream(config=None):
    lock = read_json(ROOT / "UPSTREAM.lock.json")
    if config is not None and lock["commit"] != config["model"]["upstream_commit"]:
        raise ValueError("Config/upstream commit differs")
    for name, sha in lock["files"].items():
        if file_hash(ROOT / "vendor/flux3" / name) != sha:
            raise ValueError("Upstream source changed: " + name)
    return lock


def model_paths(config, *, pup=False):
    from huggingface_hub import snapshot_download
    m = config["model"]
    prefix = "upsampler_" if pup else ""
    repo = m[prefix + "id"]
    revision = m[prefix + "revision"]
    root = WORKSPACE / "models/flux3_klein" / repo.split("/")[-1] / revision
    # Metadata is not proof that all shards were downloaded. Check the download receipt.
    receipt = root / "download_receipt.json"
    if receipt.exists():
        row = read_json(receipt)
        if row["revision"] != revision or row["repo_id"] != repo:
            raise ValueError("Model download receipt mismatch")
        for name, size in row["files"].items():
            if not (root / name).is_file() or (root / name).stat().st_size != size:
                raise ValueError("Incomplete model download: " + name)
        return root
    return Path(snapshot_download(repo_id=repo, revision=revision, local_files_only=True))
