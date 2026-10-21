"""Readiness and explicit pinned downloads, entirely under the workspace."""
from pathlib import Path
import importlib.metadata
import json
import os
from flux_mvwd.common import load_config, write_json, file_hash
from flux_mvwd.runtime import ROOT, WORKSPACE, check_upstream, model_paths


def doctor():
    import torch
    report = {"workspace": str(WORKSPACE), "repo": str(ROOT), "python": __import__("sys").executable,
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "torchvision", "transformers", "accelerate", "numpy", "safetensors")},
        "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "slurm_allocation": os.environ.get("SLURM_JOB_ID"), "upstream_commit": check_upstream()["commit"]}
    path = ROOT / "configs/level1.local.json"
    if path.exists():
        config = load_config(path)
        report["dataset_exists"] = (Path(config["dataset_root"]) / "dataset.json").exists()
        report["models"] = {}
        for pup in (False, True):
            try: report["models"]["pup" if pup else "base"] = str(model_paths(config, pup=pup))
            except Exception as error: report["models"]["pup" if pup else "base"] = type(error).__name__
    print(json.dumps(report, indent=2))
    return report


def download(args):
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    config = load_config(args.config)
    results = []
    for pup in (False, True):
        if args.only != "all" and (args.only == "pup") != pup: continue
        prefix = "upsampler_" if pup else ""
        repo, revision = (config["model"][prefix + k] for k in ("id", "revision"))
        root = WORKSPACE / "models/flux3_klein" / repo.split("/")[-1] / revision
        try:
            # Check access before starting a multi-GB transfer. Tokens stay inside
            # huggingface_hub's normal credential handling and are never logged.
            hf_hub_download(repo, "config.json", revision=revision, local_dir=root, local_files_only=False)
            patterns = None if pup else ["model.safetensors", "ae.safetensors", "text_encoder/*", "config.json", "manifest.json", "LICENSE.md", "NOTICE.md"]
            snapshot_download(repo, revision=revision, local_dir=root, allow_patterns=patterns,
                              local_files_only=False, max_workers=4)
            files = {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*")
                     if p.is_file() and ".cache" not in p.parts and p.name != "download_receipt.json"}
            write_json(root / "download_receipt.json", {"repo_id": repo, "revision": revision,
                "files": files, "total_bytes": sum(files.values()),
                "verification": "huggingface_hub immutable revision and LFS transport verification"})
            results.append({"repo": repo, "status": "ready", "path": str(root), "bytes": sum(files.values())})
        except GatedRepoError:
            results.append({"repo": repo, "status": "gated", "url": "https://huggingface.co/" + repo,
                "action": "Accept the model license with your Hugging Face account, then use hf auth login or HF_TOKEN and retry ./mvwd download"})
        except Exception as error:
            results.append({"repo": repo, "status": "failed", "error_type": type(error).__name__})
        print(json.dumps(results[-1], indent=2), flush=True)
    report_path = ROOT / "verification/model_downloads.json"
    previous = __import__("json").loads(report_path.read_text()) if report_path.exists() else []
    merged = {row["repo"]: row for row in [*previous, *results]}
    write_json(report_path, list(merged.values()))
    if any(row["status"] != "ready" for row in results):
        raise SystemExit(2)
