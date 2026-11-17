"""Atomic, complete-state method checkpoints, independently fingerprinted from baseline."""

from pathlib import Path
import torch
from flux_mvwd.common import file_hash, write_json, read_json
from flux_mvwd.cache import save_tensor
from flux_mvwd.training_state import capture_rng, restore_rng

ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    paths = [
        *sorted((ROOT / "cw_diffusion").glob("*.py")),
        ROOT / "requirements-cw.txt",
        ROOT / "UPSTREAM.lock.json",
    ]
    paths += sorted((ROOT / "flux_mvwd").glob("*.py"))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def environment_identity():
    import importlib.metadata as m

    versions = {}
    for name in [
        "torch",
        "transformers",
        "accelerate",
        "gsplat",
        "numpy",
        "scipy",
        "safetensors",
        "pillow",
        "ninja",
    ]:
        try:
            versions[name] = m.version(name)
        except m.PackageNotFoundError:
            versions[name] = None
    return versions


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    # Frozen baseline adapter weights are saved separately by the experiment's pinned initialization.
    return {
        n: p.detach().cpu()
        for n, p in model.state_dict().items()
        if n in names or not n.startswith("dit.backbone.")
    }


def save_checkpoint(
    destination, model, optimizer, progress, contract, accelerator, generator
):
    dest = Path(destination)
    if accelerator.is_main_process:
        if dest.exists():
            raise FileExistsError("Never overwrite a checkpoint: " + str(dest))
        dest.mkdir(parents=True)
        write_json(
            dest / "metadata.json",
            {
                "schema": "cw-training-state-v1",
                "complete": False,
                "contract": contract,
                "progress": progress,
            },
        )
    accelerator.wait_for_everyone()
    raw = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        save_tensor(dest / "model.pt", trainable_state(raw))
        save_tensor(dest / "optimizer.pt", optimizer.state_dict())
    save_tensor(
        dest / f"rank_{accelerator.process_index:04d}.pt",
        capture_rng(accelerator.device, generator),
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        files = {p.name: file_hash(p) for p in dest.glob("*.pt")}
        write_json(
            dest / "metadata.json",
            {
                "schema": "cw-training-state-v1",
                "complete": True,
                "contract": contract,
                "progress": progress,
                "files": files,
            },
        )
    accelerator.wait_for_everyone()


def load_checkpoint(directory, model, optimizer, contract, rank, device, generator):
    path = Path(directory)
    m = read_json(path / "metadata.json")
    if not m.get("complete") or m["contract"] != contract:
        raise ValueError("Checkpoint contract differs or incomplete")
    expected = {
        "model.pt",
        "optimizer.pt",
        *[f"rank_{i:04d}.pt" for i in range(contract["world_size"])],
    }
    if set(m["files"]) != expected:
        raise ValueError("Missing rank state")
    if any(file_hash(path / n) != sha for n, sha in m["files"].items()):
        raise ValueError("Checkpoint checksum differs")
    progress = m["progress"]
    local = contract["padded_groups"] // contract["world_size"]
    accum = contract["accumulation"]
    if (
        progress["cursor"] < 0
        or progress["cursor"] >= local
        or progress["cursor"] % accum
        or progress["step"]
        != progress["epoch"] * (local // accum) + progress["cursor"] // accum
        or progress["global_groups_seen"]
        != progress["step"] * accum * contract["world_size"]
    ):
        raise ValueError("Checkpoint progress is not a complete optimizer boundary")
    load_model(path, model)
    optimizer.load_state_dict(
        torch.load(path / "optimizer.pt", map_location=device, weights_only=True)
    )
    state = torch.load(
        path / f"rank_{rank:04d}.pt", map_location="cpu", weights_only=True
    )
    restore_rng(state, device, generator)
    return m["progress"]


def load_model(path, model, expected_contract=None):
    path = Path(path)
    metadata = read_json(path / "metadata.json")
    if not metadata.get("complete") or metadata.get("schema") != "cw-training-state-v1":
        raise ValueError("Incomplete method checkpoint")
    if expected_contract and any(
        metadata["contract"].get(key) != value
        for key, value in expected_contract.items()
    ):
        raise ValueError("Method checkpoint belongs to a different experiment")
    if file_hash(path / "model.pt") != metadata["files"].get("model.pt"):
        raise ValueError("Method model checksum differs")
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    expected = trainable_state(model)
    if set(state) != set(expected) or any(
        state[k].shape != expected[k].shape for k in expected
    ):
        raise ValueError("Method checkpoint keys/shapes differ")
    if any(
        v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()
    ):
        raise ValueError("Nonfinite checkpoint parameters")
    model.load_state_dict(state, strict=False)
