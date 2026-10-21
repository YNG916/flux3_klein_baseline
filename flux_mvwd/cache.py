"""Atomic, checksummed, resumable caches; reference images use content addressing."""
from pathlib import Path
import os
import tempfile
import torch
from .common import digest, file_hash, read_json, write_json, within


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = stream.name
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def initialize(root, identity, accelerator):
    root = Path(root)
    lock = None
    if accelerator.is_main_process:
        import fcntl
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / ".writer.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_file = root / "identity.json"
        if identity_file.exists():
            if read_json(identity_file) != identity:
                raise ValueError("Cache identity changed; use another cache directory")
        else:
            write_json(identity_file, identity)
    accelerator.wait_for_everyone()
    if read_json(root / "identity.json") != identity:
        raise ValueError("Cache identity differs across workers")
    return lock


def existing_row(root, index, query):
    path = Path(root) / "rows" / f"{index:07d}.json"
    if not path.exists():
        return None
    row = read_json(path)
    if row["query"] != query or row["query_index"] != index:
        raise ValueError("Cache query order changed")
    for relative, sha in row["files"].items():
        if file_hash(within(root, relative)) != sha:
            raise ValueError("Cache checksum mismatch: " + relative)
    return row


def complete(root, identity, count, accelerator):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        rows = [read_json(Path(root) / "rows" / f"{i:07d}.json") for i in range(count)]
        if [r["query_index"] for r in rows] != list(range(count)):
            raise ValueError("Missing or duplicate cache query")
        unique_files = {k: v for r in rows for k, v in r["files"].items()}
        byte_count = sum(within(root, p).stat().st_size for p in unique_files)
        write_json(Path(root) / "manifest.json", {**identity, "complete": True,
            "samples": rows, "sample_count": count, "unique_bytes": byte_count,
            "mean_bytes_per_query": byte_count / max(1, count)})
    accelerator.wait_for_everyone()


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, root, rows, verify=True):
        self.root, self.rows, self.verify = Path(root), rows, verify
        self.verified = set()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        for name, sha in row["files"].items():
            if self.verify and name not in self.verified:
                if file_hash(within(self.root, name)) != sha:
                    raise ValueError("Feature checksum mismatch: " + name)
                self.verified.add(name)
        load = lambda p: torch.load(within(self.root, p), map_location="cpu", weights_only=True, mmap=True)
        sample = load(row["path"])
        sample["references"] = [load(name) for name in sample["reference_paths"]]
        if sample.get("unconditional_path"):
            sample["unconditional"] = load(sample["unconditional_path"])
        return sample


def collate(samples):
    return samples  # Native Packed supports variable token lengths without padding/buckets.
