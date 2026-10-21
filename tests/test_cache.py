from types import SimpleNamespace
from pathlib import Path
import copy
import pytest
import torch
from flux_mvwd.cache import initialize, existing_row, complete, save_tensor, FeatureDataset
from flux_mvwd.common import read_json, write_json, file_hash


def test_resumable_cache_identity_checksums_and_mmap(tmp_path):
    acc = SimpleNamespace(is_main_process=True, wait_for_everyone=lambda: None)
    identity = {"schema": "test", "config": "a"}
    lock = initialize(tmp_path, identity, acc)
    query = {"id": 3}
    assert existing_row(tmp_path, 0, query) is None
    save_tensor(tmp_path / "reference.pt", (torch.ones(2, 8), torch.zeros(2, 4)))
    save_tensor(tmp_path / "uncond.pt", (torch.ones(1, 16), torch.zeros(1, 4)))
    save_tensor(tmp_path / "sample.pt", {"reference_paths": ["reference.pt"], "unconditional_path": "uncond.pt"})
    row = {"query_index": 0, "query": query, "path": "sample.pt",
           "files": {n: file_hash(tmp_path / n) for n in ["sample.pt", "reference.pt", "uncond.pt"]}}
    write_json(tmp_path / "rows/0000000.json", row)
    assert existing_row(tmp_path, 0, query) == row
    complete(tmp_path, identity, 1, acc)
    manifest = read_json(tmp_path / "manifest.json")
    assert manifest["complete"] and manifest["unique_bytes"] > 0
    assert len(FeatureDataset(tmp_path, [row])[0]["references"]) == 1
    lock.close()
    with pytest.raises(ValueError, match="identity changed"):
        initialize(tmp_path, {"schema": "other"}, acc)
    with pytest.raises(ValueError, match="query order"):
        existing_row(tmp_path, 0, {"id": 4})
    (tmp_path / "reference.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        existing_row(tmp_path, 0, query)
