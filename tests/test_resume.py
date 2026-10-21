"""Run the actual training loop with a small native DiT on CPU."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import torch

from test_data import fixture
from test_native import tiny_dit, sample
from flux_mvwd.cache import save_tensor
from flux_mvwd.common import digest, file_hash, write_json, read_json


def make_cache(path, config, split):
    from flux_mvwd.runtime import environment_signature
    rows = []
    for i in range(6):
        value = sample(i)
        refs = value.pop("references")
        uncond = value.pop("unconditional")
        names = [f"reference_{i}_{j}.pt" for j in range(2)]
        for name, tensor in zip(names, refs): save_tensor(path / name, tensor)
        value["reference_paths"] = names
        value["unconditional_path"] = f"uncond_{i}.pt"
        save_tensor(path / value["unconditional_path"], uncond)
        relative = f"sample_{i}.pt"; save_tensor(path / relative, value)
        files = {n: file_hash(path / n) for n in [relative, *names, value["unconditional_path"]]}
        rows.append({"query": {"id": i}, "path": relative, "files": files})
    write_json(path / "manifest.json", {"schema": "flux3-mvwd-features-v1", "complete": True,
        "config_sha256": digest(config), "implementation_sha256": "tiny-native-test", "samples": rows,
        "versions": environment_signature(),
        "dataset": {"split": split, "release_id": "fixture", "robot_assets": {"id": "fixture"}}})


def test_continuous_vs_full_state_resume_and_validation_rng(tmp_path):
    from flux_mvwd.training import train
    from safetensors.torch import load_file
    config = fixture(tmp_path / "dataset")
    config["training"].update(rank=2, alpha=2, micro_batch_size=1, gradient_accumulation_steps=2,
        workers=0, epochs=2, checkpoint_every=2, gradient_checkpointing=True,
        fused_optimizer=False, validate_every=1)
    cache = tmp_path / "cache"; cache.mkdir()
    val = tmp_path / "val"; val.mkdir()
    make_cache(cache, config, "train"); make_cache(val, config, "val")
    def args(name, steps, resume=None, validation=None):
        return SimpleNamespace(output=tmp_path / name, cache=cache, max_steps=steps,
            epochs=2, resume_from=resume, validation_cache=validation)
    with tiny_dit() as factory, patch("flux_mvwd.training.require_gpu"), \
            patch("flux_mvwd.training.source_hash", return_value="tiny-native-test"), \
            patch("flux_mvwd.training.load_dit", side_effect=lambda *_: factory()), \
            patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=0), \
            patch("torch.cuda.max_memory_reserved", return_value=0):
        train(config, args("continuous", 4))
        train(config, args("first", 1))
        train(config, args("resumed", 4, tmp_path / "first/final"))
        train(config, args("validated", 4, validation=val))
    expected = load_file(tmp_path / "continuous/final/lora.safetensors")
    for name in ("resumed", "validated"):
        actual = load_file(tmp_path / name / "final/lora.safetensors")
        for key in expected: torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    state = torch.load(tmp_path / "resumed/final/state_rank_0000.pt", weights_only=True)
    assert state["progress"] == {"step": 4, "epoch": 1, "cursor": 2, "global_samples_seen": 8}
    reference = torch.load(tmp_path / "continuous/final/state_rank_0000.pt", weights_only=True)
    for key, values in state["optimizer"]["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, reference["optimizer"]["state"][key][name], rtol=0, atol=0)
