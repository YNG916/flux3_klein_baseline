"""Exercise real native DiT autograd, packed isolation, LoRA and checkpointing."""
from contextlib import contextmanager
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
import pytest

from flux_mvwd.backend import flow_loss, configure_attention
from flux_mvwd.lora import inject, adapter_state, load_adapter, gradient_checkpointing


@contextmanager
def tiny_dit():
    import flux3.dit as d
    values = {"HIDDEN": 32, "HEADS": 1, "DEPTH": 1, "DEPTH_SINGLE": 1, "AXES_DIM": (8, 8, 8, 8), "MLP_RATIO": 2.0}
    with patch.multiple(d, **values), patch.dict(d.IN_CHANNELS, image=8, image_cond=8, text=16):
        configure_attention("math")
        yield lambda: d.DiT(False).to(torch.bfloat16)


def sample(index=0):
    g = torch.Generator().manual_seed(100 + index)
    pair = lambda n, c: (torch.randn(n, c, generator=g).to(torch.bfloat16), torch.zeros(n, 4))
    return {"target": pair(3, 8), "text": pair(2 + index % 2, 16), "unconditional": pair(1, 16),
            "references": [pair(2, 8), pair(2, 8)], "query": {"id": index}}


def test_real_dit_lora_gradient_and_clean_reference_contract():
    with tiny_dit() as factory:
        model = factory()
        names = inject(model, 2, 2)
        assert len(names) == 16
        s = sample(); original = copy.deepcopy(s)
        seen = []
        handle = model.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
        loss = flow_loss(model, [s], {"cfg_dropout": 0.0}, "cpu", generator=torch.Generator().manual_seed(3))
        loss.backward(); handle.remove()
        assert torch.isfinite(loss)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in trainable)
        assert any(p.grad.abs().sum() > 0 for n, p in trainable if n.endswith("lora_B"))
        assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
        packed = seen[0]
        assert packed.timesteps["image"].min() > 0
        assert not packed.timesteps["image_cond"].any() and not packed.timesteps["text"].any()
        torch.testing.assert_close(packed.data["image_cond"], torch.cat([p[0] for p in s["references"]]))
        for a, b in zip(s["references"], original["references"]):
            for x, y in zip(a, b): torch.testing.assert_close(x, y)


def test_packed_samples_do_not_attend_to_each_other():
    from flux3.pipeline import pack
    with tiny_dit() as factory:
        model = factory().eval()
        a, b = sample(0), sample(1)
        reference = lambda s: (torch.cat([p[0] for p in s["references"]]), torch.cat([p[1] for p in s["references"]]))
        with torch.no_grad():
            one = model(pack([[a["text"], reference(a), a["target"]]]))
            both = model(pack([[a["text"], reference(a), a["target"]], [b["text"], reference(b), b["target"]]]))
        torch.testing.assert_close(one, both[:3], rtol=0.02, atol=0.02)


def test_nonreentrant_checkpoint_preserves_adapter_gradient():
    with tiny_dit() as factory:
        torch.manual_seed(8); model = factory(); inject(model, 2, 2)
        checkpointed = copy.deepcopy(model); gradient_checkpointing(checkpointed)
        settings = {"cfg_dropout": 0.0}
        for candidate in (model, checkpointed):
            flow_loss(candidate, [sample()], settings, "cpu", generator=torch.Generator().manual_seed(4)).backward()
        for a, b in zip(model.parameters(), checkpointed.parameters()):
            if a.requires_grad: torch.testing.assert_close(a.grad, b.grad)


def test_lora_roundtrip_and_shape_rejection():
    with tiny_dit() as factory:
        model = factory(); inject(model, 2, 2)
        state = adapter_state(model)
        load_adapter(model, state)
        with pytest.raises(ValueError): load_adapter(model, {})
        bad = dict(state); bad[next(iter(bad))] = torch.zeros(1)
        with pytest.raises(ValueError): load_adapter(model, bad)


def test_online_text_encoder_produces_normal_frozen_features():
    from types import SimpleNamespace
    class Encoder:
        processor = SimpleNamespace(apply_chat_template=lambda *a, **k: "prompt",
            tokenizer=lambda *a, **k: {"input_ids": [1, 2]})
        @torch.inference_mode()
        def encode(self, prompt):
            return torch.ones(2, 16, dtype=torch.bfloat16), torch.zeros(2, 4)
    with tiny_dit() as factory:
        model = factory(); inject(model, 2, 2)
        s = sample(); s.update(text=None, prompt="geometry")
        s.pop("unconditional")
        loss = flow_loss(model, [s], {"cfg_dropout": 0.0}, "cpu", text_encoder=Encoder())
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
