"""Native DiT adapters: frozen BF16 backbone, trainable FP32 low-rank weights."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # Autocast performs BF16 GEMMs; master adapter weights and Adam moments remain FP32.
        return self.base(x) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scale


def inject(model, rank, alpha):
    model.requires_grad_(False)
    targets = {"q_proj", "k_proj", "v_proj", "attn_out"}
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in targets]
    if not names:
        raise ValueError("No native DiT attention projections found")
    for name in names:
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, LoRALinear(model.get_submodule(name), rank, alpha))
    return names


def adapter_state(model):
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
            if k.endswith(("lora_A", "lora_B"))}


def load_adapter(model, state):
    expected = adapter_state(model)
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
        raise ValueError("Adapter keys/shapes differ from the configured native DiT")
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Adapter contains nonfinite values")
    model.load_state_dict(state, strict=False)


def gradient_checkpointing(model):
    # Wrap forward without changing state-dict names. Non-reentrant checkpointing also
    # computes adapter gradients when the initial frozen inputs do not require gradients.
    from flux3.dit import Block
    for module in model.modules():
        if isinstance(module, Block):
            forward = module.forward
            def wrapped(*args, _forward=forward, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                return _forward(*args, **kwargs)
            module.forward = wrapped
