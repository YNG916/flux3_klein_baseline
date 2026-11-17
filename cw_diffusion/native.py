"""Native DiT execution with explicit target-only feedback boundaries, no stateful hooks."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .feedback import CanonicalFeedback
from .geometry import resize_intrinsics


@dataclass
class SceneContext:
    states: list
    K: torch.Tensor  # [B,3,3,3], at target RGB resolution
    c2w: torch.Tensor  # [B,3,4,4], scene-local metric coordinates
    target_hw: tuple
    sigma: torch.Tensor  # scalar or [B]


class CanonicalDiT(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg.validate()
        hidden = backbone.txt_in.out_features
        if max(cfg.injection_blocks) >= len(backbone.single_blocks):
            raise ValueError("Feedback index exceeds native depth")
        self.feedback = nn.ModuleDict(
            {str(i): CanonicalFeedback(cfg, hidden) for i in cfg.injection_blocks}
        )

    def forward(self, packed, context=None):
        if context is None:
            return self.backbone(packed)
        from flux3.dit import MODES, timestep_embedding, rope_embed

        model = self.backbone
        if model.flow_map:
            raise ValueError("Canonical method requires Base, not distilled flow-map")
        if packed.num_samples != 3 * len(context.states):
            raise ValueError("Packed scene triplets differ from memory batch")
        fh, fw = context.target_hw[0] // 16, context.target_hw[1] // 16
        if any(b - a != fh * fw for a, b in packed.mode_indices["image"]):
            raise ValueError("Target token shape differs from native 16-pixel packing")
        data, ids, times = dict(packed.data), dict(packed.ids), dict(packed.timesteps)
        for mode in MODES:
            if len(data[mode]) == 0:
                data[mode] = data[mode].new_zeros(1, data[mode].shape[-1])
                ids[mode] = ids[mode].new_zeros(1, 4)
                times[mode] = times[mode].new_zeros(1)
        vec = {m: model.time_in(timestep_embedding(times[m])) for m in MODES}
        mk = lambda m: "txt" if m == "text" else m
        early = {m: model.early_stream_modulations[mk(m)](vec[m]) for m in MODES}
        single = {m: model.single_stream_modulations[mk(m)](vec[m]) for m in MODES}
        x = {
            m: (model.txt_in if m == "text" else model.emb_in[m])(data[m])
            for m in MODES
        }
        freqs = {m: rope_embed(ids[m][None]) for m in MODES}

        def blockrun(block, x, freq, bounds, mod):
            if self.training and self.cfg.checkpoint_blocks:
                return checkpoint(block, x, freq, bounds, mod, use_reentrant=False)
            return block(x, freq, bounds, mod)

        for d in range(len(model.txt_mode_blocks)):
            for m in ["image", "image_cond"]:
                x[m] = blockrun(
                    model.content_mode_blocks[m][d],
                    x[m],
                    freqs[m],
                    packed.mode_indices[m],
                    early[m],
                )
            x["text"] = blockrun(
                model.txt_mode_blocks[d],
                x["text"],
                freqs["text"],
                packed.mode_indices["text"],
                early["text"],
            )
        chunks = []
        idchunks = []
        mods = ([], [], [])
        bounds = []
        placement = []
        offset = 0
        target_positions = []
        for s in range(packed.num_samples):
            start = offset
            for m in MODES:
                a, b = packed.mode_indices[m][s]
                if b > a:
                    chunks.append(x[m][a:b])
                    idchunks.append(ids[m][a:b])
                    for i in range(3):
                        mods[i].append(single[m][i][a:b])
                    placement.append((m, a, b, offset))
                    if m == "image":
                        target_positions.append(
                            torch.arange(offset, offset + b - a, device=x[m].device)
                        )
                    offset += b - a
            bounds.append((start, offset))
        hidden = torch.cat(chunks)
        freq = rope_embed(torch.cat(idchunks)[None])
        mods = tuple(torch.cat(m) for m in mods)
        positions = torch.stack(target_positions).reshape(-1)
        states = list(context.states)
        aux = []
        for index, block in enumerate(model.single_blocks):
            hidden = blockrun(block, hidden, freq, bounds, mods)
            if str(index) in self.feedback:
                targets = hidden[positions].reshape(len(states), 3, fh * fw, -1)
                updates = []
                stage = []
                for b, state in enumerate(states):
                    sigma = (
                        context.sigma if context.sigma.ndim == 0 else context.sigma[b]
                    )
                    K = resize_intrinsics(context.K[b], context.target_hw, (fh, fw))
                    # Geometry and memory arithmetic stay FP32, independent of BF16 DiT autocast.
                    with torch.autocast(device_type=hidden.device.type, enabled=False):
                        out, states[b], diagnostic = self.feedback[str(index)](
                            targets[b],
                            state,
                            K.float(),
                            context.c2w[b].float(),
                            (fh, fw),
                            sigma.float(),
                        )
                    updates.append(out.to(hidden))
                    stage.append(diagnostic)
                hidden = hidden.index_copy(
                    0, positions, torch.stack(updates).reshape(-1, hidden.shape[-1])
                )
                aux.append(stage)
        output = torch.empty_like(x["image"])
        for m, a, b, start in placement:
            if m == "image":
                output[a:b] = hidden[start : start + b - a]
        velocity = model.final_layer["image"](output, vec["image"])
        return velocity, states, aux
