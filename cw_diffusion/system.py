"""Full coupled method, with native independent streams and shared synchronous-world feedback."""

import torch
from torch import nn
from .world import WorldEncoder
from .native import CanonicalDiT, SceneContext
from .losses import CanonicalLoss, masked_mean


class CanonicalSystem(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.cfg = cfg
        self.world = WorldEncoder(cfg)
        self.dit = CanonicalDiT(backbone, cfg)
        self.criterion = CanonicalLoss(cfg)

    def training_loss(
        self, condition, samples, supervision, text_encoder, generator, stage="joint"
    ):
        """Supervision is separate and never forwarded to world/feedback inference modules."""
        with torch.autocast(device_type=condition["bev"].device.type, enabled=False):
            pred = self.world(
                condition["bev"],
                condition["world_to_bev"],
                condition["robot_states"],
                generator=generator,
            )
            losses, mapping = self.criterion.layout(pred, supervision)
            losses.update(self.criterion.geometry(pred, supervision))
        if stage == "world":
            return self.criterion.total(losses), losses
        from flux3.pipeline import pack
        from flux_mvwd.backend import checked_text

        device = condition["bev"].device
        u = torch.sigmoid(torch.randn((), device=device, generator=generator))
        t = 16 * u / (1 + 15 * u)
        noise = [
            torch.randn(s["target"][0].shape, device=device, generator=generator)
            for s in samples
        ]
        x = [((1 - t) * s["target"][0].float() + t * n) for s, n in zip(samples, noise)]
        text_drop = torch.rand((), device=device, generator=generator) < 0.1
        data = []
        for s, current in zip(samples, x):
            text = (
                checked_text(text_encoder, "" if text_drop else s["prompt"], cpu=False)
                if s.get("text") is None
                else (s["unconditional"] if text_drop else s["text"])
            )
            refs = []
            ids = []
            for i, (tokens, pos) in enumerate(s["references"]):
                pos = pos.clone()
                pos[:, 3] = -(i + 1)
                refs.append(tokens)
                ids.append(pos)
            data.append(
                [text, (torch.cat(refs), torch.cat(ids)), (current, s["target"][1])]
            )
        state = pred.state
        flow = []
        diagnostic = None
        for step in range(self.cfg.unroll_steps):
            for entry, current in zip(data, x):
                entry[2] = (current.to(torch.bfloat16), entry[2][1])
            packed = pack(data)
            packed.timesteps["image"].fill_(t.detach().item())
            context = SceneContext(
                [state],
                condition["K"][None],
                condition["c2w"][None],
                tuple(supervision["rgb"].shape[-2:]),
                t,
            )
            velocity, states, aux = self.dit(packed, context)
            state = states[0]
            diagnostic = aux[-1][0]
            pieces = velocity.split([len(a) for a in x])
            flow.append(
                torch.stack(
                    [
                        (v.float() - (n - s["target"][0].float())).square().mean()
                        for v, n, s in zip(pieces, noise, samples)
                    ]
                ).mean()
            )
            next_t = t * 0.9
            x = [current + (next_t - t) * v.float() for current, v in zip(x, pieces)]
            t = next_t
        losses["flow"] = torch.stack(flow).mean()
        with torch.autocast(device_type=device.type, enabled=False):
            losses.update(
                self.criterion.appearance(diagnostic, supervision, pred, mapping)
            )
            gt = supervision["depth_linear"]
            valid = torch.isfinite(gt) & (gt > self.cfg.near) & (gt < self.cfg.far)
            # Depth head is in RGB cameras; register GT geometry instead of reusing an unregistered map.
            from .geometry import unproject, project, resize_intrinsics, sample_map

            h, w = diagnostic["depth"].shape[-2:]
            terms = []
            for i in range(3):
                xyz = unproject(
                    torch.where(valid[i], gt[i], torch.ones_like(gt[i])),
                    supervision["geometry_K"][i],
                    supervision["modality_c2w"]["depth_linear"][i],
                ).reshape(-1, 3)
                grid, z, inside = project(
                    xyz,
                    resize_intrinsics(
                        condition["K"][i], tuple(supervision["rgb"].shape[-2:]), (h, w)
                    ),
                    condition["c2w"][i],
                    h,
                    w,
                    self.cfg.near,
                    self.cfg.far,
                )
                estimate = sample_map(diagnostic["depth"][i : i + 1], grid[None])[
                    0, :, 0
                ]
                conf = sample_map(diagnostic["mvs_confidence"][i : i + 1], grid[None])[
                    0, :, 0
                ]
                good = valid[i].flatten() & inside
                terms.append(masked_mean((estimate - z).abs() * conf.detach(), good))
            losses["mvs_depth"] = torch.stack(terms).mean()
        # Supervise the refined world itself, so the geometry update cannot be an unused side head.
        from .rendering import render

        gt_depth = supervision["depth_linear"]
        shape = (gt_depth.shape[-2] // 4, gt_depth.shape[-1] // 4)
        clean = torch.nan_to_num(gt_depth, nan=0, posinf=0)
        valid = (
            torch.isfinite(gt_depth)
            & (gt_depth > self.cfg.near)
            & (gt_depth < self.cfg.far)
        )
        reference = torch.nn.functional.interpolate(
            clean[:, None], shape, mode="nearest-exact"
        )
        valid = (
            torch.nn.functional.interpolate(
                valid[:, None].float(), shape, mode="nearest-exact"
            )
            > 0.5
        )
        refined = render(
            state.geometry,
            resize_intrinsics(supervision["geometry_K"], gt_depth.shape[-2:], shape),
            supervision["modality_c2w"]["depth_linear"],
            *shape,
            self.cfg.render_backend,
            self.cfg.near,
            self.cfg.far,
        )
        losses["refined_depth"] = masked_mean(
            (
                refined.depth.clamp_min(0.01).log() - reference.clamp_min(0.01).log()
            ).abs(),
            valid,
        )
        losses["geometry_step"] = (
            (
                (state.geometry.means - state.prior_means)
                / state.geometry.scales.detach().clamp_min(0.01)
            )
            .square()
            .mean()
        )
        return self.criterion.total(losses), losses

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
