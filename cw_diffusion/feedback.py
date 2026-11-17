from dataclasses import replace
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import project, sample_map
from .rendering import render, GaussianWorld
from .world import WorldState
from .mvs import NoiseAwareMVS


class CanonicalFeedback(nn.Module):
    def __init__(self, cfg, dit_hidden):
        super().__init__()
        self.cfg = cfg
        d = cfg.feature_dim
        self.heads = 8
        self.extract = nn.Sequential(nn.LayerNorm(dit_hidden), nn.Linear(dit_hidden, d))
        self.update = nn.Sequential(
            nn.Linear(2 * d + 2, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)
        )
        self.write_gate = nn.Sequential(
            nn.Linear(2 * d + 2, d), nn.SiLU(), nn.Linear(d, 1)
        )
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d + 2, d)
        self.v = nn.Linear(d + 2, d)
        self.out = nn.Linear(d, dit_hidden, bias=False)
        nn.init.zeros_(self.out.weight)
        self.read_gate = nn.Sequential(nn.Linear(d + 3, d), nn.SiLU(), nn.Linear(d, 1))
        self.view_residual = nn.Sequential(
            nn.Linear(d + 3, d), nn.SiLU(), nn.Linear(d, d)
        )
        nn.init.zeros_(self.view_residual[-1].weight)
        nn.init.zeros_(self.view_residual[-1].bias)
        self.geometry_gate = nn.Sequential(
            nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.mvs = NoiseAwareMVS(cfg)

    def forward(self, target_tokens, state, K, c2w, hw, sigma):
        v, n, hidden = target_tokens.shape
        h, w = hw
        d = self.cfg.feature_dim
        if v != 3 or n != h * w:
            raise ValueError("Three synchronous camera token grids required")
        features = (
            self.extract(target_tokens.float()).transpose(1, 2).reshape(v, d, h, w)
        )
        world = state.geometry
        prior = render(
            replace(world, features=state.confidence[:, None]),
            K,
            c2w,
            h,
            w,
            self.cfg.render_backend,
            self.cfg.near,
            self.cfg.far,
        )
        depths, mvs_conf, prob = self.mvs(
            features,
            K,
            c2w,
            prior.depth,
            prior.alpha,
            sigma,
            prior_confidence=prior.features,
        )
        grids, z, valid = project(
            world.means[None].expand(v, -1, -1),
            K,
            c2w,
            h,
            w,
            self.cfg.near,
            self.cfg.far,
        )
        observed = sample_map(features, grids)
        pd = sample_map(prior.depth, grids)[..., 0]
        pa = sample_map(prior.alpha, grids)[..., 0]
        tolerance = 0.05 + 0.1 * pd.abs()
        visibility = valid & ((pa < 0.05) | (z <= pd + tolerance))
        md = sample_map(depths, grids)[..., 0]
        mc = sample_map(mvs_conf, grids)[..., 0]
        center = c2w[:, :3, 3][:, None]
        direction = F.normalize(world.means[None] - center, dim=-1)
        angle = direction[..., 2].abs() * 0.25 + 0.75
        weights = (
            visibility.float()
            * angle
            * (1 - sigma).square()
            * (0.25 + 0.75 * state.confidence[None])
        )

        def update(previous, obs, weight):
            den = weight.sum(0)
            pooled = (obs * weight[..., None]).sum(0) / den[:, None].clamp_min(1e-6)
            evidence = torch.stack(
                (den.clamp_max(3) / 3, torch.ones_like(den) * (1 - sigma)), -1
            )
            x = torch.cat((previous, pooled, evidence), -1)
            gate = self.write_gate(x).sigmoid() * (den > 0)[:, None] * (1 - sigma)
            return previous + gate * self.update(x), den

        fused, den = update(world.features, observed, weights)
        new_loo = []
        previous_loo = getattr(state, "loo_features", None)
        for i in range(v):
            lw = weights.clone()
            lw[i] = 0
            previous = state.base_features if previous_loo is None else previous_loo[i]
            new_loo.append(update(previous, observed, lw)[0])
        # Bound surface refinement around immutable initial BEV anchors, never drift across objects.
        dz = (md - z) * mc * visibility
        delta = (c2w[:, :3, 2][:, None, :] * dz[..., None]).sum(0) / (
            (mc * visibility).sum(0)[:, None].clamp_min(1e-6)
        )
        evidence = torch.stack(
            (
                mc.mean(0),
                visibility.float().mean(0),
                state.confidence,
                torch.ones_like(den) * (1 - sigma),
            ),
            -1,
        )
        gate = (
            self.geometry_gate(evidence).sigmoid()
            * (1 - sigma).square()
            * (mc.sum(0) > 0)[:, None]
        )
        limit = self.cfg.geometry_step_fraction * world.scales.mean(-1, keepdim=True)
        means = world.means + gate * limit * torch.tanh(delta / limit.clamp_min(0.001))
        means = state.prior_means + torch.maximum(
            torch.minimum(means - state.prior_means, 4 * limit), -4 * limit
        )
        updated = GaussianWorld(
            means, world.quats, world.scales, world.opacity, fused, world.object_index
        )
        result = WorldState(
            updated,
            state.base_features,
            state.prior_means,
            (0.9 * state.confidence + 0.1 * (den / 3).clamp(0, 1)),
            state.age + 1,
        )
        result.loo_features = torch.stack(new_loo)
        rendered = []
        rend_depth = []
        rend_alpha = []
        invariant = []
        for i in range(v):
            # Each destination receives only other cameras' current AND historical appearance writes.
            loo = new_loo[i]
            view = self.view_residual(torch.cat((loo, direction[i]), -1)) * 0.1
            g = replace(updated, features=torch.cat((loo, loo + view), -1))
            r = render(
                g,
                K[i : i + 1],
                c2w[i : i + 1],
                h,
                w,
                self.cfg.render_backend,
                self.cfg.near,
                self.cfg.far,
            )
            invariant.append(r.features[:, :d])
            rendered.append(r.features[:, d:])
            rend_depth.append(r.depth)
            rend_alpha.append(r.alpha)
        rendered = torch.cat(rendered)
        alpha = torch.cat(rend_alpha)
        depth = torch.cat(rend_depth)
        source = (
            torch.cat((rendered, depth / self.cfg.far, alpha), 1)
            .flatten(2)
            .transpose(1, 2)
        )
        query = features.flatten(2).transpose(1, 2)
        # Add a zero-coverage null key so a wholly off-camera world cannot produce NaNs.
        source = torch.cat((source, source.new_zeros(v, 1, d + 2)), 1)
        split = lambda x: x.reshape(v, -1, self.heads, d // self.heads).transpose(1, 2)
        q, k, value = split(self.q(query)), split(self.k(source)), split(self.v(source))
        yy, xx = torch.meshgrid(
            torch.arange(h, device=q.device),
            torch.arange(w, device=q.device),
            indexing="ij",
        )
        xy = torch.stack((xx, yy), -1).reshape(n, 2).float()
        bias = -torch.cdist(xy, xy).square() / 8
        bias = F.pad(bias, (0, 1), value=0)[None, None].expand(v, 1, -1, -1).clone()
        coverage = F.pad(alpha.flatten(2)[:, 0], (0, 1), value=1) > 1e-5
        bias = bias.masked_fill(~coverage[:, None, None], float("-inf"))
        attended = (
            F.scaled_dot_product_attention(q, k, value, attn_mask=bias)
            .transpose(1, 2)
            .reshape(v, n, d)
        )
        gate_input = torch.cat(
            (
                query,
                depth.flatten(2).transpose(1, 2) / self.cfg.far,
                alpha.flatten(2).transpose(1, 2),
                torch.ones_like(query[..., :1]) * (1 - sigma),
            ),
            -1,
        )
        gate = self.read_gate(gate_input).sigmoid() * alpha.flatten(2).transpose(1, 2)
        output = target_tokens + self.out(attended).to(target_tokens) * gate.to(
            target_tokens
        )
        return (
            output,
            result,
            {
                "features": features,
                "shared_features": torch.cat(invariant),
                "depth": depths,
                "render_depth": depth,
                "alpha": alpha,
                "mvs_confidence": mvs_conf,
                "depth_probability": prob,
            },
        )
