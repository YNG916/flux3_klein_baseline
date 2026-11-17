"""Noise-calibrated plane-sweep matching; no GT depth or visibility in this module."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import pixel_rays, transform, project, sample_map


class NoiseAwareMVS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.calibrate = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(
        self, features, K, c2w, prior_depth, prior_alpha, sigma, prior_confidence=None
    ):
        v, c, h, w = features.shape
        D = self.cfg.mvs_depth_bins
        feat = F.normalize(features.float(), dim=1)
        outputs = []
        confidences = []
        probabilities = []
        offsets = torch.linspace(-1, 1, D, device=features.device)
        for i in range(v):
            depth = prior_depth[i, 0]
            alpha = prior_alpha[i, 0]
            certainty = alpha if prior_confidence is None else prior_confidence[i, 0]
            local = depth.clamp_min(self.cfg.near)[None] * torch.exp(
                offsets[:, None, None] * (0.15 + 1 - certainty[None])
            )
            global_depth = torch.exp(
                torch.linspace(
                    math.log(self.cfg.near),
                    math.log(self.cfg.far),
                    D,
                    device=features.device,
                )
            )[:, None, None].expand(D, h, w)
            candidates = torch.where(alpha[None] > 0.05, local, global_depth).clamp(
                self.cfg.near, self.cfg.far
            )
            rays = pixel_rays(K[i], h, w)
            points = transform(
                (rays[None] * candidates[..., None]).reshape(-1, 3), c2w[i]
            )
            score = torch.zeros_like(candidates)
            support = torch.zeros_like(candidates)
            for j in range(v):
                if j == i:
                    continue
                baseline = (c2w[i, :3, 3] - c2w[j, :3, 3]).norm()
                if baseline.detach().item() < 1e-4:
                    continue
                grid, z, valid = project(
                    points, K[j], c2w[j], h, w, self.cfg.near, self.cfg.far
                )
                other = sample_map(feat[j : j + 1], grid[None])[0].reshape(D, h, w, c)
                other = F.normalize(other, dim=-1)
                comparison = (other * feat[i].permute(1, 2, 0)[None]).sum(-1)
                pd = sample_map(prior_depth[j : j + 1], grid[None])[0, :, 0].reshape(
                    D, h, w
                )
                pa = sample_map(prior_alpha[j : j + 1], grid[None])[0, :, 0].reshape(
                    D, h, w
                )
                # Predicted geometry vetoes foreground occluders; unknown geometry is not treated as GT.
                visible = (pa < 0.05) | (
                    z.reshape(D, h, w) <= pd + 0.15 + 0.15 * pd.abs()
                )
                valid = valid.reshape(D, h, w) & visible
                score = score + torch.where(
                    valid, comparison, torch.zeros_like(comparison)
                )
                support = support + valid
            average = score / support.clamp_min(1)
            eligible = support > 0
            # Gaussian depth prior, normalized before combining with likelihood.
            logprior = -0.5 * (offsets[:, None, None] / 0.6).square().expand_as(average)
            logprior = torch.where(
                alpha[None] > 0.05, logprior, torch.zeros_like(logprior)
            )
            probability = (
                (logprior + 8 * average).masked_fill(~eligible, -1e4).softmax(0)
            )
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(
                0
            ) / math.log(max(D, 2))
            contrast = (
                (
                    average.masked_fill(~eligible, 0).square().sum(0)
                    / eligible.sum(0).clamp_min(1)
                    - (
                        average.masked_fill(~eligible, 0).sum(0)
                        / eligible.sum(0).clamp_min(1)
                    ).square()
                )
                .clamp_min(0)
                .add(1e-8)
                .sqrt()
                .sub(1e-4)
                .clamp_min(0)
            )
            matching = (1 - entropy).clamp(0, 1) * contrast / (contrast + 0.05)
            any_valid = eligible.any(0)
            strength = eligible.float().mean(0)
            evidence = torch.stack(
                (matching, strength, certainty, torch.ones_like(alpha) * (1 - sigma)),
                dim=-1,
            )
            weight = (
                self.calibrate(evidence).sigmoid()[..., 0]
                * (1 - sigma).square()
                * matching
                * any_valid
            )
            # Equal/unsupported matches never create geometric evidence.
            estimate = (probability * candidates).sum(0)
            fallback = torch.where(alpha > 0.05, depth, torch.zeros_like(depth))
            outputs.append(torch.where(any_valid, estimate, fallback))
            confidences.append(weight)
            probabilities.append(probability)
        return (
            torch.stack(outputs)[:, None],
            torch.stack(confidences)[:, None],
            torch.stack(probabilities),
        )
