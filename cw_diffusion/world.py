"""RGB-only BEV object discovery and object-local anisotropic feature fields."""

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .config import WorldConfig
from .geometry import rotation_6d, matrix_quaternion, quaternion_product
from .rendering import GaussianWorld


@dataclass
class WorldState:
    geometry: GaussianWorld
    base_features: torch.Tensor
    prior_means: torch.Tensor
    confidence: torch.Tensor
    age: int = 0
    loo_features: torch.Tensor | None = None

    def detached(self):
        g = self.geometry
        return WorldState(
            GaussianWorld(*(getattr(g, k).detach() for k in g.__dataclass_fields__)),
            self.base_features.detach(),
            self.prior_means.detach(),
            self.confidence.detach(),
            self.age,
            None if self.loo_features is None else self.loo_features.detach(),
        )


@dataclass
class WorldPrediction:
    state: WorldState
    classes: torch.Tensor
    centers: torch.Tensor
    sizes: torch.Tensor
    yaw: torch.Tensor
    rotations: torch.Tensor
    center_log_variance: torch.Tensor
    masks: torch.Tensor
    bev_semantic: torch.Tensor
    bev_height: torch.Tensor
    bev_occupancy: torch.Tensor
    object_features: torch.Tensor


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class WorldEncoder(nn.Module):
    def __init__(self, cfg: WorldConfig):
        super().__init__()
        self.cfg = cfg.validate()
        h = cfg.hidden_dim
        d = cfg.feature_dim
        q = cfg.object_queries
        g = cfg.gaussians_per_object
        self.encoder = nn.Sequential(
            nn.Conv2d(3, h // 4, 7, stride=2, padding=3),
            Residual(h // 4),
            nn.Conv2d(h // 4, h // 2, 3, stride=2, padding=1),
            Residual(h // 2),
            nn.Conv2d(h // 2, h, 3, stride=2, padding=1),
            Residual(h),
            Residual(h),
        )
        self.position = nn.Sequential(nn.Linear(2, h), nn.SiLU(), nn.Linear(h, h))
        self.queries = nn.Parameter(torch.randn(q, h) * 0.02)
        layer = nn.TransformerDecoderLayer(
            h, 8, 4 * h, dropout=0, batch_first=True, norm_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(
            layer, cfg.decoder_layers, norm=nn.LayerNorm(h)
        )
        self.classes = nn.Linear(h, cfg.semantic_classes + 1)
        with torch.no_grad():
            self.classes.bias[-1] = math.log(cfg.semantic_classes * 9.0)
        self.box = nn.Linear(h, 12)
        with torch.no_grad():
            self.box.bias[3:6].fill_(-3.0)
            height_fraction = (1.0 - cfg.min_height) / (cfg.max_height - cfg.min_height)
            self.box.bias[2] = math.log(height_fraction / (1 - height_fraction))
        sites = torch.quasirandom.SobolEngine(2).draw(q).clamp(0.01, 0.99)
        self.register_buffer("query_reference_logits", torch.logit(sites))
        self.center_uncertainty = nn.Linear(h, 3)
        self.mask_embed = nn.Linear(h, h)
        self.app = nn.Linear(h, d)
        self.field = nn.Linear(h, g * (11 + d))
        self.bev_semantic = nn.Conv2d(h, cfg.semantic_classes, 1)
        self.height = nn.Conv2d(h, 1, 1)
        self.occupancy = nn.Conv2d(h, 1, 1)
        self.background = nn.Conv2d(h, d + 2, 1)
        self.random_appearance = nn.Linear(d, d, bias=False)
        self.register_buffer(
            "canonical_sites", torch.quasirandom.SobolEngine(3).draw(g) * 2 - 1
        )
        # Final 3 anchors are known robot states, not scene GT objects.
        self.robot_identity = nn.Parameter(torch.randn(3, h) * 0.02)
        self.robot = nn.Sequential(
            nn.Linear(3 + 9 + 3, h), nn.SiLU(), nn.Linear(h, g * (11 + d))
        )

    def forward(self, bev, world_to_bev, robot_states, *, generator=None):
        """One synchronous frame. No semantic/instance/depth/state_before arguments are accepted."""
        if bev.ndim != 3 or bev.shape[0] != 3:
            raise ValueError("RGB-only BEV tensor required")
        h = self.cfg.hidden_dim
        d = self.cfg.feature_dim
        g = self.cfg.gaussians_per_object
        f = self.encoder(bev[None])[0]
        hh, ww = f.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, hh, device=f.device),
            torch.linspace(-1, 1, ww, device=f.device),
            indexing="ij",
        )
        memory = f.flatten(1).T + self.position(
            torch.stack((xx, yy), -1).reshape(-1, 2)
        )
        tokens = self.decoder(self.queries[None], memory[None])[0]
        classes = self.classes(tokens)
        b = self.box(tokens)
        log_variance = self.center_uncertainty(tokens).clamp(-5, 3)
        prob = classes.softmax(-1)
        p2w = torch.linalg.inv(world_to_bev.float())
        side = bev.shape[-1]
        uv = (b[:, :2] + self.query_reference_logits).sigmoid() * (side - 1)
        homogeneous = torch.cat(
            (uv, uv.new_zeros(len(uv), 1), uv.new_ones(len(uv), 1)), -1
        )
        centers = (homogeneous @ p2w.T)[:, :3]
        centers = centers + torch.stack(
            (
                b[:, 2] * 0,
                b[:, 2] * 0,
                b[:, 2].sigmoid() * (self.cfg.max_height - self.cfg.min_height)
                + self.cfg.min_height,
            ),
            -1,
        )
        span = (
            torch.stack(
                (
                    p2w[0, 0].abs(),
                    p2w[1, 1].abs(),
                    p2w.new_tensor((self.cfg.max_height - self.cfg.min_height) / side),
                )
            )
            * side
        )
        sizes = b[:, 3:6].sigmoid() * span[None] + 0.03
        rotation = rotation_6d(b[:, 6:12] + b.new_tensor([1.0, 0, 0, 0, 1.0, 0]))
        angle = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        yaw = torch.stack((angle.sin(), angle.cos()), -1)
        raw = self.field(tokens).reshape(-1, g, 11 + d)
        local = (
            (self.canonical_sites[None] + 0.2 * raw[..., :3].tanh()).clamp(-1, 1)
            * sizes[:, None]
            / 2
        )
        means = torch.einsum("qij,qgj->qgi", rotation, local) + centers[:, None]
        scales = (raw[..., 3:6].sigmoid() * 0.2 + 0.01) * sizes[:, None]
        # Compose full 3D object rotation with each canonical-local covariance rotation.
        quat = F.normalize(raw[..., 6:10] + raw.new_tensor([1.0, 0, 0, 0]), dim=-1)
        quats = quaternion_product(matrix_quaternion(rotation)[:, None], quat)
        opacity = raw[..., 10].sigmoid() * (1 - prob[:, -1, None])
        shared_noise = torch.randn(
            (len(tokens), d), device=f.device, generator=generator
        )
        app = self.app(tokens) + self.random_appearance(shared_noise) * 0.1
        features = raw[..., 11:] + app[:, None]
        indices = torch.arange(len(tokens), device=f.device).repeat_interleave(g)
        confidence = (
            (1 / (1 + log_variance.exp().mean(-1)))[:, None].expand(-1, g).reshape(-1)
        )
        means, scales, quats, opacity, features = [
            v.reshape(-1, v.shape[-1]) if v.ndim == 3 else v.flatten()
            for v in [means, scales, quats, opacity, features]
        ]
        # Spatial background field covers structure independent of detected instance slots.
        n = self.cfg.background_side
        l = self.cfg.background_levels
        bg = F.adaptive_avg_pool2d(f, (n, n))
        bg = self.background(bg).permute(1, 2, 0).reshape(-1, d + 2)
        yy, xx, zz = torch.meshgrid(
            torch.arange(n, device=f.device),
            torch.arange(n, device=f.device),
            torch.arange(l, device=f.device),
            indexing="ij",
        )
        pix = torch.stack(
            (
                (xx + 0.5) * side / n,
                (yy + 0.5) * side / n,
                torch.zeros_like(zz),
                torch.ones_like(zz),
            ),
            -1,
        ).float()
        xyz = (pix @ p2w.T)[..., :3]
        xyz = xyz + torch.stack(
            (
                zz * 0,
                zz * 0,
                (zz + 0.5) * (self.cfg.max_height - self.cfg.min_height) / l
                + self.cfg.min_height,
            ),
            -1,
        )
        nb = n * n * l
        bg_features = bg[:, :d, None].transpose(1, 2).expand(-1, l, -1).reshape(nb, d)
        bg_opacity = bg[:, d].sigmoid()[:, None].expand(-1, l).reshape(-1) * 0.5
        bg_scale = (
            span.new_tensor([1.0, 1.0, 1.0])[None].expand(nb, 3)
            * span[None]
            / span.new_tensor([n, n, l])
            * 0.35
        )
        bg_quat = f.new_tensor([1.0, 0, 0, 0]).expand(nb, 4)
        means = torch.cat((means, xyz.reshape(-1, 3)))
        scales = torch.cat((scales, bg_scale))
        quats = torch.cat((quats, bg_quat))
        opacity = torch.cat((opacity, bg_opacity))
        confidence = torch.cat(
            (confidence, bg[:, d + 1].sigmoid()[:, None].expand(-1, l).reshape(-1))
        )
        features = torch.cat((features, bg_features))
        indices = torch.cat(
            (indices, torch.full((nb,), -1, device=f.device, dtype=torch.long))
        )
        if robot_states is not None:
            # states [3,15]: local position, flattened rotation, dimensions; task-authorized input.
            encoded = self.robot[1](self.robot[0](robot_states)) + self.robot_identity
            rr = self.robot[2](encoded).reshape(3, g, 11 + d)
            rc = robot_states[:, :3]
            rR = robot_states[:, 3:12].reshape(3, 3, 3)
            rs = robot_states[:, 12:]
            local = (
                (self.canonical_sites[None] + 0.2 * rr[..., :3].tanh())
                * rs[:, None]
                / 2
            )
            rm = torch.einsum("qij,qgj->qgi", rR, local) + rc[:, None]
            rq = F.normalize(rr[..., 6:10] + rr.new_tensor([1.0, 0, 0, 0]), dim=-1)
            # Axis-aligned covariance is sufficient for rigid known robot anchor support.
            means = torch.cat((means, rm.reshape(-1, 3)))
            scales = torch.cat(
                (
                    scales,
                    ((rr[..., 3:6].sigmoid() * 0.15 + 0.02) * rs[:, None]).reshape(
                        -1, 3
                    ),
                )
            )
            rq = quaternion_product(matrix_quaternion(rR)[:, None], rq)
            quats = torch.cat((quats, rq.reshape(-1, 4)))
            opacity = torch.cat((opacity, rr[..., 10].sigmoid().flatten()))
            confidence = torch.cat((confidence, confidence.new_full((3 * g,), 0.75)))
            features = torch.cat((features, rr[..., 11:].reshape(-1, d)))
            indices = torch.cat(
                (
                    indices,
                    torch.arange(3, device=f.device).repeat_interleave(g) + len(tokens),
                )
            )
        world = GaussianWorld(
            means, quats, scales.clamp_min(0.001), opacity, features, indices
        )
        state = WorldState(world, features, means, confidence)
        masks = torch.einsum("qc,chw->qhw", self.mask_embed(tokens), f) / math.sqrt(h)
        return WorldPrediction(
            state,
            classes,
            centers,
            sizes,
            yaw,
            rotation,
            log_variance,
            masks,
            self.bev_semantic(f),
            self.height(f)[0].sigmoid() * (self.cfg.max_height - self.cfg.min_height)
            + self.cfg.min_height,
            self.occupancy(f)[0],
            app,
        )
