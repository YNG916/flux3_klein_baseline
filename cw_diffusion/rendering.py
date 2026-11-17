"""Anisotropic Gaussian alpha compositing. CUDA uses pinned gsplat, CPU is a numerical oracle."""

from dataclasses import dataclass
import torch
from .geometry import quaternion_matrix, transform


@dataclass
class GaussianWorld:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacity: torch.Tensor
    features: torch.Tensor
    object_index: torch.Tensor


@dataclass
class Render:
    features: torch.Tensor  # [V,C,H,W], normalized conditional on coverage
    depth: torch.Tensor  # expected CV z, not ray length
    alpha: torch.Tensor


def render(world, K, c2w, height, width, backend="gsplat", near=0.1, far=15.0):
    if backend == "gsplat":
        if world.means.device.type != "cuda":
            raise ValueError(
                "Production gsplat requires CUDA; CPU tests explicitly request reference"
            )
        try:
            from gsplat import rasterization
        except ImportError as e:
            raise RuntimeError(
                "Install pinned requirements-cw.txt in the separate CW environment"
            ) from e
        out, alpha, _ = rasterization(
            means=world.means.float(),
            quats=world.quats.float(),
            scales=world.scales.float(),
            opacities=world.opacity.float(),
            colors=world.features.float(),
            viewmats=torch.linalg.inv(c2w).float(),
            Ks=K.float(),
            width=width,
            height=height,
            near_plane=near,
            far_plane=far,
            render_mode="RGB+ED",
            packed=True,
            rasterize_mode="classic",
            channel_chunk=32,
        )
        a = alpha.permute(0, 3, 1, 2)
        return Render(
            out[..., :-1].permute(0, 3, 1, 2) / a.clamp_min(1e-6),
            out[..., -1][:, None],
            a,
        )
    if backend != "reference":
        raise ValueError("Unknown renderer")
    if world.means.is_cuda:
        raise ValueError(
            "Reference renderer is a CPU test oracle, not a training fallback"
        )
    outputs = []
    depths = []
    alphas = []
    R = quaternion_matrix(world.quats)
    cov = R @ torch.diag_embed(world.scales.square()) @ R.transpose(-1, -2)
    y, x = torch.meshgrid(
        torch.arange(height, device=K.device, dtype=K.dtype) + 0.5,
        torch.arange(width, device=K.device, dtype=K.dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((x, y), -1)
    for k, t in zip(K, c2w):
        view = torch.linalg.inv(t)
        means = transform(world.means, view)
        z = means[:, 2]
        uv = (means @ k.T)[:, :2] / z[:, None].clamp_min(1e-6)
        fx, fy = k[0, 0], k[1, 1]
        zz = z.clamp_min(1e-6)
        # Match gsplat's perspective Jacobian FOV clamp; off-axis means must not explode their covariance.
        xx = (means[:, 0] / zz).clamp(
            -1.3 * width / (2 * fx), 1.3 * width / (2 * fx)
        ) * zz
        yy = (means[:, 1] / zz).clamp(
            -1.3 * height / (2 * fy), 1.3 * height / (2 * fy)
        ) * zz
        J = torch.stack(
            (
                fx / zz,
                torch.zeros_like(z),
                -fx * xx / zz.square(),
                torch.zeros_like(z),
                fy / zz,
                -fy * yy / zz.square(),
            ),
            -1,
        ).reshape(-1, 2, 3)
        camera_cov = view[:3, :3][None] @ cov @ view[:3, :3].T[None]
        projected = J @ camera_cov @ J.transpose(-1, -2) + 0.3 * torch.eye(
            2, device=K.device, dtype=K.dtype
        )
        delta = pixels[None] - uv[:, None, None]
        exponent = torch.einsum(
            "nhwi,nij,nhwj->nhw", delta, torch.linalg.inv(projected), delta
        )
        a = world.opacity[:, None, None] * torch.exp(-0.5 * exponent)
        a = a * ((z > near) & (z < far))[:, None, None]
        a = a.clamp(0, 0.999)
        # Match gsplat 1.5.3's rasterizer: discard sub-byte contributions.
        # Without this, normalizing tiny coverage magnifies harmless tail differences.
        a = torch.where(a >= 1 / 255, a, torch.zeros_like(a))
        order = z.argsort()
        a = a[order]
        trans = torch.cumprod(torch.cat((torch.ones_like(a[:1]), 1 - a + 1e-10), 0), 0)[
            :-1
        ]
        # CUDA terminates before accepting a contribution that crosses this threshold.
        weight = a * trans * (trans * (1 - a) > 1e-4)
        alpha = weight.sum(0)
        den = alpha.clamp_min(1e-6)
        outputs.append(torch.einsum("nhw,nc->chw", weight, world.features[order]) / den)
        depths.append((weight * z[order, None, None]).sum(0) / den)
        alphas.append(alpha)
    return Render(
        torch.stack(outputs), torch.stack(depths)[:, None], torch.stack(alphas)[:, None]
    )
