"""Metric Z-up world, CV cameras, pixel centers at (u+.5,v+.5)."""

import torch
import torch.nn.functional as F


def transform(points, matrix):
    return points @ matrix[..., :3, :3].transpose(-1, -2) + matrix[..., None, :3, 3]


def project(points, K, camera_to_world, height, width, near=0.1, far=15.0):
    cam = transform(points, torch.linalg.inv(camera_to_world))
    z = cam[..., 2]
    pixel = cam @ K.transpose(-1, -2)
    uv = pixel[..., :2] / z[..., None].clamp_min(1e-8)
    valid = (
        (z > near)
        & (z < far)
        & (uv[..., 0] >= 0.5)
        & (uv[..., 0] <= width - 0.5)
        & (uv[..., 1] >= 0.5)
        & (uv[..., 1] <= height - 0.5)
    )
    grid = torch.stack((2 * uv[..., 0] / width - 1, 2 * uv[..., 1] / height - 1), -1)
    return grid, z, valid


def pixel_rays(K, height, width):
    y, x = torch.meshgrid(
        torch.arange(height, device=K.device, dtype=K.dtype) + 0.5,
        torch.arange(width, device=K.device, dtype=K.dtype) + 0.5,
        indexing="ij",
    )
    p = torch.stack((x, y, torch.ones_like(x)), -1)
    return p @ torch.linalg.inv(K).transpose(-1, -2)


def unproject(depth, K, camera_to_world):
    p = pixel_rays(K, *depth.shape[-2:]) * depth[..., None]
    return transform(p.reshape(-1, 3), camera_to_world).reshape(*depth.shape, 3)


def resize_intrinsics(K, source_hw, target_hw):
    scale = K.new_tensor(
        [target_hw[1] / source_hw[1], target_hw[0] / source_hw[0], 1.0]
    )
    return K * scale[..., None]


def sample_map(image, grid):
    """image [V,C,H,W], grid [V,N,2] -> [V,N,C]."""
    return (
        F.grid_sample(image, grid[:, None], align_corners=False, padding_mode="zeros")
        .squeeze(2)
        .transpose(1, 2)
    )


def safe_correspondence(
    depth_src, depth_dst, K_src, K_dst, T_src, T_dst, atol=0.05, rtol=0.01
):
    """Full bilinear footprint validity; do not let invalid depth contaminate samples."""
    h, w = depth_src.shape
    hd, wd = depth_dst.shape
    valid_s = torch.isfinite(depth_src) & (depth_src > 0)
    pts = unproject(
        torch.where(valid_s, depth_src, torch.ones_like(depth_src)), K_src, T_src
    )
    grid, z, inside = project(pts.reshape(-1, 3), K_dst, T_dst, hd, wd, near=0, far=1e6)
    good = torch.isfinite(depth_dst) & (depth_dst > 0)
    dst = torch.where(good, depth_dst, torch.zeros_like(depth_dst))
    sampled = sample_map(dst[None, None], grid[None])[0, :, 0]
    support = sample_map(good.to(dst)[None, None], grid[None])[0, :, 0]
    valid = (
        valid_s.flatten()
        & inside
        & (support > 1 - 1e-6)
        & ((sampled - z).abs() <= atol + rtol * z.abs())
    )
    return grid.reshape(h, w, 2), valid.reshape(h, w)


def yaw_matrix(yaw):
    c, s = yaw.cos(), yaw.sin()
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(
        *yaw.shape, 3, 3
    )


def quaternion_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def rotation_6d(value):
    """Continuous SO(3) representation: first two columns followed by Gram-Schmidt."""
    a, b = value[..., :3], value[..., 3:]
    x = F.normalize(a, dim=-1)
    y = F.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    z = torch.linalg.cross(x, y, dim=-1)
    return torch.stack((x, y, z), -1)


def matrix_quaternion(matrix):
    """Stable wxyz conversion; choose the best-conditioned quaternion branch."""
    m = matrix
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    magnitude = (
        torch.stack(
            (
                1 + m00 + m11 + m22,
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
            ),
            -1,
        )
        .clamp_min(1e-8)
        .sqrt()
    )
    rows = torch.stack(
        (
            torch.stack(
                (
                    magnitude[..., 0].square(),
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 2, 1] - m[..., 1, 2],
                    magnitude[..., 1].square(),
                    m[..., 1, 0] + m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] + m[..., 0, 1],
                    magnitude[..., 2].square(),
                    m[..., 2, 1] + m[..., 1, 2],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 2, 0] + m[..., 0, 2],
                    m[..., 2, 1] + m[..., 1, 2],
                    magnitude[..., 3].square(),
                ),
                -1,
            ),
        ),
        -2,
    )
    candidates = rows / (2 * magnitude[..., None].clamp_min(0.1))
    index = magnitude.argmax(-1)
    selected = candidates.gather(
        -2, index[..., None, None].expand(*index.shape, 1, 4)
    ).squeeze(-2)
    return F.normalize(selected, dim=-1)


def quaternion_product(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )
