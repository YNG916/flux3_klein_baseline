import torch
import pytest
from cw_diffusion.geometry import (
    unproject,
    project,
    resize_intrinsics,
    safe_correspondence,
)
from cw_diffusion.rendering import GaussianWorld, render


def test_pixel_center_roundtrip_and_resize():
    K = torch.tensor([[4.0, 0, 2.0], [0, 4.0, 1.5], [0, 0, 1.0]])
    depth = torch.ones(3, 4) * 2
    T = torch.eye(4)
    T[:3, 3] = torch.tensor([2.0, -1.0, 0.3])
    xyz = unproject(depth, K, T)
    grid, z, valid = project(xyz.reshape(-1, 3), K, T, 3, 4, near=0.1)
    y, x = torch.meshgrid(torch.arange(3) + 0.5, torch.arange(4) + 0.5, indexing="ij")
    expected = torch.stack((2 * x / 4 - 1, 2 * y / 3 - 1), -1).reshape(-1, 2)
    torch.testing.assert_close(grid, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(z, depth.flatten())
    assert valid.all()
    resized = resize_intrinsics(K, (3, 4), (6, 8))
    torch.testing.assert_close(resized[0], 2 * K[0])


def test_invalid_bilinear_depth_is_never_visible():
    K = torch.eye(3)
    T = torch.eye(4)
    depth = torch.ones(3, 3) * 2
    damaged = depth.clone()
    damaged[1, 1] = float("inf")
    _, valid = safe_correspondence(depth, damaged, K, K, T, T)
    assert not valid[1, 1] and not torch.isnan(valid.float()).any()


def test_gaussian_occlusion_depth_and_all_parameter_gradients():
    means = torch.tensor([[0.0, 0, 2.0], [0, 0, 4.0]], requires_grad=True)
    scales = torch.full((2, 3), 0.5, requires_grad=True)
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0.1, 0, 0]], requires_grad=True)
    opacity = torch.tensor([0.99, 0.8], requires_grad=True)
    feat = torch.tensor([[1.0, 0.0], [0, 1.0]], requires_grad=True)
    world = GaussianWorld(means, quats, scales, opacity, feat, torch.tensor([0, 1]))
    K = torch.tensor([[[4.0, 0, 2.0], [0, 4.0, 2.0], [0, 0, 1.0]]])
    T = torch.eye(4)[None]
    r = render(world, K, T, 4, 4, "reference")
    assert (
        r.features[0, 0, 1, 1] > r.features[0, 1, 1, 1] and 2 < r.depth[0, 0, 1, 1] < 4
    )
    (r.features.square().sum() + r.depth.sum() + r.alpha.sum()).backward()
    for p in [means, quats, scales, opacity, feat]:
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_production_renderer_never_silently_falls_back_to_cpu():
    world = GaussianWorld(
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0, 0, 0]]),
        torch.ones(1, 3),
        torch.ones(1),
        torch.ones(1, 2),
        torch.tensor([0]),
    )
    with pytest.raises(ValueError, match="CUDA"):
        render(world, torch.eye(3)[None], torch.eye(4)[None], 2, 2, "gsplat")


def test_full_so3_and_quaternion_roundtrip_with_gradients():
    from cw_diffusion.geometry import (
        rotation_6d,
        matrix_quaternion,
        quaternion_matrix,
        quaternion_product,
    )

    x = torch.randn(8, 6, requires_grad=True)
    R = rotation_6d(x)
    torch.testing.assert_close(
        R.transpose(-1, -2) @ R, torch.eye(3).expand(8, 3, 3), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(8), atol=1e-5, rtol=1e-5)
    q = matrix_quaternion(R)
    torch.testing.assert_close(quaternion_matrix(q), R, atol=1e-5, rtol=1e-5)
    twice = quaternion_product(q, q)
    torch.testing.assert_close(quaternion_matrix(twice), R @ R, atol=1e-5, rtol=1e-5)
    q.sum().backward()
    assert torch.isfinite(x.grad).all()
