import torch
from cw_diffusion.config import WorldConfig
from cw_diffusion.mvs import NoiseAwareMVS
from cw_diffusion.geometry import pixel_rays, transform
from cw_diffusion.feedback import CanonicalFeedback
from cw_diffusion.rendering import GaussianWorld
from cw_diffusion.world import WorldState


def test_plane_sweep_recovers_textured_plane_and_disables_at_full_noise():
    cfg = WorldConfig(
        feature_dim=8, hidden_dim=32, mvs_depth_bins=32, render_backend="reference"
    )
    m = NoiseAwareMVS(cfg)
    h = w = 12
    K = torch.tensor([[10.0, 0, 6.0], [0, 10.0, 6.0], [0, 0, 1.0]]).repeat(3, 1, 1)
    T = torch.eye(4).repeat(3, 1, 1)
    T[1, 0, 3] = 0.5
    T[2, 0, 3] = -0.5
    features = []
    for i in range(3):
        xyz = transform((pixel_rays(K[i], h, w) * 2).reshape(-1, 3), T[i]).reshape(
            h, w, 3
        )
        x, y = xyz[..., 0], xyz[..., 1]
        features.append(
            torch.stack(
                (
                    torch.sin(x * 9),
                    torch.cos(x * 9),
                    torch.sin(y * 9),
                    torch.cos(y * 9),
                    torch.sin(x * 4),
                    torch.cos(x * 4),
                    torch.sin(y * 4),
                    torch.cos(y * 4),
                )
            )
        )
    features = torch.stack(features)
    prior = torch.ones(3, 1, h, w) * 2.2
    alpha = torch.ones_like(prior) * 0.8
    depth, conf, _ = m(features, K, T, prior, alpha, torch.tensor(0.0))
    valid = conf[:, 0] > 1e-4
    assert valid.any()
    assert (depth[:, 0][valid] - 2).abs().mean() < 0.2
    _, conf, _ = m(features, K, T, prior, alpha, torch.tensor(1.0))
    assert not conf.any()


def test_leave_one_out_excludes_current_and_past_self_observations():
    torch.manual_seed(8)
    cfg = WorldConfig(
        feature_dim=8, hidden_dim=32, mvs_depth_bins=4, render_backend="reference"
    )
    m = CanonicalFeedback(cfg, 16)
    K = torch.tensor([[1.0, 0, 1.0], [0, 1.0, 1.0], [0, 0, 1.0]]).repeat(3, 1, 1)
    T = torch.eye(4).repeat(3, 1, 1)
    means = torch.tensor([[0.0, 0, 2.0]])

    def initial():
        f = torch.ones(1, 8)
        g = GaussianWorld(
            means,
            torch.tensor([[1.0, 0, 0, 0]]),
            torch.ones(1, 3) * 0.5,
            torch.ones(1) * 0.8,
            f,
            torch.tensor([0]),
        )
        return WorldState(g, f, means, torch.ones(1) * 0.5)

    x = torch.randn(3, 4, 16)
    changed = x.clone()
    changed[0] = torch.randn_like(changed[0]) * 4
    _, a, _ = m(x, initial(), K, T, (2, 2), torch.tensor(0.2))
    _, b, _ = m(changed, initial(), K, T, (2, 2), torch.tensor(0.2))
    torch.testing.assert_close(a.loo_features[0], b.loo_features[0], atol=0, rtol=0)
    assert not torch.equal(a.loo_features[1], b.loo_features[1])
    _, aa, _ = m(x, a, K, T, (2, 2), torch.tensor(0.1))
    _, bb, _ = m(changed, b, K, T, (2, 2), torch.tensor(0.1))
    torch.testing.assert_close(aa.loo_features[0], bb.loo_features[0], atol=0, rtol=0)
