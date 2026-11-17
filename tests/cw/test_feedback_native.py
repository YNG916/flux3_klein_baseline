from unittest.mock import patch
import torch
from cw_diffusion.config import WorldConfig
from cw_diffusion.world import WorldState, WorldEncoder
from cw_diffusion.rendering import GaussianWorld
from cw_diffusion.feedback import CanonicalFeedback
from cw_diffusion.native import CanonicalDiT, SceneContext
from cw_diffusion.mvs import NoiseAwareMVS


def cfg():
    return WorldConfig(
        feature_dim=8,
        hidden_dim=32,
        object_queries=4,
        gaussians_per_object=2,
        background_side=2,
        background_levels=2,
        decoder_layers=1,
        injection_blocks=(0, 2),
        mvs_depth_bins=4,
        render_backend="reference",
        checkpoint_blocks=True,
    )


def scene():
    K = torch.tensor([[1.0, 0, 1.0], [0, 1.0, 1.0], [0, 0, 1.0]]).repeat(3, 1, 1)
    T = torch.eye(4).repeat(3, 1, 1)
    return K, T


def state():
    f = torch.randn(2, 8)
    means = torch.tensor([[0.0, 0, 2.0], [0.2, 0.1, 3.0]])
    world = GaussianWorld(
        means,
        torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1),
        torch.ones(2, 3) * 0.5,
        torch.ones(2) * 0.9,
        f,
        torch.tensor([0, 1]),
    )
    return WorldState(world, f, means, torch.ones(2) * 0.5)


def test_cross_view_gradient_and_zero_initialization():
    torch.manual_seed(3)
    m = CanonicalFeedback(cfg(), 16)
    x = torch.randn(3, 4, 16, requires_grad=True)
    K, T = scene()
    y, s, aux = m(x, state(), K, T, (2, 2), torch.tensor(0.2))
    torch.testing.assert_close(y, x, atol=0, rtol=0)
    y.square().mean().backward()
    assert m.out.weight.grad.abs().sum() > 0
    m.zero_grad()
    x.grad = None
    with torch.no_grad():
        m.out.weight.normal_(std=0.05)
    y, s, aux = m(x, state(), K, T, (2, 2), torch.tensor(0.2))
    y[0].square().mean().backward()
    assert x.grad[1:].abs().sum() > 0
    assert torch.isfinite(x.grad).all()
    assert s.loo_features.shape == (3, 2, 8) and s.detached().loo_features is not None


def test_no_coverage_feedback_exact_identity_and_memory_finite():
    m = CanonicalFeedback(cfg(), 16)
    with torch.no_grad():
        m.out.weight.normal_(std=0.1)
    s = state()
    s.geometry.means = s.geometry.means + torch.tensor([1000.0, 0, 0])
    s.prior_means = s.geometry.means
    x = torch.randn(3, 4, 16)
    K, T = scene()
    y, s, aux = m(x, s, K, T, (2, 2), torch.tensor(0.9))
    torch.testing.assert_close(y, x, atol=0, rtol=0)
    assert torch.isfinite(s.geometry.features).all()
    assert not aux["mvs_confidence"].any()


def test_mvs_zero_evidence_for_identical_or_constant_views():
    m = NoiseAwareMVS(cfg())
    K, T = scene()
    features = torch.ones(3, 8, 2, 2, requires_grad=True)
    depth = torch.ones(3, 1, 2, 2) * 2
    alpha = torch.ones_like(depth) * 0.8
    _, confidence, _ = m(features, K, T, depth, alpha, torch.tensor(0.2))
    assert not confidence.any()
    T[1, 0, 3] = 0.1
    T[2, 0, 3] = -0.1
    result, confidence, _ = m(features, K, T, depth, alpha, torch.tensor(0.2))
    assert confidence.max() < 1e-3
    result.sum().backward()
    assert torch.isfinite(features.grad).all()


def test_world_encoder_rgb_only_and_object_local_shapes():
    m = WorldEncoder(cfg())
    bev = torch.rand(3, 32, 32)
    w2b = torch.tensor(
        [[4.0, 0, 0, 16.0], [0, -4.0, 0, 16.0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    )
    robots = torch.cat(
        (
            torch.zeros(3, 3),
            torch.eye(3).flatten()[None].repeat(3, 1),
            torch.ones(3, 3),
        ),
        1,
    )
    pred = m(bev, w2b, robots, generator=torch.Generator().manual_seed(2))
    assert pred.centers.shape == (4, 3) and pred.state.geometry.features.shape == (
        22,
        8,
    )
    assert torch.all(pred.state.geometry.scales > 0)
    (
        pred.bev_height.mean()
        + pred.state.geometry.features.square().mean()
        + pred.centers.mean()
    ).backward()
    assert m.field.weight.grad is not None


def test_real_native_zero_adapter_equivalence_and_checkpoint_gradient():
    import flux3.dit as d
    from flux3.pipeline import pack
    from flux_mvwd.backend import configure_attention

    values = {
        "HIDDEN": 32,
        "HEADS": 1,
        "DEPTH": 1,
        "DEPTH_SINGLE": 3,
        "AXES_DIM": (8, 8, 8, 8),
        "MLP_RATIO": 2.0,
    }
    with (
        patch.multiple(d, **values),
        patch.dict(d.IN_CHANNELS, image=8, image_cond=8, text=16),
    ):
        configure_attention("math")
        torch.manual_seed(9)
        backbone = d.DiT(False).to(torch.bfloat16)
        entries = []
        pair = lambda n, c: (torch.randn(n, c).to(torch.bfloat16), torch.zeros(n, 4))
        for i in range(3):
            entries.append([pair(1 + i, 16), pair(2, 8), pair(4, 8)])
        packed = pack(entries)
        packed.timesteps["image"].fill_(0.4)
        baseline = backbone(packed)
        model = CanonicalDiT(backbone, cfg())
        K, T = scene()
        context = SceneContext([state()], K[None], T[None], (32, 32), torch.tensor(0.4))
        result, states, aux = model(packed, context)
        torch.testing.assert_close(result, baseline, atol=0, rtol=0)
        result.square().mean().backward()
        assert model.feedback["0"].out.weight.grad.abs().sum() > 0
        assert model.feedback["2"].out.weight.grad.abs().sum() > 0
        assert len(states) == 1 and len(aux) == 2
