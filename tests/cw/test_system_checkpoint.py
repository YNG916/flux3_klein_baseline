from unittest.mock import patch
from types import SimpleNamespace
import torch
import pytest
from cw_diffusion.system import CanonicalSystem
from cw_diffusion.checkpoint import save_checkpoint, load_checkpoint, load_model
from cw_diffusion.config import WorldConfig


def test_complete_method_loss_unroll_and_finite_gradients():
    import flux3.dit as d
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
        torch.manual_seed(7)
        cfg = WorldConfig(
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
            unroll_steps=2,
        )
        model = CanonicalSystem(d.DiT(False).to(torch.bfloat16), cfg)
        T = torch.eye(4).repeat(3, 1, 1)
        T[1, 0, 3] = 0.05
        T[2, 0, 3] = -0.05
        K = torch.tensor([[16.0, 0, 16.0], [0, 16.0, 16.0], [0, 0, 1.0]]).repeat(
            3, 1, 1
        )
        condition = {
            "bev": torch.rand(3, 32, 32),
            "world_to_bev": torch.tensor(
                [[4.0, 0, 0, 16.0], [0, -4.0, 0, 16.0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
            ),
            "robot_states": torch.cat(
                (
                    torch.zeros(3, 3),
                    torch.eye(3).flatten()[None].repeat(3, 1),
                    torch.ones(3, 3),
                ),
                1,
            ),
            "K": K,
            "c2w": T,
        }
        pair = lambda n, c: (torch.randn(n, c).to(torch.bfloat16), torch.zeros(n, 4))
        samples = [
            {
                "target": pair(4, 8),
                "text": pair(2 + i, 16),
                "unconditional": pair(1, 16),
                "references": [pair(2, 8), pair(2, 8)],
            }
            for i in range(3)
        ]
        target = {
            "rgb": torch.rand(3, 3, 32, 32),
            "depth_linear": torch.ones(3, 8, 8) * 2,
            "geometry_K": K / torch.tensor([4.0, 4.0, 1.0])[:, None],
            "modality_c2w": {k: T for k in ["depth_linear", "semantic", "instance"]},
            "rgb_K": K,
            "rgb_c2w": T,
            "semantic": torch.ones(3, 8, 8, dtype=torch.long) * 4,
            "instance": torch.ones(3, 8, 8, dtype=torch.long) * 7,
            "bev_instance": torch.ones(32, 32, dtype=torch.long) * 7,
            "bev_semantic": torch.ones(32, 32, dtype=torch.long) * 4,
            "bev_height": torch.ones(32, 32) * 2,
            "bev_height_valid": torch.ones(32, 32, dtype=torch.bool),
            "bev_occupancy": torch.ones(32, 32),
            "objects": [
                {
                    "category": 4,
                    "center": torch.tensor([0.0, 0, 2.0]),
                    "aabb_size": torch.ones(3),
                    "yaw": 0.0,
                    "public_instance_id": 7,
                }
            ],
            "teacher": torch.randn(3, 4, 384),
        }
        target["depth_linear"][0, 0, 0] = float("inf")
        loss, parts = model(
            condition, samples, target, None, torch.Generator().manual_seed(3)
        )
        assert torch.isfinite(loss) and all(torch.isfinite(x) for x in parts.values())
        loss.backward()
        bad = [
            name
            for name, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        assert not bad, bad
        assert model.world.field.weight.grad.abs().sum() > 0
        assert model.dit.feedback["0"].out.weight.grad.abs().sum() > 0
        assert {
            "surface",
            "instance_discrimination",
            "appearance_distill",
            "depth",
            "flow",
        } <= parts.keys()


def test_full_state_checkpoint_rng_optimizer_and_corruption(tmp_path):
    torch.manual_seed(3)
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    opt.step()
    g = torch.Generator().manual_seed(7)
    acc = SimpleNamespace(
        is_main_process=True,
        process_index=0,
        device=torch.device("cpu"),
        wait_for_everyone=lambda: None,
        unwrap_model=lambda x: x,
    )
    contract = {
        "world_size": 1,
        "groups": 2,
        "padded_groups": 2,
        "accumulation": 1,
        "target_updates": 10,
    }
    progress = {"step": 1, "epoch": 0, "cursor": 1, "global_groups_seen": 1}
    path = tmp_path / "ck"
    save_checkpoint(path, model, opt, progress, contract, acc, g)
    next_random = torch.randn(3, generator=g)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.zero_()
    result = load_checkpoint(path, model, opt, contract, 0, "cpu", g)
    assert result == progress
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, expected[k])
    torch.testing.assert_close(torch.randn(3, generator=g), next_random)
    load_model(path, model, {"world_size": 1})
    with pytest.raises(ValueError, match="different experiment"):
        load_model(path, model, {"config_sha256": "unrelated-experiment"})
    with pytest.raises(ValueError):
        load_checkpoint(path, model, opt, {**contract, "world_size": 2}, 0, "cpu", g)
    with (path / "model.pt").open("ab") as f:
        f.write(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path, model, opt, contract, 0, "cpu", g)
    with pytest.raises(ValueError, match="checksum"):
        load_model(path, model)


def test_optimizer_separates_pretrained_adapter_from_new_modules():
    from cw_diffusion.training import optimizer_groups
    from torch import nn

    m = nn.Module()
    m.dit = nn.Module()
    m.dit.backbone = nn.Linear(2, 2)
    m.world = nn.Linear(2, 2)
    groups = optimizer_groups(
        m, {"learning_rate": 1e-4, "backbone_learning_rate": 1e-5}
    )
    by_name = {g["group_kind"]: g for g in groups}
    assert by_name["backbone"]["lr"] == 1e-5 and by_name["new"]["lr"] == 1e-4
    assert {id(x) for x in by_name["backbone"]["params"]} == {
        id(x) for x in m.dit.backbone.parameters()
    }
    assert {id(x) for x in by_name["new"]["params"]} == {
        id(x) for x in m.world.parameters()
    }
