import pytest
import torch
from cw_diffusion.data import CanonicalDataset
from cw_diffusion.world import WorldEncoder
from cw_diffusion.config import WorldConfig
from cw_diffusion.losses import masked_mean


def test_test_split_supervision_is_rejected_before_any_io():
    data = CanonicalDataset.__new__(CanonicalDataset)
    data.split = "test"
    with pytest.raises(ValueError, match="Test supervision"):
        data.supervision(0)


def test_inference_world_encoder_rejects_privileged_labels():
    cfg = WorldConfig(
        feature_dim=8,
        hidden_dim=32,
        object_queries=2,
        gaussians_per_object=2,
        decoder_layers=1,
        background_side=2,
        background_levels=2,
        injection_blocks=(0,),
        render_backend="reference",
    )
    model = WorldEncoder(cfg)
    with pytest.raises(TypeError):
        model(torch.zeros(3, 32, 32), torch.eye(4), None, instance=torch.zeros(32, 32))
    with pytest.raises(TypeError):
        model(torch.zeros(3, 32, 32), torch.eye(4), None, depth=torch.ones(32, 32))


def test_invalid_depth_mask_has_finite_zero_loss_and_gradient():
    x = torch.tensor([float("inf"), 1.0, 3.0], requires_grad=True)
    loss = masked_mean(x, torch.tensor([False, True, True]))
    assert loss == 2
    loss.backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.0, 0.5, 0.5]))


def test_queries_require_valid_capacity_and_height_range():
    with pytest.raises(ValueError):
        WorldConfig(min_height=3, max_height=2).validate()
    with pytest.raises(ValueError):
        WorldConfig(injection_blocks=(2, 1)).validate()
    with pytest.raises(ValueError):
        WorldConfig(object_queries=0).validate()
