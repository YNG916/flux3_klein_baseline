import torch
from cw_diffusion.storage import compact, split_static, restore


def test_rgb_and_labels_roundtrip_and_shared_static_payload():
    rgb = torch.randint(0, 256, (3, 3, 4, 4), dtype=torch.uint8)
    bev = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    condition = {
        "bev": bev.float() / 255,
        "world_to_bev": torch.eye(4),
        "ground_truth_loaded": False,
    }
    target = {
        "rgb": rgb.float() / 255,
        "semantic": torch.full((3, 4, 4), 252, dtype=torch.long),
        "instance": torch.full((3, 4, 4), 1328, dtype=torch.long),
        "bev_semantic": torch.ones(8, 8, dtype=torch.long),
        "bev_instance": torch.ones(8, 8, dtype=torch.long) * 1328,
        "bev_height": torch.randn(8, 8),
        "bev_height_valid": torch.ones(8, 8, dtype=torch.bool),
        "bev_occupancy": torch.ones(8, 8),
        "teacher": torch.randn(3, 4, 384),
        "depth_linear": torch.rand(3, 4, 4),
    }
    c, t = compact(condition, target)
    c, t, static = split_static(c, t)
    assert "bev" not in c and "bev_instance" not in t
    assert static["bev"].dtype == torch.uint8 and t["instance"].dtype == torch.uint16
    actual = restore({"condition": c, "supervision": t}, static)
    torch.testing.assert_close(
        actual["condition"]["bev"], condition["bev"], atol=0, rtol=0
    )
    for name in ["rgb", "semantic", "instance", "bev_height", "depth_linear"]:
        torch.testing.assert_close(
            actual["supervision"][name], target[name], atol=0, rtol=0
        )


def test_large_instance_ids_never_silently_wrap():
    c = {"bev": torch.zeros(3, 2, 2)}
    t = {
        "rgb": torch.zeros(3, 3, 2, 2),
        "semantic": torch.zeros(3, 2, 2, dtype=torch.long),
        "bev_semantic": torch.zeros(2, 2, dtype=torch.long),
        "instance": torch.ones(3, 2, 2, dtype=torch.long) * 2**31,
        "bev_instance": torch.ones(2, 2, dtype=torch.long) * 2**31,
        "bev_occupancy": torch.zeros(2, 2),
        "teacher": torch.zeros(3, 1, 384),
    }
    _, t = compact(c, t)
    assert t["instance"].dtype == torch.int64 and t["instance"].max() == 2**31
