"""Lossless RGB/label storage and per-rank content-addressed static BEV payloads."""

import torch

BEV_TARGETS = (
    "bev_semantic",
    "bev_instance",
    "bev_height",
    "bev_height_valid",
    "bev_occupancy",
)


def compact(condition, target):
    condition = dict(condition)
    target = dict(target)
    condition["bev"] = (condition["bev"] * 255).round().clamp(0, 255).to(torch.uint8)
    target["rgb"] = (target["rgb"] * 255).round().clamp(0, 255).to(torch.uint8)
    for name in ["semantic", "bev_semantic"]:
        x = target[name]
        if x.min() < 0 or x.max() > 255:
            raise ValueError(
                "Vocabulary does not fit byte storage; update storage schema explicitly"
            )
        target[name] = x.to(torch.uint8)
    for name in ["instance", "bev_instance"]:
        x = target[name]
        target[name] = (
            x.to(torch.uint16)
            if x.min() >= 0 and x.max() <= 65535
            else x.to(torch.int64)
        )
    target["bev_occupancy"] = target["bev_occupancy"].to(torch.uint8)
    # DINO precision is explicit; all geometric quantities retain FP32.
    target["teacher"] = target["teacher"].to(torch.bfloat16)
    return condition, target


def split_static(condition, target):
    condition = dict(condition)
    target = dict(target)
    static = {
        "bev": condition.pop("bev"),
        "world_to_bev": condition.pop("world_to_bev"),
        "targets": {k: target.pop(k) for k in BEV_TARGETS},
    }
    return condition, target, static


def restore(sample, static):
    sample = dict(sample)
    condition = dict(sample["condition"])
    target = dict(sample["supervision"])
    condition.update(
        bev=static["bev"].float() / 255, world_to_bev=static["world_to_bev"]
    )
    target.update(static["targets"])
    target["rgb"] = target["rgb"].float() / 255
    for name in ["semantic", "instance", "bev_semantic", "bev_instance"]:
        target[name] = target[name].long()
    target["bev_occupancy"] = target["bev_occupancy"].float()
    sample.update(condition=condition, supervision=target)
    return sample
