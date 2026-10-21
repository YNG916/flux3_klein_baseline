"""Explicit two-process CPU DDP probe; does not allocate a GPU or submit a job."""
import json
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from test_native import tiny_dit, sample
from flux_mvwd.backend import flow_loss
from flux_mvwd.lora import inject, gradient_checkpointing, adapter_state


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    with tiny_dit() as factory:
        torch.manual_seed(42)
        model = factory(); inject(model, 2, 2); gradient_checkpointing(model)
        wrapped = DistributedDataParallel(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for step in range(2):
            optimizer.zero_grad()
            loss = flow_loss(wrapped, [sample(rank)], {"cfg_dropout": 0.0}, "cpu",
                             generator=torch.Generator().manual_seed(100 + rank + step))
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step()
        for key, value in adapter_state(model).items():
            copies = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(copies, value)
            for other in copies: torch.testing.assert_close(other, value, rtol=0, atol=0)
        if rank == 0:
            print(json.dumps({"status": "passed", "world_size": world, "backend": "gloo",
                "optimizer_updates": 2, "native_dit": True, "nonreentrant_checkpointing": True,
                "adapter_states_identical_across_ranks": True}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__": main()
