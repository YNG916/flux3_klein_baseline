"""Four-rank CPU control-plane probe with native complete checkpoint roundtrip."""

import json
import signal
import sys
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tools.cw_batch_runtime import StopController, verify_checkpoint
from cw_diffusion.checkpoint import save_checkpoint, load_checkpoint


def main():
    acc = Accelerator(cpu=True)
    assert acc.num_processes == 4
    set_seed(731)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model, optimizer = acc.prepare(model, optimizer)
    root = Path(sys.argv[1])
    if acc.is_main_process:
        root.mkdir(parents=True, exist_ok=False)
    acc.wait_for_everyone()
    flag = root / f"rank_{acc.process_index}_STOP"
    controller = StopController(flag, 5)
    stop = [False]
    old = signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    generator = torch.Generator().manual_seed(100 + acc.process_index)
    contract = dict(world_size=4, padded_groups=40, accumulation=1)
    progress = dict(step=0, epoch=0, cursor=0, global_groups_seen=0)
    while not stop[0]:
        loss = model(torch.randn(2, 2, generator=generator)).square().mean()
        acc.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        progress.update(
            step=progress["step"] + 1,
            cursor=progress["cursor"] + 1,
            global_groups_seen=progress["global_groups_seen"] + 4,
        )
        if acc.process_index == 1 and progress["step"] == 2:
            flag.touch()
        controller.optimizer_boundary(acc, loss.detach(), Accelerator.reduce)
    assert progress["step"] == 2
    checkpoint = root / "interrupted"
    save_checkpoint(checkpoint, model, optimizer, progress, contract, acc, generator)
    expected_rng = torch.randn(3, generator=generator)
    expected_weights = {
        k: v.clone() for k, v in acc.unwrap_model(model).state_dict().items()
    }
    resumed = load_checkpoint(
        checkpoint,
        acc.unwrap_model(model),
        optimizer,
        contract,
        acc.process_index,
        "cpu",
        generator,
    )
    assert resumed == progress
    torch.testing.assert_close(torch.randn(3, generator=generator), expected_rng)
    for k, v in acc.unwrap_model(model).state_dict().items():
        torch.testing.assert_close(v, expected_weights[k], atol=0, rtol=0)
    verify_checkpoint(checkpoint, contract)
    acc.wait_for_everyone()
    if acc.is_main_process:
        result = dict(
            complete=True,
            world_size=4,
            device="cpu",
            stopped_step=2,
            one_rank_request_stops_all_ranks=True,
            native_optimizer_rng_checkpoint_resume=True,
            gpu_or_research_training_started=False,
        )
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    signal.signal(signal.SIGTERM, old)
    acc.end_training()


if __name__ == "__main__":
    main()
