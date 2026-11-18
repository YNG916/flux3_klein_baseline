from pathlib import Path
import signal
import torch
import pytest

from tools.cw_batch_runtime import StopController, verify_checkpoint
from cw_diffusion.checkpoint import save_checkpoint
from types import SimpleNamespace


def test_soft_stop_uses_native_handler_without_signalling_workers(tmp_path):
    seen = []
    old = signal.signal(signal.SIGTERM, lambda *args: seen.append(args))
    try:
        control = StopController(tmp_path / "STOP", 2000)
        control.checkpoint_complete({"step": 1999})
        assert not seen
        control.checkpoint_complete({"step": 2000})
        control.checkpoint_complete({"step": 2000})
        assert len(seen) == 1 and seen[0] == (signal.SIGTERM, None)
        assert control.reason == "next_complete_periodic_checkpoint"
    finally:
        signal.signal(signal.SIGTERM, old)


def test_advance_warning_is_synchronized_before_requesting_stop(tmp_path):
    seen = []
    old = signal.signal(signal.SIGTERM, lambda *_: seen.append(True))
    try:
        control = StopController(tmp_path / "STOP", 2000)

        def remote_rank_requested(acc, value, reduction):
            assert reduction == "sum" and value.item() == 0
            return value.new_tensor(1)

        control.optimizer_boundary(None, torch.tensor(1.0), remote_rank_requested)
        assert seen and control.reason == "slurm_advance_warning"
    finally:
        signal.signal(signal.SIGTERM, old)


def test_no_default_sigterm_fallback_can_kill_training_process(tmp_path):
    old = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        with pytest.raises(RuntimeError, match="cooperative"):
            StopController(tmp_path / "STOP", 1).checkpoint_complete({"step": 1})
    finally:
        signal.signal(signal.SIGTERM, old)


def test_full_budget_mode_does_not_stop_at_first_periodic_checkpoint(tmp_path):
    seen = []
    old = signal.signal(signal.SIGTERM, lambda *_: seen.append(True))
    try:
        control = StopController(tmp_path / "STOP", 23250)
        control.checkpoint_complete({"step": 2000})
        control.checkpoint_complete({"step": 23000})
        assert not seen
        control.checkpoint_complete({"step": 23250})
        assert seen
    finally:
        signal.signal(signal.SIGTERM, old)


def test_batch_checkpoint_verification_requires_complete_optimizer_and_rng(tmp_path):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    contract = dict(world_size=1, accumulation=1, padded_groups=10)
    progress = dict(step=2, epoch=0, cursor=2, global_groups_seen=2)
    acc = SimpleNamespace(
        is_main_process=True,
        process_index=0,
        device=torch.device("cpu"),
        unwrap_model=lambda x: x,
        wait_for_everyone=lambda: None,
    )
    path = tmp_path / "checkpoint"
    save_checkpoint(path, model, opt, progress, contract, acc, torch.Generator())
    assert verify_checkpoint(path, contract)["progress"] == progress
    with (path / "optimizer.pt").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(path, contract)


def test_batch_has_advance_warning_and_propagates_failure_status():
    text = (Path(__file__).resolve().parents[1] / "slurm/cw_resume.sbatch").read_text()
    assert "--signal=B:USR1@2700" in text
    assert "--time=2-00:00:00" in text
    assert 'exit "$child_status"' in text
    assert "Missing verified completion receipt" in text
