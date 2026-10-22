"""GPU replicas must not inherit Accelerate's distributed group or GPU mapping."""
import os
from types import SimpleNamespace
import pytest
from flux_mvwd.caption_worker import isolated_environment, RemoteUpsampler
from flux_mvwd.preparation import upsampler_instruction


def test_replica_isolates_gpu_and_rendezvous_but_preserves_slurm_allocation():
    env = {"CUDA_VISIBLE_DEVICES": "3,5,6,7", "RANK": "2", "LOCAL_RANK": "2",
           "WORLD_SIZE": "4", "MASTER_PORT": "29500", "SLURM_JOB_ID": "93041",
           "TORCHELASTIC_RUN_ID": "launch", "ACCELERATE_MIXED_PRECISION": "bf16"}
    result = isolated_environment(env, 2)
    assert result["CUDA_VISIBLE_DEVICES"] == "6"
    assert result["SLURM_JOB_ID"] == "93041"
    assert all(k not in result for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT", "TORCHELASTIC_RUN_ID", "ACCELERATE_MIXED_PRECISION"))
    assert env["CUDA_VISIBLE_DEVICES"] == "3,5,6,7" and env["RANK"] == "2"


def test_worker_failure_is_reported_instead_of_waiting_for_engine_descendants():
    read_fd, write_fd = os.pipe()
    worker = RemoteUpsampler.__new__(RemoteUpsampler)
    worker.responses = os.fdopen(read_fd)
    worker.process = SimpleNamespace(poll=lambda: 5, returncode=5)
    try:
        with pytest.raises(RuntimeError, match="exited with 5"):
            worker._receive()
    finally:
        worker.responses.close(); os.close(write_fd)


def test_camera_view_policy_preserves_exact_request_and_legacy_policy():
    original = "Camera K=[100,0,32]; camera position [1,2,.8]; all three robot states."
    assert upsampler_instruction(original, {"prompt": {"policy": "official_pup_plus_exact_geometry_v1"}}) == original
    new = upsampler_instruction(original, {"prompt": {"policy": "official_pup_camera_view_v2"}})
    assert new.endswith(original)
    assert "specified mounted robot camera" in new
    assert "not the desired output viewpoint" in new
