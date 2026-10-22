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


from flux_mvwd.caption_worker import generate_resilient, format_caption, overflow_settings


def choice(text, reason='stop', tokens=3):
    return SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason=reason, token_ids=[1] * tokens)])


class ScriptedEngine:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []
    def generate(self, requests, params, **kwargs):
        self.calls.append((requests, params.max_tokens))
        return next(self.responses)


def test_retry_targets_only_failed_queries_and_preserves_all_attempts():
    engine = ScriptedEngine([choice('normal'), choice('repeat', 'length', 2048), choice('')],
                            [choice('concise'), choice('still long', 'length', 512)])
    result = generate_resilient(engine, ['a', 'b', 'c'], SimpleNamespace(max_tokens=2048),
        retry_requests=['ra', 'rb', 'rc'], retry_params=SimpleNamespace(max_tokens=512))
    assert engine.calls == [(['a', 'b', 'c'], 2048), (['rb', 'rc'], 512)]
    assert [x['caption_status'] for x in result] == ['ok', 'retry_ok', 'exact_request_fallback']
    assert result[1]['attempts'][0]['caption'] == 'repeat'
    assert result[2]['attempts'][1]['output_tokens'] == 512


def test_engine_errors_are_not_masked_as_caption_fallback():
    engine = ScriptedEngine([choice('aborted', 'abort')])
    with pytest.raises(RuntimeError, match='finish reason'):
        generate_resilient(engine, ['a'], SimpleNamespace(max_tokens=2048),
            retry_requests=['ra'], retry_params=SimpleNamespace(max_tokens=512))
    assert len(engine.calls) == 1


def test_legacy_policy_still_rejects_incomplete_outputs():
    engine = ScriptedEngine([choice('truncated', 'length')])
    result = generate_resilient(engine, ['a'], SimpleNamespace(max_tokens=2048))
    assert result[0]['caption_status'] == 'failed' and len(engine.calls) == 1
    with pytest.raises(ValueError, match='did not finish'):
        format_caption(None, result[0], 'camera geometry')


def test_fallback_keeps_full_geometry_and_does_not_use_truncated_prose():
    exact = 'K=[100,0,32]; T=[1,2,.8]; robots=' + 'original;' * 100
    processor = SimpleNamespace(apply_chat_template=lambda messages, **kw: messages[0]['content'],
        tokenizer=lambda text: {'input_ids': list(text)})
    item = {'caption_status': 'exact_request_fallback', 'caption': 'TRUNCATED garbage',
            'finish_reason': 'length', 'attempts': [{'caption': 'TRUNCATED garbage'}]}
    record = format_caption(processor, item, exact)
    assert record['prompt'].endswith(exact) and 'TRUNCATED' not in record['prompt']
    assert record['raw_upsampler_caption'] == 'TRUNCATED garbage'
    assert not record['upsampler_caption_used'] and record['upsampler_attempts'] == item['attempts']


@pytest.mark.parametrize('settings', [
    {'policy':'unknown', 'retry_max_new_tokens':512},
    {'policy':'concise_retry_then_exact_v1','retry_max_new_tokens':0},
    {'policy':'concise_retry_then_exact_v1','retry_max_new_tokens':512,'retry_repetition_penalty':.5}])
def test_recovery_config_rejects_invalid_settings(settings):
    with pytest.raises(ValueError, match='overflow policy'):
        overflow_settings({'prompt': {'overflow': settings}})
