"""Isolate each vLLM replica from the parent's Accelerate process group."""
from __future__ import annotations
import base64
import io
import json
import os
import select
import subprocess
import sys
import time


def isolated_environment(environment, local_device):
    env = dict(environment)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else None
    env["CUDA_VISIBLE_DEVICES"] = devices[local_device] if devices else str(local_device)
    for key in list(env):
        if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"} or key.startswith(("TORCHELASTIC_", "ACCELERATE_")):
            env.pop(key, None)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return env


CONCISE_RETRY = ("Write one concise camera-view instruction in at most 120 words. "
    "Do not copy coordinate tables or matrices; exact constraints are appended separately. "
    "Do not repeat phrases. End the instruction normally.")
FALLBACK_INSTRUCTION = "Generate the requested mounted robot camera RGB view from the reference maps and the exact constraints below."


def overflow_settings(config):
    settings = config["prompt"].get("overflow")
    if settings is None:
        return None
    if (settings.get("policy") != "concise_retry_then_exact_v1"
            or type(settings.get("retry_max_new_tokens")) is not int
            or not 1 <= settings["retry_max_new_tokens"] <= 1024
            or not 1 <= settings.get("retry_repetition_penalty", 1.1) <= 2):
        raise ValueError("Invalid prompt overflow policy")
    return settings


def generate_resilient(llm, requests, params, *, retry_requests=None, retry_params=None):
    """Retry only length/empty failures, retain every attempt, never mask engine errors."""
    def unpack(outputs, count, strategy, budget):
        if len(outputs) != count:
            raise ValueError("vLLM output count differs from request count")
        records = []
        for output in outputs:
            choice = output.outputs[0]
            reason, caption = choice.finish_reason, choice.text.strip()
            if reason not in ("stop", "length"):
                raise RuntimeError(f"Unexpected vLLM finish reason: {reason}")
            records.append({"caption": caption, "output_tokens": len(choice.token_ids),
                "finish_reason": reason, "strategy": strategy, "max_tokens": budget})
        return records
    initial = unpack(llm.generate(requests, params, use_tqdm=False), len(requests), "official_initial", params.max_tokens)
    histories = [[row] for row in initial]
    bad = [i for i, row in enumerate(initial) if row["finish_reason"] != "stop" or not row["caption"]]
    if bad and retry_params is not None:
        if retry_requests is None or len(retry_requests) != len(requests):
            raise ValueError("Retry request support differs")
        retried = unpack(llm.generate([retry_requests[i] for i in bad], retry_params, use_tqdm=False),
            len(bad), "concise_retry", retry_params.max_tokens)
        for i, row in zip(bad, retried):
            histories[i].append(row)
    result = []
    for i, history in enumerate(histories):
        last = history[-1]
        ok = last["finish_reason"] == "stop" and bool(last["caption"])
        status = ("ok" if len(history) == 1 else "retry_ok") if ok else (
            "exact_request_fallback" if retry_params is not None else "failed")
        result.append({**last, "attempts": history, "caption_status": status})
        if len(history) > 1 or not ok:
            print(f"caption recovery batch_index={i} status={status} "
                  f"attempts={[(r['finish_reason'], r['output_tokens']) for r in history]}", flush=True)
    return result


def format_caption(processor, item, exact):
    from .preparation import fit_caption
    status = item.get("caption_status", "ok")
    fallback = status == "exact_request_fallback"
    if status == "failed" or (not fallback and (item["finish_reason"] != "stop" or not item["caption"].strip())):
        raise ValueError("Upsampler did not finish; configure an explicit overflow policy in a new experiment")
    record = fit_caption(processor, FALLBACK_INSTRUCTION if fallback else item["caption"], exact)
    if fallback:
        record.update(raw_upsampler_caption=item["caption"], fallback_instruction=FALLBACK_INSTRUCTION)
    record.update(caption_status=status, upsampler_caption_used=not fallback,
                  upsampler_attempts=item.get("attempts", []), finish_reason=item["finish_reason"])
    return record


class RemoteUpsampler:
    def __init__(self, config, device):
        import torch
        from transformers import AutoProcessor
        from .runtime import model_paths
        overflow_settings(config)
        self.text_processor = AutoProcessor.from_pretrained(model_paths(config) / "text_encoder", local_files_only=True)
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        read_fd, write_fd = os.pipe()
        self.process = subprocess.Popen([sys.executable, "-m", "flux_mvwd.caption_worker", str(write_fd)],
            env=isolated_environment(os.environ, index), stdin=subprocess.PIPE,
            pass_fds=(write_fd,), text=True, bufsize=1)
        os.close(write_fd)
        self.responses = os.fdopen(read_fd)
        self._send({"config": config})
        ready = self._receive()
        if ready.get("status") != "ready":
            raise RuntimeError("vLLM worker failed to initialize")

    def _send(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def _receive(self):
        while True:
            readable, _, _ = select.select([self.responses], [], [], 1)
            if readable:
                line = self.responses.readline()
                if not line:
                    raise RuntimeError("vLLM worker response pipe closed; inspect the step log")
                value = json.loads(line)
                if "error" in value:
                    raise RuntimeError(value["error"])
                return value
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM worker exited with {self.process.returncode}; inspect the step log")

    def batch(self, conditions):
        rows = []
        for condition in conditions:
            images = []
            for image in condition["images"]:
                buffer = io.BytesIO(); image.save(buffer, format="PNG")
                images.append(base64.b64encode(buffer.getvalue()).decode())
            rows.append({"prompt": condition["prompt"], "images": images})
        self._send({"samples": rows})
        response = self._receive()
        if len(response["samples"]) != len(conditions):
            raise ValueError("vLLM output count differs from caption batch")
        result = []
        for condition, item in zip(conditions, response["samples"]):
            result.append({**format_caption(self.text_processor, item, condition["prompt"]),
                "engine": "vllm", "generation_batch_seconds": response["seconds"],
                "generation_batch_size": len(conditions), "output_tokens": item["output_tokens"]})
        return result

    def close(self):
        if self.process.poll() is None:
            try:
                self._send({"close": True})
                self.process.stdin.close()
                self.process.wait(timeout=60)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=30)
        self.responses.close()


def main():
    # The parent can be terminated by torchrun/Slurm; do not leave a GPU engine orphaned.
    import ctypes
    import signal
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    response = os.fdopen(int(sys.argv[1]), "w", buffering=1)
    def send(value): response.write(json.dumps(value) + "\n"); response.flush()
    llm = None
    try:
        config = json.loads(sys.stdin.readline())["config"]
        from PIL import Image
        from flux3.pup import policy_reference, I2I_SCAFFOLD, SAMPLING, STOP_TOKENS, ENGINE, MAX_REFERENCES
        from vllm import SamplingParams, LLM
        from transformers import AutoProcessor
        from types import SimpleNamespace
        from .runtime import model_paths
        from .preparation import upsampler_instruction
        root = model_paths(config, pup=True)
        llm = LLM(model=str(root), dtype="bfloat16", tensor_parallel_size=1,
            enable_prefix_caching=True, generation_config="vllm",
            limit_mm_per_prompt={"image": MAX_REFERENCES}, attention_config={"backend": "TRITON_ATTN"},
            additional_config={"gdn_prefill_backend": "triton"},
            gpu_memory_utilization=config["prompt"].get("memory_fraction", 0.75), **ENGINE)
        upsampler = SimpleNamespace(llm=llm, tokenizer=llm.get_tokenizer(),
            processor=AutoProcessor.from_pretrained(root, local_files_only=True),
            prompts=json.loads((root / "system_prompts.json").read_text()))
        settings = {**SAMPLING, "max_tokens": config["prompt"]["max_new_tokens"]}
        params = SamplingParams(**settings, presence_penalty=0.,
            stop_token_ids=[upsampler.tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS])
        overflow = overflow_settings(config)
        retry_params = None if overflow is None else SamplingParams(
            **{**settings, "max_tokens": overflow["retry_max_new_tokens"]}, presence_penalty=0.,
            repetition_penalty=overflow.get("retry_repetition_penalty", 1.1),
            stop_token_ids=[upsampler.tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS])
        send({"status": "ready"})
        for line in sys.stdin:
            command = json.loads(line)
            if command.get("close"): break
            requests, retry_requests = [], []
            for sample in command["samples"]:
                source = [Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB") for s in sample["images"]]
                images = [policy_reference(image, len(source)) for image in source]
                messages = [{"role": "system", "content": upsampler.prompts["i2i"]}, {"role": "user", "content":
                    [{"type": "image"} for _ in images] + [{"type": "text", "text": I2I_SCAFFOLD[False].format(instruction=upsampler_instruction(sample["prompt"], config))}]}]
                text = upsampler.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                requests.append({"prompt": text, "multi_modal_data": {"image": images}})
                if overflow is not None:
                    messages[-1]["content"][-1]["text"] += "\n\n" + CONCISE_RETRY
                    retry_text = upsampler.processor.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
                    retry_requests.append({"prompt": retry_text, "multi_modal_data": {"image": images}})
            started = time.monotonic()
            outputs = generate_resilient(llm, requests, params, retry_requests=retry_requests,
                                         retry_params=retry_params)
            send({"seconds": time.monotonic() - started, "samples": outputs})
    except BaseException as error:
        import traceback
        traceback.print_exc()
        send({"error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if llm is not None: llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__": main()
