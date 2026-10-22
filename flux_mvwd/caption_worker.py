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


class RemoteUpsampler:
    def __init__(self, config, device):
        import torch
        from transformers import AutoProcessor
        from .runtime import model_paths
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
        from .preparation import fit_caption
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
            if item["finish_reason"] != "stop":
                raise ValueError("Upsampler reached its token budget; preserve the failure and use a new config")
            result.append({**fit_caption(self.text_processor, item["caption"], condition["prompt"]),
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
        send({"status": "ready"})
        for line in sys.stdin:
            command = json.loads(line)
            if command.get("close"): break
            requests = []
            for sample in command["samples"]:
                source = [Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB") for s in sample["images"]]
                images = [policy_reference(image, len(source)) for image in source]
                messages = [{"role": "system", "content": upsampler.prompts["i2i"]}, {"role": "user", "content":
                    [{"type": "image"} for _ in images] + [{"type": "text", "text": I2I_SCAFFOLD[False].format(instruction=upsampler_instruction(sample["prompt"], config))}]}]
                text = upsampler.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                requests.append({"prompt": text, "multi_modal_data": {"image": images}})
            started = time.monotonic()
            outputs = llm.generate(requests, params, use_tqdm=False)
            send({"seconds": time.monotonic() - started, "samples": [
                {"caption": out.outputs[0].text.strip(), "output_tokens": len(out.outputs[0].token_ids),
                 "finish_reason": out.outputs[0].finish_reason} for out in outputs]})
    except BaseException as error:
        import traceback
        traceback.print_exc()
        send({"error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if llm is not None: llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__": main()
