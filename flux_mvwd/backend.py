"""Native FLUX 3 Base training contract; no Diffusers model approximation."""
from contextlib import nullcontext
import torch


def configure_attention(backend="auto"):
    """Upstream forces cuDNN; allow PyTorch's fused backend selection for training."""
    import flux3.dit as dit
    from torch.nn.attention import SDPBackend, sdpa_kernel
    choices = {"auto": [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
               "flash": [SDPBackend.FLASH_ATTENTION], "cudnn": [SDPBackend.CUDNN_ATTENTION],
               "math": [SDPBackend.MATH]}
    choices["varlen"] = choices["auto"]
    if backend not in choices:
        raise ValueError("Unknown attention backend")
    def attention(q, k, v, bounds):
        # Variable text lengths are legal: bounds isolate every sample. A single
        # tensor batch handles equal lengths; unequal lengths use fused per-sample SDPA.
        lengths = [b-a for a, b in bounds]
        if not lengths or not any(lengths):
            return q.new_zeros(q.shape[2], q.shape[1] * q.shape[3])
        context = lambda: sdpa_kernel(choices[backend]) if q.is_cuda else nullcontext()
        if q.is_cuda and backend in ("auto", "varlen") and len(bounds) > 1 and len(set(lengths)) > 1:
            from torch.nn.attention.varlen import varlen_attn
            cumulative = torch.tensor([0, *[b for _, b in bounds]], device=q.device, dtype=torch.int32)
            flatten = lambda z: z[0].transpose(0, 1).contiguous()
            out = varlen_attn(flatten(q), flatten(k), flatten(v), cumulative, cumulative, max(lengths), max(lengths))
            return out.reshape(q.shape[2], -1)
        if len(set(lengths)) == 1:
            count, length = len(bounds), lengths[0]
            split = lambda z: z[0].reshape(z.shape[1], count, length, z.shape[3]).permute(1, 0, 2, 3)
            with context():
                out = torch.nn.functional.scaled_dot_product_attention(split(q), split(k), split(v))
            return out.transpose(1, 2).reshape(count * length, -1)
        parts = []
        for a, b in bounds:
            if b == a:
                continue
            with context():
                out = torch.nn.functional.scaled_dot_product_attention(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b])
            parts.append(out[0].transpose(0, 1).reshape(b-a, -1))
        return torch.cat(parts)
    dit.packed_attention = attention


def load_dit(config, device):
    from flux3.dit import DiT
    from flux3.checkpoint import load_checkpoint, is_flow_map
    from .runtime import check_upstream, model_paths
    check_upstream(config)
    path = model_paths(config)
    if is_flow_map(path):
        raise ValueError("Distilled flow-map weights cannot train this Base baseline")
    with torch.device("meta"):
        model = DiT(False)
    model = load_checkpoint(model, str(path), str(device), torch.bfloat16)
    configure_attention(config["training"].get("attention_backend", "auto"))
    return model


def load_encoders(config, device, *, include_text=True):
    from flux3.autoencoder import AutoEncoder
    from flux3.text_encoder import TextEncoder
    from flux3.checkpoint import load_checkpoint
    from .runtime import check_upstream, model_paths
    check_upstream(config)
    path = model_paths(config)
    with torch.device("meta"):
        vae = AutoEncoder()
    vae = load_checkpoint(vae, str(path / "ae.safetensors"), str(device), torch.float32).requires_grad_(False)
    text = load_text_encoder(config, device) if include_text else None
    return vae, text


def load_text_encoder(config, device):
    from flux3.text_encoder import TextEncoder
    from .runtime import model_paths
    text = TextEncoder(str(model_paths(config) / "text_encoder"), str(device))
    text.model.requires_grad_(False)
    return text


def checked_text(text, prompt, *, cpu=True):
    from flux3.text_encoder import MAX_LENGTH
    formatted = text.processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    length = len(text.processor.tokenizer(formatted)["input_ids"])
    if length > MAX_LENGTH:
        raise ValueError(f"Prompt is {length} tokens; refusing to silently truncate geometry at {MAX_LENGTH}")
    tokens, ids = text.encode(prompt)
    # Clone outside TextEncoder.encode's inference_mode: trainable projections
    # need ordinary tensors to save these frozen features for backward.
    if cpu:
        return tokens.cpu().clone().contiguous(), ids.cpu().clone().contiguous()
    return tokens.clone().contiguous(), ids.clone().contiguous()


def encode_image(vae, image, device):
    from flux3.pipeline import to_tensor, scaled_ids
    with torch.inference_mode():
        tokens, ids = vae.encode(to_tensor(image).to(device))
    return tokens.to(torch.bfloat16).cpu().contiguous(), scaled_ids(ids).cpu().contiguous()


def flow_loss(model, samples, settings, device, *, generator=None, training=True, text_encoder=None):
    from flux3.pipeline import pack
    data, expected = [], []
    for sample in samples:
        target = sample["target"][0].to(device, dtype=torch.bfloat16, non_blocking=True)
        image_ids = sample["target"][1].to(device, non_blocking=True)
        noise = torch.randn(target.shape, device=device, dtype=torch.float32, generator=generator)
        if settings.get("timestep_sampling", "logit_normal") == "logit_normal":
            u = torch.sigmoid(torch.randn((), device=device, generator=generator) * settings.get("logit_std", 1.0) + settings.get("logit_mean", 0.0))
        else:
            u = torch.rand((), device=device, generator=generator)
        shift = settings.get("shift", 16.0)
        t = shift * u / (1 + (shift - 1) * u)
        noisy = ((1-t) * target.float() + t * noise).to(torch.bfloat16)
        drop = training and torch.rand((), device=device, generator=generator).item() < settings.get("cfg_dropout", 0.1)
        if sample.get("text") is None:
            if text_encoder is None:
                raise ValueError("Online text features require the frozen native text encoder")
            text = checked_text(text_encoder, "" if drop else sample["prompt"], cpu=False)
        else:
            text = sample["unconditional"] if drop else sample["text"]
        text = tuple(x.to(device, non_blocking=True) for x in text)
        refs = [tuple(x.to(device, non_blocking=True) for x in pair) for pair in sample["references"]]
        cond_tokens, cond_ids = [], []
        for index, (tokens, ids) in enumerate(refs):
            ids = ids.clone(); ids[:, 3] = -(index+1)
            cond_tokens.append(tokens); cond_ids.append(ids)
        data.append([text, (torch.cat(cond_tokens), torch.cat(cond_ids)), (noisy, image_ids)])
        expected.append((noise - target.float(), t))
    packed = pack(data)
    for (a, b), (_, t) in zip(packed.mode_indices["image"], expected):
        packed.timesteps["image"][a:b] = t
    # Text/reference timesteps stay zero. Only target image tokens are predicted.
    prediction = model(packed)
    losses = []
    for (a, b), (velocity, _) in zip(packed.mode_indices["image"], expected):
        losses.append((prediction[a:b].float() - velocity).square().mean())
    return torch.stack(losses).mean()
