"""GT-free synchronous generation with persistent per-CFG-branch canonical memory."""

from pathlib import Path
import json
import torch
from flux_mvwd.common import read_json, write_json, sample_seed, digest, file_hash
from flux_mvwd.backend import load_encoders, checked_text, encode_image
from .workflow import config_at, move
from .data import CanonicalDataset
from .native import SceneContext
from .training import build_system
from .checkpoint import load_model, source_identity, environment_identity


@torch.inference_mode()
def infer(
    experiment,
    name,
    checkpoint,
    split="test",
    seed=42,
    max_episodes=None,
    resume_predictions=None,
):
    from accelerate import Accelerator
    from flux3.pipeline import pack, scaled_ids, to_pil
    from flux3.autoencoder import latent_shape, latent_ids
    from flux3.sampling import timesteps

    p, c = config_at(experiment)
    acc = Accelerator()
    device = acc.device
    if device.type != "cuda":
        raise ValueError("Real generation requires an authorized GPU allocation")
    config = read_json(p / "baseline_config.json")
    data = CanonicalDataset(config, split, max_episodes=max_episodes)
    output = (
        Path(resume_predictions).resolve()
        if resume_predictions
        else p / "predictions" / name
    )
    if not output.resolve().is_relative_to((p / "predictions").resolve()):
        raise ValueError(
            "Resume directory must belong to the same independent method experiment"
        )
    meta = {
        "schema": "flux3-mvwd-predictions-v1",
        "complete": False,
        "method": "BEV-Grounded-Canonical-World-Diffusion",
        "config": config,
        "dataset": data.data.provenance(),
        "base_seed": seed,
        "ground_truth_loaded": False,
        "lora_sha256": digest(read_json(Path(checkpoint) / "metadata.json")["files"]),
        "seed_policy": "independent-query-sha256",
        "versions": environment_identity(),
        "view_count": len(data.groups) * 3,
        "method_sources": source_identity(),
        "method_checkpoint": str(Path(checkpoint).resolve()),
        "method_checkpoint_sha256": file_hash(Path(checkpoint) / "metadata.json"),
        "cross_view_exchange": True,
        "generation_mode": "coupled-three-view-separate-native-streams",
        "memory_scope": "one synchronous frame; separate unconditional/conditional histories",
    }
    from .prediction_store import PredictionStore

    lock = None
    if acc.is_main_process:
        store = PredictionStore(output, meta, create=not bool(resume_predictions))
        import fcntl

        lock = (output / ".writer.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    acc.wait_for_everyone()
    store = PredictionStore(output, meta)
    owned = range(acc.process_index, len(data.groups), acc.num_processes)
    pending = [i for i in owned if store.read_group(i, data.groups[i]) is None]
    if not pending:
        print(
            f"rank={acc.process_index} all assigned synchronous frames already complete",
            flush=True,
        )
    else:
        model = build_system(p, c, device).eval()
        load_model(
            checkpoint,
            model,
            {
                "config_sha256": file_hash(p / "config.json"),
                "sources": source_identity(),
                "environment": environment_identity(),
            },
        )
        vae, text = load_encoders(config, device)
    h, w = config["target_hw"]
    fh, fw = latent_shape(h, w)
    captions = Path(c["baseline_experiment"]) / "captions" / split
    cm = read_json(captions / "manifest.json")
    if not cm.get("complete") or cm["ground_truth_loaded"]:
        raise ValueError("Complete GT-free captions required")
    lookup = {
        (
            r["query"]["episode_id"],
            r["query"]["frame_index"],
            r["query"]["robots"][0],
        ): r
        for r in cm["samples"]
    }
    schedule = timesteps(config["inference"]["steps"], config["inference"]["shift"])
    with (output / f"{name}_rank_{acc.process_index:04d}.jsonl").open("x") as stream:
        for index in pending:
            raw = data.conditions(index)
            native = raw.pop("native_conditions")
            condition = move(raw, device)
            group = raw["group"]
            generator = torch.Generator(device=device).manual_seed(
                int(digest([seed, group, "world-appearance"])[:15], 16)
            )
            with torch.autocast(device_type="cuda", enabled=False):
                initial = model.world(
                    condition["bev"],
                    condition["world_to_bev"],
                    condition["robot_states"],
                    generator=generator,
                )
            states = [initial.state, initial.state]
            samples = []
            prompts = []
            seeds = []
            idlist = []
            images = []
            for robot, cond in enumerate(native):
                cap = read_json(
                    captions
                    / lookup[(group["episode_id"], group["frame_index"], robot)]["path"]
                )
                from flux_mvwd.preparation import input_key

                caption_row = lookup[(group["episode_id"], group["frame_index"], robot)]
                if cap["ground_truth_loaded"] is not False or any(
                    file_hash(captions / f) != sha
                    for f, sha in caption_row["files"].items()
                ):
                    raise ValueError("Caption file/provenance differs")
                if cap["input_sha256"] != input_key(cond):
                    raise ValueError("Caption belongs to different BEV/camera input")
                s = sample_seed(
                    seed,
                    data.data.meta["release_id"],
                    group["episode_id"],
                    group["frame_index"],
                    [robot],
                    "independent",
                )
                seeds.append(s)
                noise = (
                    torch.randn(
                        (fh * fw, 128), generator=torch.Generator().manual_seed(s)
                    )
                    .to(torch.bfloat16)
                    .float()
                    .to(device)
                )
                ids = latent_ids(fh, fw, device)
                idlist.append(ids)
                images.append(noise)
                refs = []
                refids = []
                for j, im in enumerate(cond["images"]):
                    a, b = encode_image(vae, im, device)
                    a, b = a.to(device), b.to(device)
                    b = b.clone()
                    b[:, 3] = -(j + 1)
                    refs.append(a)
                    refids.append(b)
                samples.append((torch.cat(refs), torch.cat(refids)))
                prompts.append(cap["prompt"])
            texts = [checked_text(text, "", cpu=False)] + [
                checked_text(text, s, cpu=False) for s in prompts
            ]
            current = torch.cat(images)
            for ti in range(len(schedule) - 1):
                t = schedule[ti].to(device)
                delta = float(schedule[ti + 1] - schedule[ti])
                entries = []
                for branch in [0, 1]:
                    for robot in range(3):
                        entries.append(
                            [
                                texts[0] if branch == 0 else texts[robot + 1],
                                samples[robot],
                                (
                                    current[robot * fh * fw : (robot + 1) * fh * fw],
                                    scaled_ids(idlist[robot]),
                                ),
                            ]
                        )
                packed = pack(entries)
                packed.timesteps["image"].fill_(float(t))
                context = SceneContext(
                    states,
                    condition["K"][None].expand(2, -1, -1, -1),
                    condition["c2w"][None].expand(2, -1, -1, -1),
                    (h, w),
                    t,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    velocity, states, _ = model.dit(packed, context)
                unconditional, conditional = velocity.chunk(2)
                current = current + delta * (
                    unconditional.float()
                    + config["inference"]["guidance"]
                    * (conditional.float() - unconditional.float())
                )
            stage = store.staging(index, name)
            frame_rows = []
            for robot, tokens in enumerate(current.chunk(3)):
                image = to_pil(vae.decode(tokens, idlist[robot]))
                relative = store.frame_path(group) / f"robot_{robot:02d}.png"
                dest = stage / f"robot_{robot:02d}.png"
                image.save(dest)
                row = {
                    **group,
                    "robot_id": robot,
                    "seed": seeds[robot],
                    "group_robots": [robot],
                    "path": str(relative),
                    "sha256": file_hash(dest),
                }
                write_json(
                    dest.with_suffix(".generation.json"),
                    {
                        **row,
                        "prompt": prompts[robot],
                        "ground_truth_loaded": False,
                        "world_seed": int(
                            digest([seed, group, "world-appearance"])[:15], 16
                        ),
                    },
                )
                # Export diagnostic predicted geometry, never a GT map.
                import numpy as np
                from .rendering import render
                from .geometry import resize_intrinsics

                sh, sw = h // 4, w // 4
                world = states[1].geometry
                geometry = render(
                    world,
                    resize_intrinsics(
                        condition["K"][robot : robot + 1], (h, w), (sh, sw)
                    ),
                    condition["c2w"][robot : robot + 1],
                    sh,
                    sw,
                    c["world"]["render_backend"],
                    c["world"]["near"],
                    c["world"]["far"],
                )
                labels = model.criterion.semantic(
                    geometry.features.permute(0, 2, 3, 1)
                ).argmax(-1)[0]
                geometry_file = dest.with_suffix(".world_geometry.npz")
                np.savez_compressed(
                    geometry_file,
                    depth=geometry.depth[0, 0].cpu().numpy(),
                    alpha=geometry.alpha[0, 0].cpu().numpy(),
                    semantic=labels.cpu().numpy().astype(np.uint8),
                    K=resize_intrinsics(condition["K"][robot], (h, w), (sh, sw))
                    .cpu()
                    .numpy(),
                    camera_to_local_world=condition["c2w"][robot].cpu().numpy(),
                )
                row["world_geometry_path"] = str(
                    relative.with_suffix(".world_geometry.npz")
                )
                row["world_geometry_sha256"] = file_hash(geometry_file)
                frame_rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
            store.commit(index, group, stage, frame_rows)
            print(
                f"rank={acc.process_index} generated synchronous frame={index + 1}/{len(data.groups)}",
                flush=True,
            )
    acc.wait_for_everyone()
    if acc.is_main_process:
        store.finish(data.groups, name)
    acc.wait_for_everyone()
    if lock is not None:
        lock.close()
