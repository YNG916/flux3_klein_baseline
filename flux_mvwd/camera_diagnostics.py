"""Fixed-noise camera interventions, separate from benchmark predictions.

Virtual camera interventions keep all robot base states unchanged. They do not
simulate robot motion, validate free space, or provide counterfactual GT.
"""
from __future__ import annotations

import copy
import itertools
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import digest, file_hash, read_json, sample_seed, within, write_json
from .data import Level1Data, check_pose


def with_camera(condition, camera):
    """Replace only the target camera, preserving the original prompt template."""
    result = copy.deepcopy(condition)
    check_pose(np.asarray(camera["camera_to_world"]))
    check_pose(np.asarray(camera["camera_to_local_world"]))
    result["cameras"] = [copy.deepcopy(camera)]
    result["payload"]["cameras"] = [{k: v for k, v in camera.items() if k != "camera_to_world"}]
    prefix, separator, encoded = condition["prompt"].partition("Conditions:\n")
    if not separator or json.loads(encoded) != condition["payload"]:
        raise ValueError("Condition prompt does not match its structured payload")
    result["prompt"] = prefix + separator + json.dumps(result["payload"], separators=(",", ":"))
    from .camera_conditioning import apply_camera_cues
    return apply_camera_cues(result, result.get("camera_cue_variant", "matrix_text"))


def camera_cases(data, episode_id, frame):
    """Recorded views followed by interventions on robot 0's query camera."""
    conditions = [data.conditions({"episode_id": episode_id, "frame_index": frame, "robots": [r]}) for r in (0, 1, 2)]
    reference = conditions[0]
    cases = [{"name": f"recorded_{r}", "condition": condition, "gt_robot": r,
              "intervention": {"kind": "recorded", "robot_id": r}}
             for r, condition in enumerate(conditions)]
    cases.append({"name": "repeat_0", "condition": copy.deepcopy(reference), "gt_robot": 0,
                  "intervention": {"kind": "identical_repeat"}})
    for degrees in (30, -30):
        camera = copy.deepcopy(reference["cameras"][0])
        angle = np.deg2rad(degrees)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        for key in ("camera_to_world", "camera_to_local_world"):
            T = np.asarray(camera[key], dtype=float)
            T[:3, :3] = rotation @ T[:3, :3]
            camera[key] = T.tolist()
        cases.append({"name": f"yaw_{'plus' if degrees > 0 else 'minus'}30",
                      "condition": with_camera(reference, camera), "gt_robot": None,
                      "intervention": {"kind": "virtual_camera_world_z_yaw", "degrees": degrees}})
    camera = copy.deepcopy(reference["cameras"][0])
    offset = np.asarray(camera["camera_to_world"])[:3, 0] * 0.5
    for key in ("camera_to_world", "camera_to_local_world"):
        T = np.asarray(camera[key], dtype=float)
        T[:3, 3] += offset
        camera[key] = T.tolist()
    cases.append({"name": "right_0p5m", "condition": with_camera(reference, camera), "gt_robot": None,
                  "intervention": {"kind": "virtual_camera_local_right", "meters": 0.5}})
    camera = copy.deepcopy(reference["cameras"][0])
    K = np.asarray(camera["K"], dtype=float)
    K[0, 0] *= 1.25
    K[1, 1] *= 1.25
    camera["K"] = K.tolist()
    cases.append({"name": "focal_1p25x", "condition": with_camera(reference, camera), "gt_robot": None,
                  "intervention": {"kind": "virtual_camera_focal_scale", "factor": 1.25}})
    for case in cases:
        condition = case["condition"]
        for key in ("robot_states", "asset", "local_world_to_bev_pixel"):
            if condition["payload"][key] != reference["payload"][key]:
                raise ValueError(f"Intervention changed shared {key}")
        actual = [condition["images"][0], condition.get("camera_cue_base_image", condition["images"][1])]
        expected = [reference["images"][0], reference.get("camera_cue_base_image", reference["images"][1])]
        if any(a.tobytes() != b.tobytes() for a, b in zip(actual, expected)):
            raise ValueError("Camera diagnostic must preserve BEV and shared robot annotations")
    return cases


def camera_overlay(condition):
    """Target camera rays for human review only; never supplied to FLUX."""
    if condition.get("camera_cue_variant"):
        return condition["images"][1].copy()
    image = condition["images"][1].copy()
    draw = ImageDraw.Draw(image)
    camera = condition["cameras"][0]
    T = np.asarray(camera["camera_to_world"])
    K = np.asarray(camera["K"])
    w2p = condition["world_to_bev"]
    h, w = condition["payload"]["target_view_hw"]
    origin = T[:, 3]
    start = (w2p @ origin)[:2]
    draw.ellipse((*tuple(start - 4), *tuple(start + 4)), fill="magenta")
    for u in (0.5, w / 2, w - 0.5):
        ray = T[:3, :3] @ np.linalg.solve(K, [u, h / 2, 1])
        norm = np.linalg.norm(ray[:2])
        if norm > 1e-8:
            endpoint = origin.copy()
            endpoint[:2] += ray[:2] / norm * 1.5
            draw.line([tuple(start), tuple((w2p @ endpoint)[:2])], fill="magenta", width=2)
    return image


def plan(config, args):
    """CPU condition interventions; virtual cameras have no counterfactual GT."""
    if config["mode"] != "independent" or config["robots"] != [0, 1, 2]:
        raise ValueError("Camera plan requires independent mode and all robots")
    if args.split not in ("train", "val") or args.max_groups < 1:
        raise ValueError("Use bounded train/val camera diagnostics")
    data = Level1Data(config, args.split, scenes=args.scene, max_episodes=args.max_episodes)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    groups = []
    anchors = itertools.islice((q for q in data.queries() if q["robots"] == [0]), args.max_groups)
    for i, query in enumerate(anchors):
        seed = sample_seed(args.seed, data.meta["release_id"], query["episode_id"], query["frame_index"], [0], "independent")
        cases = []
        for case in camera_cases(data, query["episode_id"], query["frame_index"]):
            condition = case["condition"]
            folder = output / f"group_{i:03d}" / case["name"]; folder.mkdir(parents=True)
            for j, image in enumerate(condition["images"]): image.save(folder / f"condition_{j}.png")
            (folder / "prompt.txt").write_text(condition["prompt"])
            write_json(folder / "geometry.json", condition["payload"])
            cases.append({"name": case["name"], "seed": seed, "prompt_sha256": digest(condition["prompt"]),
                "condition_sha256": [__import__("hashlib").sha256(im.tobytes()).hexdigest() for im in condition["images"]],
                "intervention": case["intervention"]})
        groups.append({"query": query, "cases": cases})
    write_json(output / "metadata.json", {"schema": "flux3-mvwd-camera-plan-v1", "complete": True,
        "ground_truth_loaded": False, "benchmark_ready": False, "groups": groups})
