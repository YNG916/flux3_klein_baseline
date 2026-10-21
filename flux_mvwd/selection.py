"""Metadata-only, deterministic scene-balanced episode manifests."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .common import digest, read_json, write_json

PROTOCOL_KEYS = ("mode", "robots", "frames_per_episode", "frame_sample_seed")


def make_selection(data, *, episodes_per_scene, seed=42, scene_count=None):
    if data.split == "test":
        raise ValueError("Subset design uses train/val only; keep test for final evaluation")
    if episodes_per_scene < 1 or (scene_count is not None and scene_count < 1):
        raise ValueError("Selection counts must be positive")
    groups = defaultdict(list)
    for record in data.records:
        groups[record["scene"]].append(record)
    scenes = sorted(groups, key=lambda s: (digest([seed, "scene", s]), s))
    if scene_count is not None:
        if scene_count > len(scenes):
            raise ValueError("scene_count exceeds split scenes")
        scenes = scenes[:scene_count]
    for scene in scenes:
        if len(groups[scene]) < episodes_per_scene:
            raise ValueError(f"Scene {scene} has fewer episodes than requested")
        groups[scene].sort(key=lambda r: (digest([seed, "episode", r["id"]]), r["id"]))
    # Round-robin scenes: bounded prefixes also span scenes rather than one scene.
    records = [groups[s][i] for i in range(episodes_per_scene) for s in scenes]
    body = {"schema": "flux3-mvwd-selection-v1", "release_id": data.meta["release_id"],
        "split": data.split, "split_sha256": data.split_hash,
        "query_protocol": {k: data.config[k] for k in PROTOCOL_KEYS},
        "policy": {"name": "scene-balanced-sha256-round-robin-v1", "seed": seed,
                   "episodes_per_scene": episodes_per_scene, "scene_count": len(scenes)},
        "episodes": [{"id": r["id"], "scene": r["scene"]} for r in records]}
    return {**body, "selection_sha256": digest(body)}


def selected_records(manifest, records, *, config, split, split_hash, release):
    body = {k: v for k, v in manifest.items() if k != "selection_sha256"}
    if manifest.get("schema") != "flux3-mvwd-selection-v1" or digest(body) != manifest.get("selection_sha256"):
        raise ValueError("Selection checksum/schema mismatch")
    if (manifest["split"], manifest["split_sha256"], manifest["release_id"]) != (split, split_hash, release):
        raise ValueError("Selection belongs to a different split/release")
    if manifest["query_protocol"] != {k: config[k] for k in PROTOCOL_KEYS}:
        raise ValueError("Selection query protocol differs")
    by_id = {r["id"]: r for r in records}
    ids = [r["id"] for r in manifest["episodes"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Selection must have unique nonempty episodes")
    for row in manifest["episodes"]:
        if row["id"] not in by_id or row["scene"] != by_id[row["id"]]["scene"]:
            raise ValueError("Selection episode/scene absent from pinned split")
    return [by_id[e] for e in ids]


def select(config, args):
    from .data import Level1Data
    data = Level1Data(config, args.split, scenes=args.scene)
    manifest = make_selection(data, episodes_per_scene=args.episodes_per_scene,
                              seed=args.selection_seed, scene_count=args.scene_count)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection.json", manifest)
    queries_per_episode = config["frames_per_episode"] * (len(config["robots"]) if config["mode"] == "independent" else 1)
    write_json(output / "summary.json", {"episodes": len(manifest["episodes"]),
        "queries": len(manifest["episodes"]) * queries_per_episode,
        "views": len(manifest["episodes"]) * config["frames_per_episode"] * len(config["robots"]),
        "episodes_by_scene": dict(Counter(r["scene"] for r in manifest["episodes"])),
        "ground_truth_loaded": False, "selection_sha256": manifest["selection_sha256"]})
    print(output / "selection.json")
