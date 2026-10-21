from pathlib import Path
import json
import numpy as np
from .common import digest, file_hash, sample_seed, within, write_json
from .data import Level1Data

def evaluate(config, args):
    """Reuse existing metric mathematics; validate FLUX's own manifest schema."""
    import sys
    from PIL import Image
    from .common import frame_selection, read_json
    sys.path.insert(0, str(Path(args.metrics_repo).resolve()))
    from mvwd_metrics import Level1Evaluator
    from mvwd_metrics.io import load_gt_episode, json_safe
    from adapters.mvwd_raw import MVWDRaw
    from mvwd_metrics.geometry import gt_surface_correspondences
    from itertools import combinations
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    all_rows, metas, identity = {}, [], None
    for directory in map(Path, args.predictions):
        meta = read_json(directory / "metadata.json")
        if meta["schema"] != "flux3-mvwd-predictions-v1" or not meta["complete"] or meta["ground_truth_loaded"]:
            raise ValueError("Incomplete or non-GT-free prediction set")
        if digest(meta["config"]) != digest(config):
            raise ValueError("Prediction config mismatch")
        signature = (meta["lora_sha256"], digest(meta["dataset"]["robot_assets"]), meta["dataset"]["release_id"], digest(meta["versions"]))
        if identity is None:
            identity = signature
        if identity != signature:
            raise ValueError("Mixed model/asset/environment provenance")
        seed = meta["base_seed"]
        if seed in all_rows:
            raise ValueError("Duplicate base seed")
        rows = {}
        for line in (directory / "predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["episode_id"], row["robot_id"], row["frame_index"])
            if key in rows:
                raise ValueError("Duplicate prediction")
            robots = [key[1]] if config["mode"] == "independent" else [0, 1, 2]
            if row["group_robots"] != robots or row["seed"] != sample_seed(seed, meta["dataset"]["release_id"], key[0], key[2], robots, config["mode"]):
                raise ValueError("Seed/group policy mismatch")
            rows[key] = within(directory, row["path"])
            if row.get("sha256") and file_hash(rows[key]) != row["sha256"]:
                raise ValueError("Prediction checksum mismatch")
        if len(rows) != meta["view_count"]:
            raise ValueError("Manifest count mismatch")
        all_rows[seed] = rows
        metas.append(meta)
    seeds = sorted(all_rows)
    if not seeds or any(set(all_rows[s]) != set(all_rows[seeds[0]]) for s in seeds):
        raise ValueError("Seed supports differ")
    split = metas[0]["dataset"]["split"]
    data = Level1Data(config, split)
    expected = {(q["episode_id"], robot, q["frame_index"]) for q in data.queries() for robot in q["robots"]}
    keys = set(all_rows[seeds[0]])
    if any(m["dataset"]["split"] != split or m["dataset"]["split_sha256"] != data.split_hash or m["dataset"]["release_id"] != data.meta["release_id"] or m["dataset"]["robot_assets"] != data.provenance()["robot_assets"] for m in metas):
        raise ValueError("Evaluation requires matching pinned split/assets/release")
    official = split == "test" and seeds == [42, 43, 44] and keys == expected and config["target_hw"] == [256, 448] and config["frames_per_episode"] == 2 and config["frame_sample_seed"] == 42 and config["robots"] == [0, 1, 2]
    if not args.preview and not official:
        raise ValueError("Formal evaluation requires full test, fixed 2 frames, 3 robots, 3 seeds, 256x448")
    if not keys <= expected:
        raise ValueError("Queries differ from fixed frame protocol")
    evaluator = Level1Evaluator(seeds=seeds, metrics=args.metrics, device=args.device, weights_dir=args.weights_dir)
    raw = MVWDRaw(config["dataset_root"], split)
    by_episode = {}
    for e, r, t in sorted(keys):
        by_episode.setdefault(e, []).append((r, t))
    geometric = "mv_gtwarp_dino" in args.metrics
    views_file = (output / "per_query.jsonl").open("x")
    pairs_file = (output / "per_pair.jsonl").open("x")
    try:
        for record in raw.records:
            episode = record["id"]
            if episode not in by_episode:
                continue
            units = by_episode[episode]
            # Existing reader handles actual RGB/depth camera registration.
            gt_images, cameras = load_gt_episode(raw, record, units, tuple(config["target_hw"]), geometry=geometric)
            gt = np.stack([gt_images[unit] for unit in units])
            for seed in seeds:
                predicted = []
                for r, t in units:
                    with Image.open(all_rows[seed][(episode, r, t)]) as im:
                        if im.mode != "RGB" or (im.height, im.width) != tuple(config["target_hw"]):
                            raise ValueError("RGB dimensions differ from protocol")
                        predicted.append(np.asarray(im).transpose(2, 0, 1).astype(np.float32) / 255)
                predicted = np.stack(predicted)
                for row in evaluator.update_views(episode, seed, units, predicted, gt):
                    views_file.write(json.dumps(json_safe(row)) + "\n")
                if geometric:
                    for a, b in combinations(range(len(units)), 2):
                        if units[a][1] != units[b][1] or units[a][0] == units[b][0]:
                            continue
                        correspondences = gt_surface_correspondences(cameras[units[a]], cameras[units[b]])
                        if correspondences["ratio"] <= 0.1:
                            continue
                        backend = evaluator.backend("mv_gtwarp_dino")
                        features = backend.features(predicted[[a, b]])
                        row = evaluator.update_gtwarp_pair(episode, seed, (units[a], units[b]), correspondences,
                            features=(features[0], features[1]))
                        pairs_file.write(json.dumps(json_safe(row)) + "\n")
    finally:
        views_file.close()
        pairs_file.close()
    result = evaluator.finish()
    result.update(formal_coverage=official, preview=args.preview,
        benchmark_ready=official and not args.preview and set(args.metrics) == {"psnr", "lpips", "fid", "mv_gtwarp_dino"}
            and all(v.get("mean") is not None for v in result["metrics"].values()),
        ground_truth_used_only_for_evaluation=True, prediction_metadata=metas)
    result["metric_source_sha256"] = {str(p.relative_to(Path(args.metrics_repo))): file_hash(p)
        for p in sorted((Path(args.metrics_repo) / "mvwd_metrics").glob("*.py"))}
    write_json(output / "metrics.json", json_safe(result))
    lines = ["# Metrics", "", f"Formal coverage: {official}; preview: {args.preview}", "",
             "| Metric | Mean | Std |", "|---|---:|---:|"]
    for name, values in result["metrics"].items():
        lines.append(f"| {name} | {values.get('mean')} | {values.get('std', '')} |")
    (output / "METRICS.md").write_text("\n".join(lines) + "\n")
