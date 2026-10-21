"""CPU-only GT/BEV exports after completed generation; prediction pixels are immutable."""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw

from flux_mvwd.common import digest, file_hash, frame_selection, read_json, sample_seed, within, write_json
from flux_mvwd.data import Level1Data

SCHEMA = "flux3-mvwd-visualizations-v2"


def save_image(path, image):
    """Resume partial exports without overwriting preexisting or legacy images."""
    if path.exists():
        with Image.open(path) as old:
            if old.mode != image.mode or old.size != image.size or old.tobytes() != image.tobytes():
                raise FileExistsError(f"Existing image differs; refusing to overwrite: {path}")
    else:
        image.save(path)


def save_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"Existing metadata differs: {path}")
    else:
        write_json(path, value)


def panels(images, labels, *, height):
    resized = [im.resize((round(im.width * height / im.height), height), Image.Resampling.BOX)
               if im.height != height else im for im in images]
    canvas = Image.new("RGB", (sum(im.width for im in resized), height + 24), "white")
    draw, x = ImageDraw.Draw(canvas), 0
    for label, im in zip(labels, resized):
        draw.text((x + 4, 4), label, fill="black")
        canvas.paste(im, (x, 24)); x += im.width
    return canvas


def verify_exports(directory, records, receipt):
    for record in records:
        if file_hash(within(directory, record["prediction_path"])) != record["prediction_sha256"]:
            raise ValueError("Original prediction changed since visualization export")
        for name, checksum in record["files"].items():
            if file_hash(within(directory, name)) != checksum:
                raise ValueError(f"Visualization checksum mismatch: {name}")
    for name, checksum in receipt.get("group_files", {}).items():
        if file_hash(within(directory, name)) != checksum:
            raise ValueError(f"Group comparison checksum mismatch: {name}")
    if file_hash(directory / "index.html") != receipt["gallery_sha256"]:
        raise ValueError("Visualization gallery checksum mismatch")


def export_comparisons(predictions):
    directory = Path(predictions).resolve()
    meta_path, rows_path = directory / "metadata.json", directory / "predictions.jsonl"
    meta = read_json(meta_path)
    if meta.get("schema") != "flux3-mvwd-predictions-v1" or meta.get("complete") is not True or meta.get("ground_truth_loaded") is not False:
        raise ValueError("Require completed GT-free generation before reading any GT")
    import json
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if not rows or len(rows) != meta["view_count"]:
        raise ValueError("Prediction manifest count mismatch")
    keys = [(r["episode_id"], r["frame_index"], r["robot_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate prediction query")
    identity = {"prediction_metadata_sha256": file_hash(meta_path), "prediction_manifest_sha256": file_hash(rows_path)}
    receipt_path, manifest_path = directory / "visualization_receipt.json", directory / "visualizations.jsonl"
    old = read_json(receipt_path) if receipt_path.exists() else None
    if old:
        if old.get("schema") != SCHEMA or any(old.get(k) != v for k, v in identity.items()):
            raise ValueError("Visualization receipt belongs to different predictions")
        if old.get("complete"):
            records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            if len(records) != len(rows) or file_hash(manifest_path) != old["visualization_manifest_sha256"]:
                raise ValueError("Visualization manifest mismatch")
            verify_exports(directory, records, old)
            print(f"Verified existing GT/BEV exports for {len(rows)} views: {directory / 'index.html'}")
            return old
    elif manifest_path.exists():
        raise ValueError("Untracked visualization manifest; inspect before retrying")

    # Match the conditioning source used by the original run before redrawing its BEV.
    experiment_manifest = directory.parent.parent / "manifest.json"
    conditioning_hashes = {}
    if experiment_manifest.exists():
        sources = read_json(experiment_manifest).get("source_sha256", {})
        repo = Path(__file__).resolve().parents[1]
        for name in ("flux_mvwd/data.py", "flux_mvwd/camera_conditioning.py", "flux_mvwd/common.py", "flux_mvwd/__init__.py"):
            if name in sources:
                if file_hash(repo / name) != sources[name]:
                    raise ValueError("Original conditioning implementation differs; use its matching source checkout for BEV export")
                conditioning_hashes[name] = sources[name]
    data = Level1Data(meta["config"], meta["dataset"]["split"])
    provenance = data.provenance()
    if digest(meta["config"]) != digest(data.config) or any(meta["dataset"].get(k) != provenance.get(k)
            for k in ("release_id", "split_sha256", "robot_assets")):
        raise ValueError("Prediction dataset release/split/assets differ from current dataset")
    # Validate all identities before the first GT read.
    for row in rows:
        record = data.by_id[row["episode_id"]]
        group = row["group_robots"]
        if row["robot_id"] not in group or not set(group) <= set(data.config["robots"]):
            raise ValueError("Prediction robot/group mismatch")
        if group != ([row["robot_id"]] if data.config["mode"] == "independent" else [0, 1, 2]):
            raise ValueError("Prediction group differs from generation mode")
        if row["frame_index"] not in frame_selection(record["frames"], data.config["frames_per_episode"],
                                                       row["episode_id"], data.config["frame_sample_seed"]):
            raise ValueError("Prediction frame differs from fixed query protocol")
        if row["seed"] != sample_seed(meta["base_seed"], data.meta["release_id"], row["episode_id"],
                                      row["frame_index"], group, data.config["mode"]):
            raise ValueError("Prediction seed/group mismatch")
        prediction = within(directory, row["path"])
        if row.get("sha256") and file_hash(prediction) != row["sha256"]:
            raise ValueError("Prediction checksum mismatch before GT access")
    receipt = {"schema": SCHEMA, "complete": False, **identity, "images": len(rows),
        "gt_read_only_after_generation": True, "prediction_pixels_modified": False,
        "renderer_sha256": file_hash(Path(__file__)), "conditioning_source_sha256": conditioning_hashes, "dataset_release": data.meta["release_id"]}
    write_json(receipt_path, receipt)
    records, groups = [], defaultdict(list)
    try:
        # Rebuild the small export manifest on retry; image contents must match exactly.
        with manifest_path.open("w") as stream:
            for row in rows:
                query = {"episode_id": row["episode_id"], "frame_index": row["frame_index"], "robots": row["group_robots"]}
                condition = data.conditions(query)
                target = data.target({**query, "robots": [row["robot_id"]]})
                path = within(directory, row["path"])
                with Image.open(path) as image:
                    predicted = image.convert("RGB")
                if predicted.size != target.size:
                    raise ValueError("Prediction/GT full-FoV dimensions differ")
                episode_dir = directory / row["episode_id"]
                if not path.is_relative_to(episode_dir):
                    raise ValueError("Prediction path does not match its episode")
                exports = {
                    episode_dir / "env_bev.png": condition["images"][0],
                    path.with_suffix(".gt.png"): target,
                    path.with_suffix(".camera_bev.png"): condition["images"][1],
                    path.with_suffix(".comparison.png"): panels([target, predicted],
                        ["GT", "FLUX prediction (diagnostic)"], height=target.height),
                    path.with_suffix(".bev_comparison.png"): panels([*condition["images"], target, predicted],
                        ["Global BEV", "Camera/robot condition BEV", "GT", "Prediction"], height=target.height),
                }
                for name, image in exports.items():
                    save_image(name, image)
                calibration = episode_dir / "bev_calibration.json"
                save_json(calibration, {"world_to_bev_pixel": condition["world_to_bev"].tolist(),
                    "bev_hw": [condition["images"][0].height, condition["images"][0].width],
                    "pixel_centers": "BEV integer centers; world metric Z-up"})
                camera_file = path.with_suffix(".camera.json")
                save_json(camera_file, {"query": query, "output_robot": row["robot_id"],
                    "cameras": condition["cameras"], "conditions": condition["payload"],
                    "prompt_sha256": digest(condition["prompt"])})
                names = [*exports, calibration, camera_file]
                item = {"query": query, "robot_id": row["robot_id"], "prediction_path": row["path"],
                    "prediction_sha256": file_hash(path),
                    "files": {str(f.relative_to(directory)): file_hash(f) for f in names}}
                stream.write(json.dumps(item) + "\n"); stream.flush(); records.append(item)
                groups[(row["episode_id"], row["frame_index"])].append(row)
                print(f"Exported GT/BEV comparisons {len(records)}/{len(rows)}", flush=True)
        group_files, gallery = {}, []
        h, w = data.config["target_hw"]
        for (episode, frame), members in sorted(groups.items()):
            members.sort(key=lambda r: r["robot_id"])
            canvas = Image.new("RGB", (len(members) * w, 3 * (h + 24)), "white")
            draw = ImageDraw.Draw(canvas)
            for column, row in enumerate(members):
                original = within(directory, row["path"])
                sources = [original.with_suffix(".camera_bev.png"), original.with_suffix(".gt.png"), original]
                for index, (label, source) in enumerate(zip(("Condition BEV", "GT", "Prediction"), sources)):
                    with Image.open(source) as im:
                        im = im.convert("RGB"); im.thumbnail((w, h), Image.Resampling.BOX)
                        x, y = column * w, index * (h + 24)
                        draw.text((x + 4, y + 4), f"{label} / robot {row['robot_id']}", fill="black")
                        canvas.paste(im, (x + (w - im.width) // 2, y + 24))
            group_path = Path(episode) / f"frame_{frame:03d}.multiview_comparison.png"
            save_image(directory / group_path, canvas)
            group_files[str(group_path)] = file_hash(directory / group_path)
            title = html.escape(f"{data.by_id[episode]['scene']} / {episode} / frame {frame}")
            gallery.append(f'<section><h2>{title}</h2><a href="{quote(str(group_path))}"><img loading="lazy" src="{quote(str(group_path))}" alt="BEV, GT and predictions"></a></section>')
        page = '<!doctype html><html><meta charset="utf-8"><title>BEV / GT / prediction review</title><style>body{font:14px sans-serif;margin:24px}img{max-width:100%;height:auto}h2{font-size:15px}section{margin:32px 0}</style>'
        page += f'<h1>BEV / GT / prediction review: {len(rows)} views</h1><p>GT read only after generation. Rows: actual condition BEV, GT, prediction. Each column is one robot camera. Global BEV: each episode/env_bev.png. Model LoRA: {html.escape(str(meta["lora_sha256"]))}</p>' + ''.join(gallery) + '</html>\n'
        (directory / "index.html").write_text(page)
        receipt.update(complete=True, exported_images=len(records), multiview_groups=len(groups),
            group_files=group_files, gallery_sha256=file_hash(directory / "index.html"),
            visualization_manifest_sha256=file_hash(manifest_path))
        write_json(receipt_path, receipt)
    except BaseException as exc:
        receipt.update(exported_images=len(records), error=f"{type(exc).__name__}: {exc}")
        write_json(receipt_path, receipt)
        raise
    print(f"Completed GT/BEV gallery: {directory / 'index.html'}", flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    args = p.parse_args()
    export_comparisons(args.predictions)


if __name__ == "__main__":
    main()
