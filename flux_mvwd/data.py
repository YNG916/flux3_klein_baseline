"""Read only permitted Level 1 conditions; targets are a separate method."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import ADAPTER_VERSION
from .common import digest, file_hash, frame_selection, read_json, within


def resize_bev(rgb, world_to_pixel, size):
    h, w = rgb.shape[:2]
    side = max(h, w)
    left, top = (side - w) // 2, (side - h) // 2
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(Image.fromarray(rgb), (left, top))
    scale = size / side
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = scale
    affine[0, 3] = scale * (left + 0.5) - 0.5
    affine[1, 3] = scale * (top + 0.5) - 0.5
    return canvas.resize((size, size), Image.Resampling.BOX), affine @ world_to_pixel


def check_pose(pose):
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Invalid pose")
    if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4):
        raise ValueError("Pose is not a rigid transform")


class Level1Data:
    def __init__(self, config, split, *, scenes=None, max_episodes=None, selection=None):
        self.config = config
        self.root = Path(config["dataset_root"]).expanduser().resolve()
        self.meta = read_json(self.root / "dataset.json")
        if split not in ("train", "val", "test"):
            raise ValueError("Use pinned train, val, test")
        self.split = split
        relative = self.meta["splits"][split]["path"]
        path = within(self.root, relative)
        self.split_hash = file_hash(path)
        if self.split_hash != self.meta["checksums"][relative]:
            raise ValueError("Split checksum mismatch")
        records = [__import__("json").loads(s) for s in path.read_text().splitlines() if s.strip()]
        if len(records) != self.meta["splits"][split]["count"] or any(r["split"] != split for r in records):
            raise ValueError("Split count/assignment mismatch")
        self.selection_hash = None
        if selection is not None:
            if scenes or max_episodes is not None:
                raise ValueError("Use selection alone, without scene/max_episodes overrides")
            from .selection import selected_records
            manifest = read_json(selection)
            records = selected_records(manifest, records, config=config, split=split,
                split_hash=self.split_hash, release=self.meta["release_id"])
            self.selection_hash = manifest["selection_sha256"]
        if scenes:
            if set(scenes) - {r["scene"] for r in records}:
                raise ValueError("Selected scene is absent from split")
            records = [r for r in records if r["scene"] in scenes]
        if max_episodes is not None:
            if max_episodes < 1:
                raise ValueError("max_episodes must be positive")
            records = records[:max_episodes]
        if not records:
            raise ValueError("Empty selection")
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.assets_root = Path(config.get("robot_assets_root") or self.root / "robot_assets/MVWD_RobotAssets_v1_gt_aligned").resolve()
        self.asset_manifest = read_json(self.assets_root / "manifest.json")
        if self.asset_manifest["meters_per_unit"] != 1 or self.asset_manifest["up_axis"] != "Z":
            raise ValueError("Robot assets must be metric Z-up")
        self.appearances = self.asset_file("appearances.json")
        self.mounts = self.asset_file("camera_mount.json")
        parts = self.asset_file("geometry/parts.json")["parts"]
        self.part_names = sorted({p["prim_path"] for p in parts})
        for r in records:
            if r["provenance"]["robot_fingerprint"] != self.asset_manifest["template_fingerprint"]:
                raise ValueError("Robot template differs from dataset")
        self._observations = OrderedDict()
        self._bev = OrderedDict()
        self._rgb = OrderedDict()

    def asset_file(self, name):
        path = within(self.assets_root, name)
        receipt = self.asset_manifest["files"][name]
        if path.stat().st_size != receipt["size"] or file_hash(path) != receipt["sha256"]:
            raise ValueError("Robot metadata checksum mismatch: " + name)
        return read_json(path)

    @staticmethod
    def remember(cache, key, value, limit=2):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def queries(self):
        for record in self.records:
            frames = frame_selection(record["frames"], self.config["frames_per_episode"], record["id"], self.config["frame_sample_seed"])
            for frame in frames:
                groups = [self.config["robots"]] if self.config["mode"] == "joint" else [[r] for r in self.config["robots"]]
                for robots in groups:
                    yield {"episode_id": record["id"], "frame_index": frame, "robots": robots}

    def provenance(self):
        return {**({"selection_sha256": self.selection_hash} if self.selection_hash else {}),
                "release_id": self.meta["release_id"], "split": self.split,
                "split_sha256": self.split_hash, "episode_ids": [r["id"] for r in self.records],
                "robot_assets": {"manifest_sha256": file_hash(self.assets_root / "manifest.json"),
                                  "asset_id": self.asset_manifest["asset_id"]},
                "adapter": ADAPTER_VERSION}

    def conditions(self, query):
        record = self.by_id[query["episode_id"]]
        frame, robots = query["frame_index"], query["robots"]
        if not 0 <= frame < record["frames"] or not robots or set(robots) - set(self.config["robots"]):
            raise ValueError("Invalid query")
        episode = within(self.root, record["episode_path"])
        key = record["id"]
        if key not in self._observations:
            obs = read_json(episode / "observations_before.json")
            self.remember(self._observations, key, {(o["robot_id"], o["physical_time_index"]): o["camera"] for o in obs})
        observations = self._observations[key]
        floor = int(read_json(episode / "generation_metrics.json")["trajectory"]["floor_index"])
        bev_key = (record["configuration_path"], floor)
        if bev_key not in self._bev:
            prefix = f"floor_{floor:02d}/"
            with np.load(within(self.root, record["configuration_path"]) / "bev/environment_base.npz", allow_pickle=False) as packet:
                rgb = packet[prefix + "rgb"]
                w2p = packet[prefix + "calibration_world_to_pixel"]
                floor_z = float(packet[prefix + "calibration_floor_z"])
            image, w2p = resize_bev(rgb, w2p, self.config["bev_size"])
            self.remember(self._bev, bev_key, (image, w2p, floor_z))
        clean, w2p, floor_z = self._bev[bev_key]
        origin = np.linalg.inv(w2p) @ np.array([(clean.width - 1) / 2, (clean.height - 1) / 2, floor_z, 1])
        origin[2] = floor_z
        annotated = clean.copy()
        draw = ImageDraw.Draw(annotated)
        # Grid is derived from calibration, never GT occupancy or semantic maps.
        p2w = np.linalg.inv(w2p)
        corners = np.array([p2w @ [u, v, floor_z, 1] for u in (0, clean.width - 1) for v in (0, clean.height - 1)])
        for axis in (0, 1):
            for position in range(int(np.ceil(corners[:, axis].min())), int(np.floor(corners[:, axis].max())) + 1):
                a, b = corners.min(axis=0), corners.max(axis=0)
                a[axis] = b[axis] = position
                pa, pb = w2p @ a, w2p @ b
                draw.line([tuple(pa[:2]), tuple(pb[:2])], fill=(190, 190, 190), width=1)
        with np.load(episode / "trajectories.npz", allow_pickle=False) as packet:
            poses = [packet[f"robot_{r:02d}_base_to_world"][frame].astype(float) for r in range(record["robots"])]
        state = []
        cameras = []
        h, w = self.config["target_hw"]
        for r, pose in enumerate(poses):
            check_pose(pose)
            camera = observations[(f"robot_{r:02d}", frame)]
            height = float(camera["camera_height_m"])
            nominal = min(map(float, self.mounts["mounts_by_height_m"]), key=lambda x: abs(x - height))
            if abs(height - nominal) > 0.001:
                raise ValueError("Unsupported camera mast height")
            color = self.appearances["robots"][f"robot_{r:02d}"]
            point = w2p @ pose[:, 3]
            end = w2p @ (pose @ np.array([0.7, 0, 0, 1]))
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=tuple(color["rgb8"]), outline="black", width=1)
            draw.line([tuple(point[:2]), tuple(end[:2])], fill=tuple(color["rgb8"]), width=3)
            draw.text((point[0] + 6, point[1] - 6), str(r), fill="black", stroke_width=1, stroke_fill="white")
            local_pose = pose.copy()
            local_pose[:3, 3] -= origin[:3]
            state.append({"robot_id": r, "base_to_local_world": local_pose.tolist(),
                          "camera_height_m": height, "appearance": color["display_name"]})
            if r in robots:
                T = np.asarray(camera.get("modality_camera_to_world", {}).get("rgb", camera["camera_to_world"]), dtype=float)
                check_pose(T)
                K = np.diag([w / camera["width"], h / camera["height"], 1]) @ np.asarray(camera["pixel_intrinsics"], dtype=float)
                T_local = T.copy()
                T_local[:3, 3] -= origin[:3]
                cameras.append({"robot_id": r, "K": K.tolist(), "camera_to_local_world": T_local.tolist(),
                                "camera_to_world": T.tolist(), "near_m": camera["near_m"], "far_m": camera["far_m"]})
        cameras = [next(c for c in cameras if c["robot_id"] == r) for r in robots]
        local_to_world = np.eye(4)
        local_to_world[:3, 3] = origin[:3]
        payload = {"world_axes": "metric Z-up; camera +X right +Y down +Z forward; column vectors",
                   "pixel_centers": "u+0.5,v+0.5 for K; integer centers for BEV",
                   "local_origin_world_m": origin[:3].tolist(),
                   "local_world_to_bev_pixel": (w2p @ local_to_world).tolist(),
                   "target_view_hw": [h, w], "robot_states": state,
                   "cameras": [{k: v for k, v in c.items() if k != "camera_to_world"} for c in cameras],
                   "asset": {"asset_id": self.asset_manifest["asset_id"], "parts": self.part_names,
                             "conditioning": "metadata only; meshes are not rendered in this variant"}}
        layout = "one perspective RGB image" if len(robots) == 1 else "one horizontal strip of three perspective RGB panels, left-to-right robots 0,1,2"
        import json
        prompt = (f"Generate {layout} of this same environment, following every specified camera. "
                  "Image 1 is the clean orthographic environment BEV. Image 2 shows the same BEV with a one-meter grid, "
                  "robot identity labels and base-forward direction lines. These annotations are not scene objects. "
                  "Preserve the scene layout and robot identities, render perspective and occlusions, and include robot bodies wherever visible. "
                  "Do not draw annotations, text, borders or grids in the output. Conditions:\n" + json.dumps(payload, separators=(",", ":")))
        condition = {"images": [clean.copy(), annotated], "prompt": prompt, "query": query,
                     "cameras": cameras, "world_to_bev": w2p, "payload": payload}
        from .camera_conditioning import apply_camera_cues
        return apply_camera_cues(condition, self.config.get("camera_conditioning", "matrix_text"))

    def target(self, query):
        record = self.by_id[query["episode_id"]]
        episode = within(self.root, record["episode_path"])
        h, w = self.config["target_hw"]
        images = []
        for robot in query["robots"]:
            path = episode / "robot_views/before" / f"robot_{robot:02d}.npz"
            if str(path) not in self._rgb:
                with np.load(path, allow_pickle=False) as packet:
                    # The fixed protocol visits three observers twice per episode.
                    # Keep one video per observer so the second frame does not
                    # re-inflate the same compressed 60-frame NPZ video.
                    self.remember(self._rgb, str(path), packet["rgb"], limit=len(self.config["robots"]))
            images.append(Image.fromarray(self._rgb[str(path)][query["frame_index"]]).resize((w, h), Image.Resampling.BOX))
        canvas = Image.new("RGB", (w * len(images), h))
        for i, image in enumerate(images):
            canvas.paste(image, (i * w, 0))
        return canvas

    def cache_identity(self):
        from .common import file_hash
        return digest({"dataset": self.provenance(), "config": self.config,
                       "implementation": file_hash(__file__)})
