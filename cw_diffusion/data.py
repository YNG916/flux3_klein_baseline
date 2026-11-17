"""Separate whitelisted condition access from train/val-only privileged supervision."""

import numpy as np
import torch
from PIL import Image
from flux_mvwd.common import read_json, within, frame_selection, file_hash
from flux_mvwd.data import Level1Data


class CanonicalDataset:
    def __init__(self, baseline_config, split, *, max_episodes=None):
        if baseline_config["mode"] != "independent":
            raise ValueError(
                "Reuse independent captions with three separate native view streams"
            )
        self.data = Level1Data(baseline_config, split, max_episodes=max_episodes)
        self.config = baseline_config
        self.split = split
        self.root = self.data.root
        if (
            file_hash(self.root / "classes.json")
            != self.data.meta["checksums"]["classes.json"]
        ):
            raise ValueError("Semantic vocabulary checksum mismatch")
        classes = read_json(self.root / "classes.json")
        self.classes = classes
        self.category_to_id = {x["name"]: x["class_id"] for x in classes["classes"]}
        self.raw_to_dense = {
            int(i): x["class_id"]
            for x in classes["classes"]
            for i in x["public_semantic_ids"]
        }
        self._geometry_cache = {}
        self.groups = [
            {"episode_id": r["id"], "frame_index": f}
            for r in self.data.records
            for f in frame_selection(
                r["frames"],
                baseline_config["frames_per_episode"],
                r["id"],
                baseline_config["frame_sample_seed"],
            )
        ]

    def dense(self, array):
        keys, inv = np.unique(array, return_inverse=True)
        missing = set(map(int, keys)) - set(self.raw_to_dense)
        if missing:
            raise ValueError("Unknown semantic IDs: " + str(missing))
        return np.asarray([self.raw_to_dense[int(k)] for k in keys], np.int64)[
            inv
        ].reshape(array.shape)

    def conditions(self, index):
        """Only RGB BEV, before camera calibration, and authorized robot states."""
        group = self.groups[index]
        conds = [self.data.conditions({**group, "robots": [r]}) for r in [0, 1, 2]]
        c = conds[0]
        payload = c["payload"]
        states = []
        for robot in payload["robot_states"]:
            t = np.asarray(robot["base_to_local_world"], np.float32)
            height = float(robot["camera_height_m"])
            # Fixed rough support from known robot size; not scene-object GT geometry.
            center = t[:3, 3].copy()
            center[2] += height / 2
            states.append(
                np.concatenate((center, t[:3, :3].reshape(-1), [0.55, 0.55, height]))
            )
        return {
            "group": group,
            "bev": torch.from_numpy(np.asarray(c["images"][0]).copy())
            .permute(2, 0, 1)
            .float()
            / 255,
            "world_to_bev": torch.tensor(
                payload["local_world_to_bev_pixel"], dtype=torch.float32
            ),
            "robot_states": torch.tensor(np.stack(states), dtype=torch.float32),
            "K": torch.tensor(
                np.stack([z["cameras"][0]["K"] for z in conds]), dtype=torch.float32
            ),
            "c2w": torch.tensor(
                np.stack([z["cameras"][0]["camera_to_local_world"] for z in conds]),
                dtype=torch.float32,
            ),
            "origin": torch.tensor(
                payload["local_origin_world_m"], dtype=torch.float32
            ),
            "native_conditions": conds,
            "ground_truth_loaded": False,
        }

    def supervision(self, index):
        if self.split not in ("train", "val"):
            raise ValueError(
                "Test supervision is unavailable to preparation/training; use independent post-generation evaluator"
            )
        condition = self.conditions(index)
        group = condition["group"]
        r = self.data.by_id[group["episode_id"]]
        ep = within(self.root, r["episode_path"])
        frame = group["frame_index"]
        origin = condition["origin"].numpy()
        floor = int(
            read_json(ep / "generation_metrics.json")["trajectory"]["floor_index"]
        )
        prefix = f"floor_{floor:02d}/"
        config = within(self.root, r["configuration_path"])
        with np.load(config / "bev/environment_base.npz", allow_pickle=False) as a:
            bev = {
                k: a[prefix + k]
                for k in ["semantic", "instance", "height", "occupancy"]
            }
        # Nearest labels preserve integer instance identity; validity resized independently.
        side = self.config["bev_size"]
        hh, ww = bev["semantic"].shape
        size = max(hh, ww)
        target = {}
        for name, array in bev.items():
            valid = np.isfinite(array)
            array = np.where(valid, array, 0)
            if name == "semantic":
                array = self.dense(array)
            canvas = np.zeros((size, size), array.dtype)
            y = (size - hh) // 2
            x = (size - ww) // 2
            canvas[y : y + hh, x : x + ww] = array
            if name == "height":
                canvas = canvas - float(origin[2])
                finite = np.zeros((size, size), np.uint8)
                finite[y : y + hh, x : x + ww] = valid
                target["bev_height_valid"] = torch.from_numpy(
                    np.asarray(
                        Image.fromarray(finite).resize(
                            (side, side), Image.Resampling.NEAREST
                        )
                    ).copy()
                ).bool()
            result = np.asarray(
                Image.fromarray(
                    canvas.astype(np.float32)
                    if name == "height"
                    else canvas.astype(np.int32)
                ).resize((side, side), Image.Resampling.NEAREST)
            ).copy()
            target["bev_" + name] = torch.from_numpy(
                result.astype(
                    np.float32 if name in ("height", "occupancy") else np.int64
                )
            )
        obs = {
            (x["robot_id"], x["physical_time_index"]): x["camera"]
            for x in read_json(ep / "observations_before.json")
        }
        modalities = {n: [] for n in ["depth_linear", "semantic", "instance"]}
        poses = {n: [] for n in modalities}
        geometry_K = []
        for robot in [0, 1, 2]:
            cache_key = (r["id"], robot)
            frames = frame_selection(
                r["frames"],
                self.config["frames_per_episode"],
                r["id"],
                self.config["frame_sample_seed"],
            )
            if cache_key not in self._geometry_cache:
                self._geometry_cache = {
                    k: v for k, v in self._geometry_cache.items() if k[0] == r["id"]
                }
                with np.load(
                    ep / f"robot_views/before/robot_{robot:02d}.npz", allow_pickle=False
                ) as a:
                    self._geometry_cache[cache_key] = {
                        name: a[name][frames].copy() for name in modalities
                    }
            for name in modalities:
                array = self._geometry_cache[cache_key][name][frames.index(frame)]
                if name == "semantic":
                    array = self.dense(array)
                modalities[name].append(torch.from_numpy(array.copy()))
            cam = obs[(f"robot_{robot:02d}", frame)]
            geometry_K.append(cam["geometry_pixel_intrinsics"])
            for name in modalities:
                pose = np.asarray(
                    cam.get("modality_camera_to_world", {}).get(
                        name, cam["camera_to_world"]
                    ),
                    np.float32,
                ).copy()
                pose[:3, 3] -= origin
                poses[name].append(torch.from_numpy(pose))
        target.update(
            {
                k: torch.stack(v).float()
                if k in ("depth_linear", "normal")
                else torch.stack(v).long()
                for k, v in modalities.items()
            }
        )
        target["modality_c2w"] = {k: torch.stack(v) for k, v in poses.items()}
        target["geometry_K"] = torch.tensor(np.asarray(geometry_K), dtype=torch.float32)
        # Stable instance labels -> object-state IDs are scene-scoped, never global integer IDs.
        taxonomy = read_json(within(self.root, r["taxonomy_path"]))
        catalog = taxonomy["instance_catalogs"][r["scene"]]
        catalog = {x["object_state_id"]: x for x in catalog if x.get("object_state_id")}
        state = read_json(ep / "state_before.json")
        objects = []
        supported = set(np.unique(bev["instance"]).tolist()) | set(
            target["instance"].unique().tolist()
        )
        for ob in state["objects"]:
            item = catalog.get(ob["instance_id"])
            if item is None or item["public_instance_id"] not in supported:
                continue
            lo = np.asarray(ob["bbox_min_world"], np.float32) - origin
            hi = np.asarray(ob["bbox_max_world"], np.float32) - origin
            pose = np.asarray(ob["object_to_world"], np.float32)
            yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
            objects.append(
                {
                    "public_instance_id": item["public_instance_id"],
                    "category": self.category_to_id[ob["category"]],
                    "center": (lo + hi) / 2,
                    "aabb_size": hi - lo,
                    "yaw": yaw,
                    "rotation": pose[:3, :3].copy(),
                    "object_state_id": ob["instance_id"],
                }
            )
        target["objects"] = objects
        target["rgb"] = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(self.data.target({**group, "robots": [robot]})).copy()
                )
                .permute(2, 0, 1)
                .float()
                / 255
                for robot in [0, 1, 2]
            ]
        )
        target["ground_truth_loaded"] = True
        return target
