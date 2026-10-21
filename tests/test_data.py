from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from flux_mvwd.common import file_hash, frame_selection, load_config, sample_seed, write_json
from flux_mvwd.data import Level1Data, resize_bev


def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(__file__).parents[1] / "configs/examples/level1_independent.json").read_text())
    config.update(dataset_root=str(root), bev_size=256, target_hw=[64, 96], camera_conditioning="matrix_text")
    episode, configuration = root / "episode", root / "configuration"
    episode.mkdir(); (configuration / "bev").mkdir(parents=True)
    write_json(episode / "generation_metrics.json", {"trajectory": {"floor_index": 0}})
    # BEV world x maps to u; world y maps to -v. Two meters wide, one high.
    w2p = np.array([[10, 0, 0, 0], [0, -10, 0, 9], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    np.savez(configuration / "bev/environment_base.npz", **{"floor_00/rgb": np.full((10, 20, 3), 100, np.uint8),
        "floor_00/calibration_world_to_pixel": w2p, "floor_00/calibration_floor_z": np.array(0.0)})
    trajectories, observations = {}, []
    (episode / "robot_views/before").mkdir(parents=True)
    for r in range(3):
        base = np.eye(4); base[0, 3] = r * 0.4
        trajectories[f"robot_{r:02d}_base_to_world"] = np.stack([base, base])
        for frame in range(2):
            T = np.eye(4); T[:3, 3] = [r * 0.4, 0.3, 0.8]
            observations.append({"robot_id": f"robot_{r:02d}", "physical_time_index": frame,
                "camera": {"camera_to_world": T.tolist(), "pixel_intrinsics": [[100, 0, 96], [0, 100, 64], [0, 0, 1]],
                    "width": 192, "height": 128, "camera_height_m": 0.8, "near_m": 0.1, "far_m": 15}})
        rgb = np.zeros((2, 128, 192, 3), np.uint8);rgb[..., r] = 255
        np.savez(episode / f"robot_views/before/robot_{r:02d}.npz", rgb=rgb)
    np.savez(episode / "trajectories.npz", **trajectories)
    write_json(episode / "observations_before.json", observations)
    assets = root / "robot_assets/MVWD_RobotAssets_v1_gt_aligned"
    assets.mkdir(parents=True)
    documents = {"appearances.json": {"robots": {f"robot_{r:02d}": {"rgb8": [255 if i == r else 0 for i in range(3)], "display_name": str(r)} for r in range(3)}},
        "camera_mount.json": {"mounts_by_height_m": {"0.8": {}}},
        "geometry/parts.json": {"parts": [{"prim_path": "chassis"}]}}
    for path, value in documents.items():write_json(assets / path, value)
    write_json(assets / "manifest.json", {"asset_id": "fixture", "template_fingerprint": "robot", "meters_per_unit": 1, "up_axis": "Z",
        "files": {p: {"size": (assets / p).stat().st_size, "sha256": file_hash(assets / p)} for p in documents}})
    metadata = {"release_id": "fixture", "splits": {}, "checksums": {}}
    (root / "splits").mkdir()
    for split in ("train", "val", "test"):
        record = {"id": "ep", "scene": "room", "split": split, "episode_path": "episode", "configuration_path": "configuration",
            "frames": 2, "fps": 10, "robots": 3, "provenance": {"robot_fingerprint": "robot"}}
        path = root / f"splits/{split}.jsonl";path.write_text(json.dumps(record) + "\n")
        metadata["splits"][split] = {"path": f"splits/{split}.jsonl", "count": 1}
        metadata["checksums"][f"splits/{split}.jsonl"] = file_hash(path)
    write_json(root / "dataset.json", metadata)
    return config


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_no_target_reads_during_conditioning(self):
        data = Level1Data(self.config, "train")
        original = np.load
        def guarded(path, *args, **kwargs):
            if "robot_views" in str(path) or "after" in str(path) or "world_before" in str(path):
                self.fail("GT observation accessed by condition path")
            return original(path, *args, **kwargs)
        with patch("numpy.load", side_effect=guarded):
            condition = data.conditions(next(data.queries()))
        self.assertEqual(len(condition["payload"]["robot_states"]), 3)
        self.assertNotIn("target", condition)

    def test_joint_panel_identity_and_camera_order(self):
        self.config["mode"] = "joint"
        data = Level1Data(self.config, "train")
        query = next(data.queries())
        target = np.asarray(data.target(query))
        self.assertEqual(target.shape, (64, 288, 3))
        for r in range(3):
            self.assertTrue((target[:, r * 96:(r+1)*96, r] == 255).all())
        self.assertEqual([c["robot_id"] for c in data.conditions(query)["cameras"]], [0, 1, 2])

    def test_intrinsics_scaling_preserves_ray(self):
        data = Level1Data(self.config, "train")
        K = np.array(data.conditions(next(data.queries()))["cameras"][0]["K"])
        original = np.array([[100, 0, 96], [0, 100, 64], [0, 0, 1]])
        np.testing.assert_allclose(np.linalg.solve(K, [24.25, 16.25, 1]), np.linalg.solve(original, [48.5, 32.5, 1]))

    def test_bev_letterbox_pixel_center_roundtrip(self):
        rgb = np.zeros((10, 20, 3), np.uint8)
        _, calibrated = resize_bev(rgb, np.eye(4), 256)
        point = np.array([3., 4., 0., 1.])
        pixel = calibrated @ point
        np.testing.assert_allclose(pixel[:2], [12.8 * 3.5 - 0.5, 12.8 * 9.5 - 0.5])
        np.testing.assert_allclose(np.linalg.inv(calibrated) @ pixel, point)

    def test_local_frame_reprojects_to_same_bev(self):
        data = Level1Data(self.config, "train")
        condition = data.conditions(next(data.queries()))
        payload = condition["payload"]
        robot = np.array(payload["robot_states"][1]["base_to_local_world"])
        local_projection = np.array(payload["local_world_to_bev_pixel"]) @ robot[:, 3]
        world_projection = condition["world_to_bev"] @ np.array([0.4, 0, 0, 1])
        np.testing.assert_allclose(local_projection, world_projection)

    def test_split_tamper_rejected(self):
        path = self.root / "splits/train.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Level1Data(self.config, "train")

    def test_asset_tamper_rejected(self):
        (self.root / "robot_assets/MVWD_RobotAssets_v1_gt_aligned/appearances.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Level1Data(self.config, "train")

    def test_fixed_frames_match_top2pano_policy(self):
        import hashlib
        selected = frame_selection(60, 2, "episode")
        expected = [slot * 30 + int(hashlib.sha256(f"42:episode:frame-stratum:{slot}".encode()).hexdigest()[:16], 16) % 30 for slot in range(2)]
        self.assertEqual(selected, expected)

    def test_seed_group_policy(self):
        joint = sample_seed(42, "release", "episode", 0, [0, 1, 2], "joint")
        independent = [sample_seed(42, "release", "episode", 0, [r], "independent") for r in range(3)]
        self.assertEqual(len(set([joint, *independent])), 4)


if __name__ == "__main__":
    unittest.main()
