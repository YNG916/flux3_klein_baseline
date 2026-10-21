"""Calibration-derived camera cues must stay synchronized and GT-free."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_data import fixture
from flux_mvwd.camera_conditioning import apply_camera_cues, camera_summary
from flux_mvwd.camera_diagnostics import camera_cases, plan
from flux_mvwd.common import load_config, read_json, write_json
from flux_mvwd.data import Level1Data


class CameraCueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = fixture(self.root / "dataset")
        # CV camera right=-Y, down=-Z, forward=+X in metric Z-up world.
        path = self.root / "dataset/episode/observations_before.json"
        observations = read_json(path)
        for obs in observations:
            T = np.array(obs["camera"]["camera_to_world"])
            T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            obs["camera"]["camera_to_world"] = T.tolist()
        write_json(path, observations)

    def tearDown(self):
        self.temp.cleanup()

    def cues(self):
        config = copy.deepcopy(self.config)
        config["camera_conditioning"] = "bev_frustum_text_v1"
        return Level1Data(config, "train")

    def test_legacy_default_is_unchanged_and_asset_text_is_preserved(self):
        original = Level1Data(self.config, "train")
        explicit = copy.deepcopy(self.config);explicit["camera_conditioning"] = "matrix_text"
        query = next(original.queries())
        a, b = original.conditions(query), Level1Data(explicit, "train").conditions(query)
        self.assertEqual(a["prompt"], b["prompt"])
        self.assertNotIn("query_camera_summary", a["payload"])
        self.assertIs(apply_camera_cues(a), a)
        new = self.cues().conditions(query)
        for key in ("asset", "robot_states", "cameras", "local_world_to_bev_pixel"):
            self.assertEqual(new["payload"][key], a["payload"][key])
        np.testing.assert_array_equal(new["images"][0], a["images"][0])
        np.testing.assert_array_equal(new["camera_cue_base_image"], a["images"][1])
        self.assertNotEqual(new["images"][1].tobytes(), a["images"][1].tobytes())

    def test_summary_uses_optical_pose_and_border_rays(self):
        data = self.cues();c = data.conditions(next(data.queries()))
        summary = c["payload"]["query_camera_summary"]
        np.testing.assert_allclose(summary["forward_local_unit"], [1, 0, 0])
        self.assertAlmostEqual(summary["heading_deg_from_positive_x_toward_positive_y"], 0)
        self.assertAlmostEqual(summary["pitch_deg_above_horizontal"], 0)
        np.testing.assert_allclose(summary["center_local_m"], np.array(c["cameras"][0]["camera_to_local_world"])[:3, 3])
        # Resized K: fx=fy=50, cx=48, cy=32, full 96x64 view.
        self.assertAlmostEqual(summary["horizontal_fov_deg"], np.rad2deg(2 * np.arctan(47.5/50)))
        self.assertAlmostEqual(summary["vertical_fov_deg"], np.rad2deg(2 * np.arctan(31.5/50)))

    def test_interventions_regenerate_summary_and_image_without_stale_cues(self):
        data = self.cues();cases = {c["name"]:c for c in camera_cases(data, "ep", 0)}
        base = cases["recorded_0"]["condition"]
        for name in ("yaw_plus30", "yaw_minus30", "right_0p5m", "focal_1p25x"):
            c = cases[name]["condition"]
            summary = c["payload"]["query_camera_summary"]
            self.assertEqual(summary, camera_summary(c["cameras"][0], self.config["target_hw"]))
            self.assertEqual(c["prompt"].count("QUERY CAMERA:"), 1)
            self.assertNotEqual(c["images"][1].tobytes(), base["images"][1].tobytes())
            self.assertEqual(c["payload"]["robot_states"], base["payload"]["robot_states"])
            np.testing.assert_array_equal(c["camera_cue_base_image"], base["camera_cue_base_image"])
        self.assertAlmostEqual(cases["yaw_plus30"]["condition"]["payload"]["query_camera_summary"]["heading_deg_from_positive_x_toward_positive_y"],30)
        self.assertLess(cases["focal_1p25x"]["condition"]["payload"]["query_camera_summary"]["horizontal_fov_deg"],base["payload"]["query_camera_summary"]["horizontal_fov_deg"])
        before = base["prompt"], base["images"][1].tobytes()
        apply_camera_cues(base, "bev_frustum_text_v1")
        self.assertEqual(before,(base["prompt"],base["images"][1].tobytes()))

    def test_full_plan_has_shared_scene_but_query_specific_cues_and_no_gt(self):
        config = copy.deepcopy(self.config);config["camera_conditioning"]="bev_frustum_text_v1"
        out = self.root / "plan"
        args = SimpleNamespace(split="train",scene=None,max_episodes=1,max_groups=1,seed=42,
            dry_run=True,lora=None,zero_shot=False,cpu_offload=False,output=str(out))
        with patch.object(Level1Data,"target",side_effect=AssertionError("GT read")):
            plan(config,args)
        cases = read_json(out/'metadata.json')["groups"][0]["cases"]
        self.assertEqual(len({c["condition_sha256"][0] for c in cases}),1)
        self.assertGreater(len({c["condition_sha256"][1] for c in cases}),3)
        self.assertEqual(cases[0]["condition_sha256"],cases[3]["condition_sha256"])
        self.assertEqual(cases[0]["prompt_sha256"],cases[3]["prompt_sha256"])
        self.assertEqual(len({c["seed"] for c in cases}),1)

    def test_invalid_or_joint_variant_is_rejected(self):
        config = copy.deepcopy(self.config);config["camera_conditioning"]="wrong"
        p=self.root/'config.json';write_json(p,config)
        with self.assertRaisesRegex(ValueError,"camera_conditioning"):load_config(p)
        config["camera_conditioning"]="bev_frustum_text_v1";config["mode"]="joint";write_json(p,config)
        with self.assertRaisesRegex(ValueError,"independent"):load_config(p)


if __name__ == "__main__":
    unittest.main()
