"""Protect GT-free generation, immutable predictions and correct robot/group exports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from test_data import fixture
from flux_mvwd.common import file_hash, sample_seed, write_json
from flux_mvwd.data import Level1Data
from tools.visualizations import export_comparisons, panels


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = fixture(self.root / "data")
        self.predictions = self.root / "experiment/predictions/run"
        self.predictions.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def predictions_fixture(self, joint=False):
        if joint:
            self.config["mode"] = "joint"
        data = Level1Data(self.config, "train")
        rows = []
        for query in data.queries():
            for robot in query["robots"]:
                path = Path("ep") / f"robot_{robot:02d}" / f"frame_{query['frame_index']:03d}.png"
                (self.predictions / path).parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (96, 64), (100, 100, 100)).save(self.predictions / path)
                rows.append({"episode_id": "ep", "frame_index": query["frame_index"], "robot_id": robot,
                    "group_robots": query["robots"], "path": str(path),
                    "seed": sample_seed(42, "fixture", "ep", query["frame_index"], query["robots"], self.config["mode"])})
        (self.predictions / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write_json(self.predictions / "metadata.json", {"schema": "flux3-mvwd-predictions-v1", "complete": True,
            "config": self.config, "dataset": data.provenance(), "ground_truth_loaded": False,
            "base_seed": 42, "lora_sha256": "fixture", "view_count": len(rows)})
        return data, rows

    def test_incomplete_or_gt_used_generation_never_reads_targets(self):
        self.predictions_fixture()
        metadata = self.predictions / "metadata.json"
        original = json.loads(metadata.read_text())
        for values in ({"complete": False}, {"ground_truth_loaded": True}, {"ground_truth_loaded": None}):
            write_json(metadata, {**original, **values})
            with patch.object(Level1Data, "target", side_effect=AssertionError("GT leak")):
                with self.assertRaisesRegex(ValueError, "completed GT-free"):
                    export_comparisons(self.predictions)
        self.assertFalse((self.predictions / "visualization_receipt.json").exists())

    def test_preserves_predictions_legacy_comparison_and_verified_repeat(self):
        data, rows = self.predictions_fixture()
        first = self.predictions / rows[0]["path"]
        target = data.target({"episode_id": "ep", "frame_index": 0, "robots": [0]})
        with Image.open(first) as im:
            legacy = panels([target, im.convert("RGB")], ["GT", "FLUX prediction (diagnostic)"], height=64)
        old = first.with_suffix(".comparison.png"); legacy.save(old)
        hashes = {row["path"]: file_hash(self.predictions / row["path"]) for row in rows}
        old_hash = file_hash(old)
        report = export_comparisons(self.predictions)
        self.assertTrue(report["complete"])
        self.assertEqual(report["exported_images"], 6)
        self.assertEqual(report["multiview_groups"], 2)
        self.assertEqual(file_hash(old), old_hash)
        self.assertEqual(hashes, {r["path"]: file_hash(self.predictions / r["path"]) for r in rows})
        self.assertFalse(json.loads((self.predictions / "metadata.json").read_text())["ground_truth_loaded"])
        self.assertTrue(first.with_suffix(".bev_comparison.png").exists())
        self.assertTrue((self.predictions / "index.html").exists())
        with patch.object(Level1Data, "target", side_effect=AssertionError("Verified repeat should reuse exports")):
            self.assertEqual(export_comparisons(self.predictions), report)
        # A changed prediction or export must never be silently accepted or replaced.
        Image.new("RGB", (96, 64), "black").save(first)
        with self.assertRaisesRegex(ValueError, "Original prediction changed"):
            export_comparisons(self.predictions)

    def test_joint_condition_group_and_per_robot_gt(self):
        _, rows = self.predictions_fixture(joint=True)
        export_comparisons(self.predictions)
        for row in rows:
            path = self.predictions / row["path"]
            camera = json.loads(path.with_suffix(".camera.json").read_text())
            self.assertEqual(camera["query"]["robots"], [0, 1, 2])
            self.assertEqual(len(camera["cameras"]), 3)
            with Image.open(path.with_suffix(".gt.png")) as gt:
                color = gt.getpixel((0, 0))
            self.assertEqual(color, tuple(255 if i == row["robot_id"] else 0 for i in range(3)))

    def test_bad_seed_rejected_before_gt_reads(self):
        _, rows = self.predictions_fixture()
        rows[0]["seed"] += 1
        (self.predictions / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        with patch.object(Level1Data, "target", side_effect=AssertionError("Bad query read GT")):
            with self.assertRaisesRegex(ValueError, "seed/group"):
                export_comparisons(self.predictions)

    def test_mismatched_original_conditioning_source_blocks_gt_export(self):
        self.predictions_fixture()
        write_json(self.predictions.parent.parent / "manifest.json",
                   {"source_sha256": {"flux_mvwd/data.py": "0" * 64}})
        with patch.object(Level1Data, "target", side_effect=AssertionError("Wrong source read GT")):
            with self.assertRaisesRegex(ValueError, "conditioning implementation differs"):
                export_comparisons(self.predictions)
