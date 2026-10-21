"""Optional integration tests against the user's existing metric implementation."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from test_data import fixture
from flux_mvwd.common import sample_seed, write_json
from flux_mvwd.data import Level1Data
from flux_mvwd.evaluation import evaluate


@unittest.skipUnless(os.environ.get("MVWD_METRICS_REPO"), "Set MVWD_METRICS_REPO for metric integration")
class MetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = fixture(self.root / "data")
        self.predictions = self.root / "predictions"
        self.predictions.mkdir()
        data = Level1Data(self.config, "test")
        rows = []
        for i, query in enumerate(data.queries()):
            robot = query["robots"][0]
            Image.new("RGB", (96, 64), (120, 120, 120)).save(self.predictions / f"{i}.png")
            rows.append({"episode_id": query["episode_id"], "robot_id": robot, "frame_index": query["frame_index"],
                "group_robots": [robot], "path": f"{i}.png", "seed": sample_seed(42, "fixture", "ep", query["frame_index"], [robot], "independent")})
        self.rows = rows
        self.write_rows()
        write_json(self.predictions / "metadata.json", {"schema": "flux3-mvwd-predictions-v1", "complete": True,
            "ground_truth_loaded": False, "config": self.config, "dataset": data.provenance(), "lora_sha256": None,
            "versions": {"synthetic_fixture": True}, "base_seed": 42, "view_count": len(rows)})

    def write_rows(self):
        (self.predictions / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))

    def tearDown(self):
        self.temp.cleanup()

    def args(self, preview=True):
        return argparse.Namespace(metrics_repo=os.environ["MVWD_METRICS_REPO"], output=str(self.root / "eval"),
            predictions=[str(self.predictions)], metrics=["psnr"], device="cpu", weights_dir=str(self.root / "weights"), preview=preview)

    def test_psnr_evaluation_uses_all_views(self):
        evaluate(self.config, self.args())
        report = json.loads((self.root / "eval/metrics.json").read_text())
        self.assertFalse(report["benchmark_ready"])
        self.assertGreater(report["metrics"]["psnr"]["mean"], 0)
        self.assertEqual(len((self.root / "eval/per_query.jsonl").read_text().splitlines()), 6)

    def test_wrong_seed_is_rejected(self):
        self.rows[0]["seed"] += 1
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Seed/group"):
            evaluate(self.config, self.args())


if __name__ == "__main__":
    unittest.main()
