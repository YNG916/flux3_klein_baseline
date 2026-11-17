import json
import pytest
from PIL import Image
from cw_diffusion.prediction_store import PredictionStore
from flux_mvwd.common import file_hash, write_json


def payload(store, index, group, attempt="attempt0"):
    stage = store.staging(index, attempt)
    rows = []
    for robot in range(3):
        f = stage / f"robot_{robot:02d}.png"
        Image.new("RGB", (4, 4), (30 * robot, 40, 60)).save(f)
        write_json(f.with_suffix(".generation.json"), {"ground_truth_loaded": False})
        rows.append(
            {
                **group,
                "robot_id": robot,
                "group_robots": [robot],
                "seed": 42 + robot,
                "path": str(store.frame_path(group) / f.name),
                "sha256": file_hash(f),
            }
        )
    return stage, rows


def test_atomic_frame_resume_preserves_complete_triplets(tmp_path):
    meta = {
        "ground_truth_loaded": False,
        "view_count": 6,
        "model": "frozen",
        "seed": 42,
    }
    root = tmp_path / "predictions"
    store = PredictionStore(root, meta, create=True)
    groups = [{"episode_id": "ep_fixture", "frame_index": i} for i in [1, 2]]
    stage, rows = payload(store, 0, groups[0])
    store.commit(0, groups[0], stage, rows)
    assert not stage.exists() and len(store.read_group(0, groups[0])) == 3
    # Interrupted second frame is retained under staging and never reported as complete.
    partial = store.staging(1, "failed")
    (partial / "partial.txt").write_text("retained")
    resumed = PredictionStore(root, meta)
    assert resumed.read_group(1, groups[1]) is None
    stage, rows = payload(resumed, 1, groups[1], "retry")
    resumed.commit(1, groups[1], stage, rows)
    resumed.finish(groups, "retry")
    assert (partial / "partial.txt").exists() and json.loads(
        (root / "metadata.json").read_text()
    )["complete"]
    assert len((root / "predictions.jsonl").read_text().splitlines()) == 6
    resumed_again = PredictionStore(root, meta)
    resumed_again.finish(groups, "verified")


def test_model_mix_partial_triplet_and_corruption_rejected(tmp_path):
    meta = {"ground_truth_loaded": False, "view_count": 3, "model": "one"}
    store = PredictionStore(tmp_path / "out", meta, create=True)
    group = {"episode_id": "ep", "frame_index": 1}
    stage, rows = payload(store, 0, group)
    with pytest.raises(ValueError, match="three"):
        store.commit(0, group, stage, rows[:2])
    store.commit(0, group, stage, rows)
    with pytest.raises(ValueError, match="identical"):
        PredictionStore(tmp_path / "out", {**meta, "model": "different"})
    f = tmp_path / "out" / rows[0]["path"]
    with f.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        store.read_group(0, group)
