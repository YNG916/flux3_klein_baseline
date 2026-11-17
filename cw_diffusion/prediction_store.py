"""Atomic synchronous-triplet commits; resume keeps complete frames and failed attempts."""

from pathlib import Path
import json, os
from flux_mvwd.common import digest, file_hash, read_json, write_json, within


class PredictionStore:
    def __init__(self, root, metadata, create=False):
        self.root = Path(root)
        self.metadata = {**metadata, "complete": False}
        if metadata.get("ground_truth_loaded") is not False:
            raise ValueError("GT-free generation metadata required")
        self.identity = digest(self.metadata)
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            write_json(self.root / "metadata.json", self.metadata)
        old = read_json(self.root / "metadata.json")
        if {**old, "complete": False} != self.metadata:
            raise ValueError(
                "Prediction resume requires identical model, source, environment, seed and query support"
            )

    @staticmethod
    def frame_path(group):
        return Path(group["episode_id"]) / f"frame_{group['frame_index']:03d}"

    def read_group(self, index, group):
        path = self.root / self.frame_path(group) / "frame_receipt.json"
        if not path.exists():
            return None
        receipt = read_json(path)
        if (
            receipt.get("schema") != "cw-complete-synchronous-frame-v1"
            or receipt.get("generation_identity") != self.identity
            or receipt.get("group_index") != index
            or receipt.get("group") != group
        ):
            raise ValueError("Frame receipt differs from generation contract")
        rows = receipt["rows"]
        if len(rows) != 3 or {r["robot_id"] for r in rows} != {0, 1, 2}:
            raise ValueError("Incomplete frame triplet")
        for r in rows:
            if (
                r["episode_id"] != group["episode_id"]
                or r["frame_index"] != group["frame_index"]
            ):
                raise ValueError("Frame row query differs")
        for relative, sha in receipt["files"].items():
            if file_hash(within(path.parent, relative)) != sha:
                raise ValueError("Completed frame payload changed")
        return rows

    def staging(self, index, attempt):
        if Path(attempt).name != attempt:
            raise ValueError("Simple attempt name required")
        stage = self.root / ".staging" / attempt / f"{index:07d}"
        stage.mkdir(parents=True, exist_ok=False)
        return stage

    def commit(self, index, group, stage, rows):
        destination = self.root / self.frame_path(group)
        if destination.exists():
            raise FileExistsError("Never overwrite a committed frame")
        expected = {0, 1, 2}
        if len(rows) != 3 or {r["robot_id"] for r in rows} != expected:
            raise ValueError("Commit requires all three cameras")
        for row in rows:
            if (
                row["episode_id"] != group["episode_id"]
                or row["frame_index"] != group["frame_index"]
            ):
                raise ValueError("Output query differs")
            relative = Path(row["path"]).relative_to(self.frame_path(group))
            image = within(stage, relative)
            if file_hash(image) != row["sha256"]:
                raise ValueError("Prediction checksum differs")
            if (
                read_json(image.with_suffix(".generation.json")).get(
                    "ground_truth_loaded"
                )
                is not False
            ):
                raise ValueError("Generation sidecar must be GT-free")
        files = {
            str(p.relative_to(stage)): file_hash(p)
            for p in stage.rglob("*")
            if p.is_file()
        }
        write_json(
            stage / "frame_receipt.json",
            {
                "schema": "cw-complete-synchronous-frame-v1",
                "generation_identity": self.identity,
                "group_index": index,
                "group": group,
                "rows": rows,
                "files": files,
                "ground_truth_loaded": False,
            },
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, destination)
        return self.read_group(index, group)

    def finish(self, groups, attempt):
        rows = []
        for i, group in enumerate(groups):
            completed = self.read_group(i, group)
            if completed is None:
                raise ValueError("Missing synchronous frame")
            rows.extend(completed)
        if len(rows) != self.metadata["view_count"]:
            raise ValueError("Full generation coverage differs")
        rows.sort(key=lambda r: (r["episode_id"], r["frame_index"], r["robot_id"]))
        content = "".join(json.dumps(r) + "\n" for r in rows)
        manifest = self.root / "predictions.jsonl"
        if manifest.exists():
            if manifest.read_text() != content:
                raise ValueError("Existing final manifest differs")
        else:
            temporary = self.root / f".predictions-{attempt}.tmp"
            temporary.write_text(content)
            os.rename(temporary, manifest)
        write_json(self.root / "metadata.json", {**self.metadata, "complete": True})
