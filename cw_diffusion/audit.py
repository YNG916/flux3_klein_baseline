"""CPU-only inventory of real MVWD labels; no model execution or test target decoding."""

from pathlib import Path
import json, zipfile, datetime
import numpy as np
from flux_mvwd.common import read_json, write_json, file_hash


def audit(root, output):
    root = Path(root).resolve()
    meta = read_json(root / "dataset.json")
    representatives = {}
    counts = {}
    for split in ["train", "val", "test"]:
        path = root / meta["splits"][split]["path"]
        if file_hash(path) != meta["checksums"][meta["splits"][split]["path"]]:
            raise ValueError("Dataset split checksum differs")
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        counts[split] = len(records)
        for r in records:
            representatives.setdefault(r["scene"], (split, r))
    reports = []
    for scene, (split, r) in sorted(representatives.items()):
        taxonomy = read_json(root / r["taxonomy_path"])
        catalog = taxonomy["instance_catalogs"][scene]
        config = root / r["configuration_path"] / "bev/environment_base.npz"
        views = root / r["episode_path"] / "robot_views/before/robot_00.npz"
        with zipfile.ZipFile(config) as z:
            bev_names = z.namelist()
        with zipfile.ZipFile(views) as z:
            view_names = z.namelist()
        required = [
            "rgb.npy",
            "depth_linear.npy",
            "semantic.npy",
            "instance.npy",
            "normal.npy",
        ]
        assert all(n in view_names for n in required)
        floors = sorted(
            set(n.split("/")[0] for n in bev_names if n.endswith("/rgb.npy"))
        )
        assert all(
            f"{floor}/{n}.npy" in bev_names
            for floor in floors
            for n in ["rgb", "semantic", "instance", "height", "occupancy"]
        )
        reports.append(
            {
                "scene": scene,
                "split": split,
                "catalog_instances": len(catalog),
                "floor_count": len(floors),
                "all_view_modalities": True,
                "all_bev_supervision": True,
                "object_state_available": (
                    root / r["episode_path"] / "state_before.json"
                ).exists(),
                "taxonomy_sha256": file_hash(root / r["taxonomy_path"]),
            }
        )
    # Decode ONE train example to check actual dtypes/invalid values and ID linkage.
    train = next(r for split, r in representatives.values() if split == "train")
    ep = root / train["episode_path"]
    taxonomy = read_json(root / train["taxonomy_path"])
    catalog = taxonomy["instance_catalogs"][train["scene"]]
    with np.load(ep / "robot_views/before/robot_00.npz", allow_pickle=False) as a:
        arrays = {
            n: {"shape": list(a[n].shape), "dtype": str(a[n].dtype)} for n in a.files
        }
        ids = np.unique(a["instance"][0])
        depth = a["depth_linear"][0]
        validity = {
            "valid_depth_fraction": float((np.isfinite(depth) & (depth > 0)).mean()),
            "observed_instance_ids": list(map(int, ids)),
        }
    known = {int(x["public_instance_id"]) for x in catalog}
    validity["reserved_robot_instance_ids"] = [1, 2, 3]
    validity["unknown_nonzero_instance_ids"] = sorted(
        set(map(int, ids)) - known - {0, 1, 2, 3}
    )
    state = read_json(ep / "state_before.json")
    obj = state["objects"][0]
    result = {
        "schema": "cw-supervision-audit-v1",
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "release_id": meta["release_id"],
        "split_counts": counts,
        "class_count": meta["class_count"],
        "scenes_checked": len(reports),
        "max_catalog_instances": max(x["catalog_instances"] for x in reports),
        "scene_inventory": reports,
        "train_array_probe": arrays,
        "train_label_probe": validity,
        "object_fields": sorted(obj),
        "inference_conditions": "RGB BEV + authorized camera/robot metadata only",
        "training_targets": [
            "253-class semantic",
            "scene-scoped instance masks and ID mapping",
            "CV depth with modality-specific poses",
            "normal arrays (available; depth-derived normals used in training)",
            "BEV semantic/instance/height/occupancy",
            "object 6DoF poses, world AABBs and categories",
        ],
        "not_claimed": [
            "complete unseen 3D occupancy",
            "dense canonical UV textures",
            "material BRDF parameter supervision",
            "test scene GT objects as inputs",
        ],
        "test_payload_arrays_decoded": False,
        "geometry_note": "AABB dimensions are not an oriented box; predict orientation separately and compare abs(R)*size to world AABB.",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "audit.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "scenes_checked",
                    "max_catalog_instances",
                    "class_count",
                    "train_label_probe",
                    "training_targets",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result
