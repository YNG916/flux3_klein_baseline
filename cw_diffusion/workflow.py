"""Experiment contracts and checksum-preserving reuse of existing baseline encodings."""

from pathlib import Path
import datetime
import torch
from flux_mvwd.common import file_hash, read_json, write_json, digest, within
from flux_mvwd.cache import save_tensor, FeatureDataset
from .checkpoint import source_identity, environment_identity, ROOT
from .config import WorldConfig
from .data import CanonicalDataset

EXPERIMENTS = ROOT / "artifacts/cw/experiments"


def config_at(experiment):
    if Path(experiment).name != experiment or experiment in (".", ".."):
        raise ValueError("Simple experiment name required")
    p = EXPERIMENTS / experiment
    m = read_json(p / "manifest.json")
    c = read_json(p / "config.json")
    if (
        file_hash(p / "config.json") != m["config_sha256"]
        or source_identity() != m["sources"]
    ):
        raise ValueError("Frozen method config/source changed; create a new experiment")
    if file_hash(p / "baseline_config.json") != m["baseline_config_sha256"]:
        raise ValueError("Pinned baseline configuration snapshot changed")
    base_config = read_json(p / "baseline_config.json")
    dataset_root = Path(base_config["dataset_root"])
    if (
        file_hash(dataset_root / "classes.json") != m["classes_sha256"]
        or file_hash(dataset_root / "dataset.json") != m["dataset_json_sha256"]
    ):
        raise ValueError("Pinned dataset release or class vocabulary changed")
    if (
        file_hash(Path(c["baseline_checkpoint"]) / "lora.safetensors")
        != m["baseline_adapter_sha256"]
    ):
        raise ValueError("Baseline initialization weights changed")
    if any(
        file_hash(Path(c["baseline_experiment"]) / "captions" / split / "manifest.json")
        != sha
        for split, sha in m["baseline_captions_sha256"].items()
    ):
        raise ValueError("Pinned baseline captions changed")
    if environment_identity() != m["environment"]:
        raise ValueError("Method environment changed")
    return p, c


def initialize(name, config_path):
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Simple experiment name required")
    c = read_json(config_path)
    wc = WorldConfig(**c["world"]).validate()
    baseline = Path(c["baseline_experiment"]).resolve()
    original = read_json(baseline / "manifest.json")
    base = read_json(baseline / "config.json")
    if file_hash(baseline / "config.json") != original["config_sha256"]:
        raise ValueError("Baseline config changed")
    # Explicit bridge: only the new human CLI dispatch may differ; data/model/cache code must be identical.
    for f, h in original["source_sha256"].items():
        if f != "mvwd" and file_hash(ROOT / f) != h:
            raise ValueError("Baseline implementation changed: " + f)
    ck = Path(c["baseline_checkpoint"]).resolve()
    cm = read_json(ck / "metadata.json")
    if not cm.get("complete") or cm["state_files"]["lora.safetensors"] != file_hash(
        ck / "lora.safetensors"
    ):
        raise ValueError("Incomplete baseline initialization checkpoint")
    if cm["step"] != 34870:
        raise ValueError(
            "This full-method recipe initializes from the declared completed FLUX baseline"
        )
    p = EXPERIMENTS / name
    p.mkdir(parents=True, exist_ok=False)
    for d in [
        "steps",
        "logs",
        "reports",
        "cache",
        "training",
        "predictions",
        "evaluations",
        "source",
    ]:
        (p / d).mkdir()
    c["world"] = wc.to_dict()
    c["baseline_experiment"] = str(baseline)
    c["baseline_checkpoint"] = str(ck)
    write_json(p / "config.json", c)
    write_json(p / "baseline_config.json", base)
    import shutil

    shutil.copytree(
        ROOT / "cw_diffusion",
        p / "source/cw_diffusion",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    sources = source_identity()
    write_json(
        p / "manifest.json",
        {
            "schema": "cw-experiment-v1",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sources": sources,
            "config_sha256": file_hash(p / "config.json"),
            "environment": environment_identity(),
            "baseline_config_sha256": file_hash(p / "baseline_config.json"),
            "classes_sha256": file_hash(Path(base["dataset_root"]) / "classes.json"),
            "dataset_json_sha256": file_hash(
                Path(base["dataset_root"]) / "dataset.json"
            ),
            "baseline_manifest_sha256": file_hash(baseline / "manifest.json"),
            "baseline_adapter_sha256": file_hash(ck / "lora.safetensors"),
            "baseline_captions_sha256": {
                split: file_hash(baseline / "captions" / split / "manifest.json")
                for split in ("train", "val", "test")
            },
            "bridge": "Existing baseline data/model/cache implementations unchanged; new mvwd cw dispatch only",
            "dataset_supervision": "train/val only; inference RGB BEV and authorized robot/camera metadata only",
        },
    )
    write_json(
        p / "plan.json",
        {
            "epochs": c["training"]["epochs"],
            "inherited_baseline_epochs": 10,
            "additional_method_epochs": c["training"]["epochs"],
            "synchronous_frames_per_epoch": 37194,
            "views_per_epoch": 111582,
            "world_size": 4,
            "per_rank_synchronous_frames": 1,
            "accumulation": c["training"]["accumulation"],
            "target_hw": base["target_hw"],
            "stage": c["training"]["stage"],
            "full_architecture": True,
            "ablation_runs": False,
            "supervision": [
                "BEV semantic/instance/height/occupancy",
                "perspective RGB/depth/semantic/instance",
                "object AABB and pose",
                "frozen DINO features",
            ],
            "not_available": [
                "complete unseen 3D occupancy",
                "ground-truth canonical texture atlas",
                "material parameter labels",
            ],
            "primary_checkpoint": "fixed declared endpoint; never select with test GT",
        },
    )
    print(p)


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move(v, device) for v in value)
    return value


def tensor_tree(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: tensor_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tensor_tree(v) for v in value]
    return value


class GroupCache(torch.utils.data.Dataset):
    def __init__(self, root, baseline):
        self.root = Path(root)
        self.meta = read_json(self.root / "manifest.json")
        if self.meta.get("source") != source_identity():
            raise ValueError("Group cache source contract changed")
        if not self.meta.get("complete"):
            raise ValueError("Group cache incomplete")
        self.rows = self.meta["rows"]
        self.native_root = Path(baseline) / "features" / self.meta["split"]
        native = read_json(self.native_root / "manifest.json")
        if (
            file_hash(self.native_root / "manifest.json")
            != self.meta["native_manifest_sha256"]
        ):
            raise ValueError("Baseline cache manifest changed")
        self.native = FeatureDataset(self.native_root, native["samples"])
        self.verified = set()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        f = within(self.root, r["path"])
        for relative, sha in r["files"].items():
            if relative not in self.verified:
                if file_hash(within(self.root, relative)) != sha:
                    raise ValueError("Method group/static cache changed")
                self.verified.add(relative)
        sample = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        from .storage import restore

        static = torch.load(
            within(self.root, sample["static_path"]),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        sample = restore(sample, static)
        sample["samples"] = [self.native[j] for j in r["native_indices"]]
        return sample


def load_teacher(device):
    weights = ROOT.parent / "top2pano/models/metrics/torch_hub"
    repositories = sorted(weights.glob("facebookresearch_dino_*"))
    if not repositories:
        raise FileNotFoundError("Pinned local DINO source is missing")
    teacher = torch.hub.load(
        str(repositories[0]), "dino_vits16", source="local", pretrained=False
    )
    ck = weights / "checkpoints/dino_deitsmall16_pretrain.pth"
    if (
        file_hash(ck)
        != "1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7"
    ):
        raise ValueError("DINO checkpoint changed")
    teacher.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    teacher = teacher.to(device).eval().requires_grad_(False)
    return teacher, {
        "weights_sha256": file_hash(ck),
        "source_sha256": {
            str(p.relative_to(repositories[0])): file_hash(p)
            for p in repositories[0].glob("*.py")
        },
    }


def prepare(experiment, split, max_episodes=None):
    if split not in ("train", "val"):
        raise ValueError("Never prepare privileged test targets")
    from accelerate import Accelerator

    p, c = config_at(experiment)
    acc = Accelerator()
    device = acc.device
    data = CanonicalDataset(
        read_json(p / "baseline_config.json"), split, max_episodes=max_episodes
    )
    root = p / "cache" / split
    if acc.is_main_process:
        root.mkdir(exist_ok=True)
    acc.wait_for_everyone()
    native_root = Path(c["baseline_experiment"]) / "features" / split
    native = read_json(native_root / "manifest.json")
    if not native.get("complete") or native["dataset"]["split"] != split:
        raise ValueError("Complete baseline feature cache required")
    lookup = {
        (
            r["query"]["episode_id"],
            r["query"]["frame_index"],
            r["query"]["robots"][0],
        ): i
        for i, r in enumerate(native["samples"])
    }
    teacher, teacher_identity = load_teacher(device)
    identity = {
        "schema": "cw-group-cache-v1",
        "split": split,
        "groups_sha256": digest(data.groups),
        "source": source_identity(),
        "native_manifest_sha256": file_hash(native_root / "manifest.json"),
        "teacher": teacher_identity,
        "ground_truth_loaded": True,
        "storage_schema": "static-bev-u8-rgb-exact-labels-dino-bf16-v1",
    }
    identity_path = root / "identity.json"
    if acc.is_main_process:
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("Cache preparation contract changed")
        write_json(identity_path, identity)
    acc.wait_for_everyone()
    for i in (
        j
        for j in range(len(data.groups))
        if (j // data.config["frames_per_episode"]) % acc.num_processes
        == acc.process_index
    ):
        row = root / f"rows/{i:07d}.json"
        if row.exists():
            r = read_json(row)
            if any(file_hash(within(root, n)) != sha for n, sha in r["files"].items()):
                raise ValueError("Existing group cache checksum mismatch")
            continue
        cond = data.conditions(i)
        from flux_mvwd.preparation import input_key
        from flux_mvwd.cache import existing_row

        caps = Path(c["baseline_experiment"]) / "captions" / split
        for robot in range(3):
            ni = lookup[
                (cond["group"]["episode_id"], cond["group"]["frame_index"], robot)
            ]
            query = native["samples"][ni]["query"]
            caprow = existing_row(caps, ni, query)
            if caprow is None:
                raise ValueError("Baseline caption missing")
            cap = read_json(within(caps, caprow["path"]))
            if cap["ground_truth_loaded"] is not False or cap[
                "input_sha256"
            ] != input_key(cond["native_conditions"][robot]):
                raise ValueError(
                    "Fresh conditions differ from cached baseline features"
                )
        target = data.supervision(i)
        with torch.inference_mode():
            rgb = target["rgb"].to(device)
            mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            features = teacher.get_intermediate_layers((rgb - mean) / std, n=1)[0][
                :, 1:
            ]
        target.update(
            teacher=features.cpu().contiguous(), rgb_K=cond["K"], rgb_c2w=cond["c2w"]
        )
        condition = {k: v for k, v in cond.items() if k != "native_conditions"}
        from .storage import compact, split_static

        condition, target = compact(condition, target)
        condition, target, static = split_static(condition, target)
        record = data.data.by_id[cond["group"]["episode_id"]]
        static_key = digest(
            [record["configuration_path"], cond["world_to_bev"].tolist()]
        )
        static_file = f"objects/rank_{acc.process_index:04d}/{static_key}.pt"
        if not (root / static_file).exists():
            save_tensor(root / static_file, tensor_tree(static))
        filename = f"groups/{i:07d}.pt"
        save_tensor(
            root / filename,
            tensor_tree(
                {
                    "condition": condition,
                    "supervision": target,
                    "static_path": static_file,
                }
            ),
        )
        indices = [
            lookup[(cond["group"]["episode_id"], cond["group"]["frame_index"], v)]
            for v in range(3)
        ]
        write_json(
            row,
            {
                "group_index": i,
                "path": filename,
                "sha256": file_hash(root / filename),
                "native_indices": indices,
                "files": {
                    filename: file_hash(root / filename),
                    static_file: file_hash(root / static_file),
                },
            },
        )
        print(
            f"rank={acc.process_index} prepared group={i + 1}/{len(data.groups)}",
            flush=True,
        )
    acc.wait_for_everyone()
    if acc.is_main_process:
        rows = [read_json(root / f"rows/{i:07d}.json") for i in range(len(data.groups))]
        write_json(
            root / "manifest.json",
            {
                **identity,
                "complete": True,
                "rows": rows,
                "full_split": len(rows) * 3 == native["sample_count"],
                "groups": len(rows),
                "views": 3 * len(rows),
            },
        )
    acc.wait_for_everyone()
