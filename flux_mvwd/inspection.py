from pathlib import Path
import json
from .preparation import dataset as select_data
from .common import digest, write_json

def inspect(config, args):
    data = select_data(config, args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    query_count = 0
    # Count only metadata. Decode a bounded preview, not the full dataset.
    for query in data.queries():
        if query_count < args.preview_samples:
            condition = data.conditions(query)
            dest = output / f"sample_{query_count:04d}"
            dest.mkdir()
            for i, image in enumerate(condition["images"]):
                image.save(dest / f"condition_{i}.png")
            (dest / "prompt.txt").write_text(condition["prompt"])
            write_json(dest / "geometry.json", {"query": query, "cameras": condition["cameras"],
                "payload": condition["payload"], "world_to_bev": condition["world_to_bev"].tolist()})
            if args.with_targets:
                data.target(query).save(dest / "target.png")
        query_count += 1
    report = {"dataset": data.provenance(), "query_count": query_count,
        "view_count": query_count * (3 if config["mode"] == "joint" else 1),
        "ground_truth_loaded": bool(args.with_targets and args.preview_samples),
        "mode": config["mode"], "config_sha256": digest(config)}
    write_json(output / "inspection.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "dataset"}, indent=2))

