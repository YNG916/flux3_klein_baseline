import json
import sys
from cw_diffusion import cli, workflow


def test_preparation_default_name_and_unique_retry_logging(tmp_path, monkeypatch):
    for name in ["steps", "logs"]:
        (tmp_path / name).mkdir()
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(cli, "config_at", lambda _: (tmp_path, {}))
    calls = []
    monkeypatch.setattr(workflow, "prepare", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["cw", "prepare", "EXP", "--split", "train"])
    cli.main()
    state = json.loads((tmp_path / "steps/prepare_train.json").read_text())
    assert state["status"] == "completed" and state["exit_code"] == 0
    monkeypatch.setattr(
        sys,
        "argv",
        ["cw", "prepare", "EXP", "--split", "train", "--name", "prepare_train_a001"],
    )
    cli.main()
    assert (tmp_path / "steps/prepare_train_a001.json").exists()
    assert len(calls) == 2 and len(list((tmp_path / "logs").glob("*.log"))) == 2
