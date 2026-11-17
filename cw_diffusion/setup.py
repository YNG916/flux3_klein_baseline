"""Create a separate CW dependency overlay without modifying the baseline environment."""

from pathlib import Path
import os, subprocess, sys


def main():
    root = Path(__file__).resolve().parents[1]
    workspace = Path(os.environ.get("MULTIVIEW_STORAGE_ROOT", root.parent.parent))
    baseline = workspace / "envs/flux3-klein"
    environment = workspace / "envs/cw-diffusion"
    subprocess.run(
        [str(baseline / "bin/python"), "-m", "venv", "--without-pip", str(environment)],
        check=True,
    )
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    target = environment / "lib" / version / "site-packages"
    target.mkdir(parents=True, exist_ok=True)
    (target / "baseline_readonly.pth").write_text(
        str(baseline / "lib" / version / "site-packages") + "\n"
    )
    subprocess.run(
        [
            str(baseline / "bin/python"),
            "-m",
            "pip",
            "--python",
            str(environment / "bin/python"),
            "install",
            "--no-deps",
            "--ignore-installed",
            "-r",
            str(root / "requirements-cw.txt"),
        ],
        check=True,
    )
    print("CW environment:", environment)
    print(
        "Baseline packages are imported read-only; baseline environment was not updated."
    )
    print("Next: ./mvwd cw doctor")


if __name__ == "__main__":
    main()
