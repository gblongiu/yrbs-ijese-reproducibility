"""Run the complete article workflow in an isolated output directory."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data2023", type=Path, required=True)
    parser.add_argument("--data2021", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    work = args.work_dir.resolve()
    if work == ROOT or ROOT in work.parents:
        raise SystemExit("--work-dir must be outside the repository directory")
    if work.exists() and any(work.iterdir()):
        raise SystemExit("--work-dir must be absent or empty")
    work.mkdir(parents=True, exist_ok=True)

    for name in ["scripts", "src", "revision_r1", "outputs"]:
        shutil.copytree(ROOT / name, work / name, dirs_exist_ok=True)

    python = Path(sys.executable)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    commands = [
        [python, "revision_r1/scripts/prepare_sources.py", "--data2023", args.data2023.resolve(), "--data2021", args.data2021.resolve()],
        [python, "scripts/01_build_dataset.py"],
        [python, "revision_r1/scripts/verify_2023_ascii_source.py", args.data2023.resolve()],
        [python, "revision_r1/scripts/regenerate_training_selection.py"],
        [python, "revision_r1/scripts/run_2023_revision_analyses.py"],
        [python, "revision_r1/scripts/run_2021_external_validation.py"],
        [python, "revision_r1/scripts/design_sensitivity.py"],
        [python, "revision_r1/scripts/final_evidence.py"],
        [python, "revision_r1/scripts/export_article_tables.py"],
        [python, "revision_r1/scripts/build_revision_figures.py"],
    ]
    for command in commands:
        print("Running:", " ".join(map(str, command[1:])), flush=True)
        subprocess.run([str(part) for part in command], cwd=work, env=env, check=True)
    print(f"Workflow complete: {work}")


if __name__ == "__main__":
    main()
