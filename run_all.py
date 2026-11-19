"""Полный прогон: данные → обучение → оценка → бенч → replay → monitor → retrain."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(cmd: list[str]):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20k заказов вместо 200k")
    args = ap.parse_args()
    py = sys.executable
    q = ["--quick"] if args.quick else []
    step([py, "data/generate.py", *q])
    step([py, "-m", "src.train", *q])
    step([py, "-m", "src.evaluate", *q])
    step([py, "-m", "src.benchmark", *q, "--n", "2000"])
    step([py, "-m", "src.replay", *q])
    step([py, "-m", "src.monitor", *q])
    step([py, "-m", "src.retrain", *q])
    step([py, "-m", "pytest", "-q"])
    print("OK")


if __name__ == "__main__":
    main()
