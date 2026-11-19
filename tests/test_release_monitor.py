from pathlib import Path

import pytest

from data.generate import generate
from src.monitor import run as monitor_run
from src.replay import run as replay_run
from src.train import load_json, run as train_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ready(tmp_path_factory):
    base = tmp_path_factory.mktemp("rel")
    data_dir = base / "data"
    art = base / "artifacts"
    cfg = load_json(ROOT / "configs" / "data.json")
    generate(cfg, data_dir, quick=True)
    train_run(data_dir / "orders_quick.csv", art, quick=True)
    return data_dir / "orders_quick.csv", art


def test_replay_performs_rollback(ready, monkeypatch):
    data, art = ready
    # replay пишет в ROOT/reports — нормально для учебного стенда
    summary = replay_run(data, art, quick=True)
    assert summary["n_events"] > 0
    assert summary["n_rollbacks"] >= 1


def test_monitor_flags_shift(ready):
    data, _ = ready
    report = monitor_run(data, inject_shift=True)
    assert report["alerts"], "ожидали PSI-алерт после сдвига"
    assert report["first_signal"]["action"] == "investigate_data_before_quality"
