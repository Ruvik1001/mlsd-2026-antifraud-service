import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from data.generate import generate
from src.service import FeatureStore, create_app
from src.train import load_json, run as train_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    base = tmp_path_factory.mktemp("proj")
    data_dir = base / "data"
    art = base / "artifacts"
    cfg = load_json(ROOT / "configs" / "data.json")
    generate(cfg, data_dir, quick=True)
    # train пишет reports в ROOT — ок
    train_run(data_dir / "orders_quick.csv", art, quick=True)
    return art, data_dir


def test_ready_and_predict(artifacts):
    art, _ = artifacts
    app = create_app(artifact_dir=art, store=FeatureStore({"k": 2}))
    with TestClient(app) as c:
        assert c.get("/health/ready").status_code == 200
        r = c.post(
            "/predict",
            json={
                "request_id": "r1",
                "order_id": "o1",
                "amount_rub": 1000,
                "account_age_days": 100,
                "orders_1h": 0,
                "device_risk": 0.1,
                "payment_method": "card",
                "store_key": "k",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] in {"approve", "review"}
        assert body["model_version"]
        assert body["source"] == "model"


def test_unknown_payment_review(artifacts):
    art, _ = artifacts
    app = create_app(artifact_dir=art)
    with TestClient(app) as c:
        r = c.post(
            "/predict",
            json={
                "request_id": "r2",
                "order_id": "o2",
                "amount_rub": 1000,
                "account_age_days": 100,
                "device_risk": 0.1,
                "payment_method": "bitcoin",
            },
        )
        assert r.status_code == 200
        assert r.json()["decision"] == "review"
        assert "unknown_payment_method" in r.json()["reasons"]


def test_store_timeout_degraded(artifacts):
    art, _ = artifacts
    app = create_app(artifact_dir=art, store=FeatureStore({"k": 5}, delay_ms=50))
    # timeout 20ms in config
    with TestClient(app) as c:
        r = c.post(
            "/predict",
            json={
                "request_id": "r3",
                "order_id": "o3",
                "amount_rub": 5000,
                "account_age_days": 40,
                "orders_1h": 1,
                "device_risk": 0.2,
                "payment_method": "card",
                "store_key": "k",
            },
        )
        assert r.status_code == 200
        assert r.json()["degraded"] is True
        assert "feature_timeout" in r.json()["reasons"]


def test_ready_503_without_model(tmp_path):
    app = create_app(artifact_dir=tmp_path)
    with TestClient(app) as c:
        assert c.get("/health/ready").status_code == 503
        r = c.post(
            "/predict",
            json={
                "request_id": "r4",
                "order_id": "o4",
                "amount_rub": 25000,
                "account_age_days": 3,
                "device_risk": 0.1,
                "payment_method": "card",
            },
        )
        assert r.status_code == 200
        assert r.json()["source"] == "rule-v1"
