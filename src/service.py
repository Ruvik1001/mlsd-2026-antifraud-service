"""FastAPI сервис антифрода с версией, деградациями и очередью review."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

from src.features import FEATURE_NAMES, encode_payment

ROOT = Path(__file__).resolve().parents[1]


class OrderIn(BaseModel):
    request_id: str
    order_id: str
    amount_rub: float
    account_age_days: float
    orders_1h: float | None = None
    device_risk: float
    payment_method: str
    store_key: str | None = None


class FeatureStore:
    def __init__(self, values: dict[str, float] | None = None, delay_ms: float = 0.0, fail: bool = False):
        self.values = values or {}
        self.delay_ms = delay_ms
        self.fail = fail

    async def get(self, key: str):
        if self.fail:
            raise ConnectionError("store down")
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000.0)
        return self.values.get(key)


class Bundle:
    def __init__(self, path: Path):
        raw = joblib.load(path / "model.joblib")
        man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if list(raw["feature_names"]) != list(FEATURE_NAMES):
            raise ValueError("feature order mismatch")
        self.scaler = raw["scaler"]
        self.linear = raw["linear"]
        self.tree = raw["tree"]
        self.threshold = float(raw["threshold"])
        self.model_version = str(man["model_version"])
        self.model_kind = raw.get("model_kind", "tree")
        self.use_batch = True  # оптимизация: batch predict_proba

    def features_vec(self, amount, age, orders_1h, device_risk, payment_method):
        pc, pw, po = encode_payment(payment_method)
        return np.array([[amount, age, orders_1h, device_risk, pc, pw, po]], dtype=np.float64)

    def predict_proba(self, x: np.ndarray) -> float:
        if self.model_kind == "linear":
            xs = self.scaler.transform(x)
            return float(self.linear.predict_proba(xs)[0, 1])
        return float(self.tree.predict_proba(x)[0, 1])

    def predict_proba_batch(self, x: np.ndarray) -> np.ndarray:
        if self.model_kind == "linear":
            return self.linear.predict_proba(self.scaler.transform(x))[:, 1]
        return self.tree.predict_proba(x)[:, 1]


def rule_fallback(amount, age):
    if amount >= 20000 and age < 7:
        return "review", 1.0, "rule-v1"
    return "approve", 0.0, "rule-v1"


def create_app(artifact_dir: Path | None = None, store: FeatureStore | None = None, serve_cfg: dict | None = None):
    artifact_dir = Path(artifact_dir or (ROOT / "artifacts"))
    serve_cfg = serve_cfg or json.loads((ROOT / "configs" / "serve.json").read_text(encoding="utf-8"))
    timeout_s = float(serve_cfg.get("feature_timeout_ms", 20)) / 1000.0
    queue_limit = int(serve_cfg.get("review_queue_limit", 3000))

    state: dict[str, Any] = {
        "bundle": None,
        "store": store or FeatureStore(),
        "review_queue": 0,
        "registry": CollectorRegistry(),
    }

    req_c = Counter("antifraud_requests_total", "requests", ["decision", "source"], registry=state["registry"])
    deg_c = Counter("antifraud_degradations_total", "degradations", ["reason"], registry=state["registry"])
    lat_h = Histogram("antifraud_latency_seconds", "latency", registry=state["registry"])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            state["bundle"] = Bundle(artifact_dir)
        except Exception:
            state["bundle"] = None
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        if state["bundle"] is None:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return {"status": "ready", "model_version": state["bundle"].model_version}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(state["registry"]), media_type=CONTENT_TYPE_LATEST)

    async def fetch_orders_1h(key: str | None, fallback: float | None):
        reasons = []
        if key is None:
            return float(fallback or 0.0), reasons
        try:
            val = await asyncio.wait_for(state["store"].get(key), timeout=timeout_s)
        except asyncio.TimeoutError:
            deg_c.labels(reason="feature_timeout").inc()
            return float(fallback or 0.0), ["feature_timeout"]
        except ConnectionError:
            deg_c.labels(reason="feature_unavailable").inc()
            return float(fallback or 0.0), ["feature_unavailable"]
        if val is None:
            deg_c.labels(reason="feature_missing").inc()
            return float(fallback or 0.0), ["feature_missing"]
        return float(val), reasons

    @app.post("/predict")
    async def predict(order: OrderIn):
        t0 = time.perf_counter()
        reasons: list[str] = []
        degraded = False
        source = "model"

        # валидация входа
        if order.payment_method not in {"card", "wallet", "cod"}:
            if serve_cfg.get("unknown_category_policy") == "review":
                decision, prob, source = "review", 1.0, "policy"
                reasons.append("unknown_payment_method")
                degraded = True
                out = _pack(order, decision, prob, source, degraded, reasons, state)
                _observe(req_c, lat_h, decision, source, t0)
                return out
            return JSONResponse({"error": "unknown_payment_method"}, status_code=400)

        required = [order.amount_rub, order.account_age_days, order.device_risk]
        if any(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in required):
            return JSONResponse({"error": "missing_required"}, status_code=400)

        orders_1h, fr = await fetch_orders_1h(order.store_key, order.orders_1h)
        reasons.extend(fr)
        if fr:
            degraded = True

        bundle: Bundle | None = state["bundle"]
        if bundle is None:
            decision, prob, source = rule_fallback(order.amount_rub, order.account_age_days)
            degraded = True
            reasons.append("model_unavailable")
        else:
            x = bundle.features_vec(
                order.amount_rub,
                order.account_age_days,
                orders_1h,
                order.device_risk,
                order.payment_method,
            )
            # оптимизация: один вызов batch API даже для 1 строки (тот же код путь)
            if bundle.use_batch:
                prob = float(bundle.predict_proba_batch(x)[0])
            else:
                prob = bundle.predict_proba(x)
            decision = "review" if prob >= bundle.threshold else "approve"
            source = "model"

        if decision == "review":
            if state["review_queue"] >= queue_limit:
                decision = "approve"
                degraded = True
                reasons.append("review_queue_full")
                deg_c.labels(reason="review_queue_full").inc()
            else:
                state["review_queue"] += 1

        out = _pack(order, decision, prob, source, degraded, reasons, state)
        _observe(req_c, lat_h, decision, source, t0)
        return out

    return app


def _pack(order, decision, prob, source, degraded, reasons, state):
    b = state["bundle"]
    return {
        "request_id": order.request_id,
        "order_id": order.order_id,
        "decision": decision,
        "probability": float(prob),
        "model_version": None if b is None else b.model_version,
        "source": source,
        "degraded": degraded,
        "reasons": reasons,
    }


def _observe(req_c, lat_h, decision, source, t0):
    req_c.labels(decision=decision, source=source).inc()
    lat_h.observe(time.perf_counter() - t0)


app = create_app()
