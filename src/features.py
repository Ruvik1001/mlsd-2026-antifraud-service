"""Point-in-time признаки без утечки будущего."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "amount_rub",
    "account_age_days",
    "orders_1h",
    "device_risk",
    "pay_card",
    "pay_wallet",
    "pay_cod",
]

PAY_MAP = {"card": 0, "wallet": 1, "cod": 2}


def encode_payment(method: str) -> tuple[float, float, float]:
    m = str(method)
    return (
        1.0 if m == "card" else 0.0,
        1.0 if m == "wallet" else 0.0,
        1.0 if m == "cod" else 0.0,
    )


def row_features(row) -> np.ndarray:
    pc, pw, po = encode_payment(row["payment_method"])
    return np.array(
        [
            float(row["amount_rub"]),
            float(row["account_age_days"]),
            float(row["orders_1h"]),
            float(row["device_risk"]),
            pc,
            pw,
            po,
        ],
        dtype=np.float64,
    )


def matrix_from_frame(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([row_features(r) for _, r in df.iterrows()])


def point_in_time_orders_1h(history: pd.DataFrame, user_id: str, decision_ts: int) -> float:
    """Считаем только события, которые и произошли, и поступили до decision_ts."""
    if history is None or len(history) == 0:
        return 0.0
    h = history
    mask = (
        (h["user_id"] == user_id)
        & (h["event_ts"] < decision_ts)
        & (h["ingest_ts"] <= decision_ts)
        & (h["event_ts"] >= decision_ts - 3600)
    )
    return float(mask.sum())


def assert_no_future_leak(history: pd.DataFrame, user_id: str, decision_ts: int, baseline: float):
    """Позднее событие не меняет уже посчитанный прогнозный признак."""
    before = point_in_time_orders_1h(history, user_id, decision_ts)
    late = history.copy()
    if len(late) == 0:
        late = pd.DataFrame(
            columns=["user_id", "event_ts", "ingest_ts", "order_id"]
        )
    extra = {
        "user_id": user_id,
        "event_ts": decision_ts + 10,
        "ingest_ts": decision_ts + 20,
        "order_id": "late-1",
    }
    late = pd.concat([late, pd.DataFrame([extra])], ignore_index=True)
    after = point_in_time_orders_1h(late, user_id, decision_ts)
    if before != after or before != baseline:
        raise AssertionError(f"leak: before={before} after={after} baseline={baseline}")
    return True
