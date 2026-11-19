import pandas as pd

from src.features import assert_no_future_leak, point_in_time_orders_1h


def test_late_event_does_not_change_feature():
    hist = pd.DataFrame(
        [
            {"user_id": "u1", "event_ts": 1000, "ingest_ts": 1010, "order_id": "a"},
            {"user_id": "u1", "event_ts": 1500, "ingest_ts": 1510, "order_id": "b"},
            {"user_id": "u1", "event_ts": 2000, "ingest_ts": 2500, "order_id": "c"},  # ingest после decision
        ]
    )
    decision_ts = 2100
    base = point_in_time_orders_1h(hist, "u1", decision_ts)
    # c не должно войти: ingest_ts > decision
    assert base == 2.0
    assert assert_no_future_leak(hist, "u1", decision_ts, base)


def test_ingest_after_decision_excluded():
    hist = pd.DataFrame(
        [{"user_id": "u1", "event_ts": 100, "ingest_ts": 500, "order_id": "x"}]
    )
    assert point_in_time_orders_1h(hist, "u1", 200) == 0.0
