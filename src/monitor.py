"""Мониторинг: PSI/дрейф сразу, качество — после зрелости меток."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train import cost_at_threshold, load_json

ROOT = Path(__file__).resolve().parents[1]


def psi(ref, cur, bins=10):
    r = np.asarray(ref, dtype=float)
    c = np.asarray(cur, dtype=float)
    qs = np.arange(1, bins) / bins
    edges = np.unique(np.quantile(r, qs))
    cuts = np.concatenate([[-np.inf], edges, [np.inf]])
    a = np.histogram(r, bins=cuts)[0].astype(float)
    b = np.histogram(c, bins=cuts)[0].astype(float)
    k = len(a)
    p = (a + 0.5) / (a.sum() + 0.5 * k)
    q = (b + 0.5) / (b.sum() + 0.5 * k)
    return float(np.sum((q - p) * np.log(q / p)))


def delayed_quality(preds: pd.DataFrame, labels: pd.DataFrame, asof_ts: int, thr: float):
    """Считаем метрики только по заказам с готовой меткой на asof_ts."""
    m = preds.merge(labels, on="order_id", how="left")
    ready = m["label_ready_ts"] <= asof_ts
    mature = m.loc[ready & m["fraud"].notna()].copy()
    immature_share = float((~ready).mean()) if len(m) else 0.0
    if len(mature) == 0:
        return {"n_mature": 0, "immature_share": immature_share, "cost": None, "auc_proxy_reviews": None}
    y = mature["fraud"].to_numpy(dtype=int)
    p = mature["score"].to_numpy(dtype=float)
    c = cost_at_threshold(y, p, thr)
    return {
        "n_mature": int(len(mature)),
        "immature_share": immature_share,
        "ready_share": float(ready.mean()),
        **c,
    }


def run(data_path: Path, inject_shift: bool = True):
    cfg = load_json(ROOT / "configs" / "release.json")
    train_cfg = load_json(ROOT / "configs" / "train.json")
    data_cfg = load_json(ROOT / "configs" / "data.json")
    df = pd.read_csv(data_path)

    ref_lo, ref_hi = train_cfg["train_days"]
    cur_lo, cur_hi = train_cfg["replay_days"]
    ref = df[(df["day"] >= ref_lo) & (df["day"] < ref_hi)]
    cur = df[(df["day"] >= cur_lo) & (df["day"] < cur_hi)].copy()

    incident = []
    if inject_shift:
        # поломка контракта: device_risk «залип» + сдвиг amount
        n = len(cur)
        cur.loc[cur.index[: n // 3], "device_risk"] = 0.99
        cur.loc[cur.index[n // 3 : 2 * n // 3], "amount_rub"] = cur.loc[
            cur.index[n // 3 : 2 * n // 3], "amount_rub"
        ] * 2.5
        incident.append(
            {
                "type": "contract_break_and_mix_shift",
                "device_risk_stuck_share": 1 / 3,
                "amount_boost_share": 1 / 3,
            }
        )

    features = ["amount_rub", "account_age_days", "orders_1h", "device_risk"]
    drift = {}
    alerts = []
    for f in features:
        val = psi(ref[f], cur[f])
        drift[f] = val
        if val >= float(cfg["psi_alert"]):
            alerts.append({"feature": f, "psi": val, "action": "investigate_data_before_quality"})

    # имитация скоров: выше amount/device → выше score
    score = (
        0.00001 * cur["amount_rub"].to_numpy()
        + 0.8 * cur["device_risk"].to_numpy()
        + 0.05 * cur["orders_1h"].to_numpy()
    )
    score = 1 / (1 + np.exp(-(score - 0.5)))
    preds = pd.DataFrame(
        {
            "order_id": cur["order_id"],
            "score": score,
            "event_ts": cur["event_ts"],
            "day": cur["day"],
        }
    )
    labels = cur[["order_id", "fraud", "label_ready_ts"]]

    # asof: середина replay + ещё немного
    asof = int(cur["event_ts"].median()) + 3 * 86400
    thr = 0.5
    man = ROOT / "artifacts" / "manifest.json"
    if man.exists():
        thr = float(json.loads(man.read_text(encoding="utf-8"))["threshold"])

    q = delayed_quality(preds, labels, asof, thr)
    # объяснение пропусков: в реальности метки чаще у проверенных
    selection_note = (
        "В проде метки богаче у review-заказов; оценка FN по непроверенным approve "
        "смещена вниз. Здесь учебная проверка раскрывает все метки через 7 дней."
    )

    report = {
        "reference_days": [ref_lo, ref_hi],
        "current_days": [cur_lo, cur_hi],
        "psi": drift,
        "alerts": alerts,
        "incident": incident,
        "delayed_quality": q,
        "selection_bias_note": selection_note,
        "first_signal": alerts[0] if alerts else None,
        "n_obs_for_alert": int(len(cur)),
    }
    (ROOT / "reports" / "monitor.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"alerts": len(alerts), "psi_device_risk": drift.get("device_risk")}, ensure_ascii=False))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    data = Path(args.data) if args.data else ROOT / "data" / ("orders_quick.csv" if args.quick else "orders.csv")
    run(data)


if __name__ == "__main__":
    main()
