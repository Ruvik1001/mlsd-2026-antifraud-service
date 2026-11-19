"""Синтетический генератор заказов антифрода, seed 2026."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_cfg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _calibrate_intercept(rng, n, base_logit_fn, target_rate, lo=-12.0, hi=2.0, steps=40):
    """Подбор свободного члена так, чтобы E[fraud] ≈ target_rate на обучающем периоде."""
    x_demo = _sample_raw(rng, n)
    base = base_logit_fn(x_demo)
    best_b, best_err = 0.0, 1.0
    for i in range(steps):
        b = lo + (hi - lo) * i / (steps - 1)
        p = 1.0 / (1.0 + np.exp(-(base + b)))
        err = abs(float(p.mean()) - target_rate)
        if err < best_err:
            best_err, best_b = err, b
    return float(best_b)


def _sample_raw(rng, n):
    amount = rng.lognormal(mean=8.2, sigma=0.7, size=n).clip(200, 250_000)
    account_age = rng.integers(0, 1800, size=n).astype(float)
    orders_1h = rng.poisson(0.4, size=n).astype(float)
    # редкие всплески частоты
    burst = rng.random(n) < 0.03
    orders_1h = orders_1h + burst * rng.integers(3, 12, size=n)
    device_risk = rng.beta(1.4, 6.0, size=n)
    pay = rng.choice(["card", "wallet", "cod"], size=n, p=[0.55, 0.30, 0.15])
    return {
        "amount_rub": amount,
        "account_age_days": account_age,
        "orders_1h": orders_1h.astype(float),
        "device_risk": device_risk,
        "payment_method": pay,
    }


def _base_logit(feats):
    amount = feats["amount_rub"]
    age = feats["account_age_days"]
    oh = feats["orders_1h"]
    dr = feats["device_risk"]
    pay = feats["payment_method"]
    pay_cod = (pay == "cod").astype(float)
    # нарушение вероятнее при частых заказах + высоком риске устройства
    return (
        0.000015 * amount
        - 0.0025 * age
        + 0.55 * oh
        + 3.2 * dr
        + 0.8 * (oh * dr)
        + 0.35 * pay_cod
    )


def generate(cfg: dict, out_dir: Path, quick: bool = False) -> pd.DataFrame:
    seed = int(cfg["seed"])
    n = 20_000 if quick else int(cfg["n_orders"])
    days = int(cfg["days"])
    delay = int(cfg["label_delay_days"])
    rng = np.random.default_rng(seed)

    # калибруем intercept на «обучающем» куске (первые 14 дней ≈ 14/30)
    n_cal = max(5000, n // 3)
    intercept = _calibrate_intercept(
        rng, n_cal, _base_logit, float(cfg["target_fraud_rate_train"])
    )

    rng = np.random.default_rng(seed)
    feats = _sample_raw(rng, n)
    logit = _base_logit(feats) + intercept
    # скрытая переменная — только для проверки качества
    p_fraud = 1.0 / (1.0 + np.exp(-logit))
    fraud = rng.random(n) < p_fraud

    # время: равномерно по 30 дням, event_ts = decision moment
    day_idx = rng.integers(0, days, size=n)
    second_in_day = rng.integers(0, 24 * 3600, size=n)
    event_ts = day_idx * 86400 + second_in_day
    # поступление события: 0..120 с задержки телеметрии
    ingest_lag = rng.integers(0, 120, size=n)
    ingest_ts = event_ts + ingest_lag
    # метка готова через 7 суток после события
    label_ready_ts = event_ts + delay * 86400

    order_ids = [f"ord-{seed}-{i:06d}" for i in range(n)]
    user_ids = [f"u-{(i * 97 + seed) % (n // 2 + 1):06d}" for i in range(n)]

    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "user_id": user_ids,
            "day": day_idx,
            "event_ts": event_ts,
            "ingest_ts": ingest_ts,
            "label_ready_ts": label_ready_ts,
            "amount_rub": feats["amount_rub"],
            "account_age_days": feats["account_age_days"],
            "orders_1h": feats["orders_1h"],
            "device_risk": feats["device_risk"],
            "payment_method": feats["payment_method"],
            "fraud": fraud.astype(int),
            "p_fraud_hidden": p_fraud,
            "intercept": intercept,
        }
    )
    df = df.sort_values("event_ts").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("orders_quick.csv" if quick else "orders.csv")
    df.to_csv(path, index=False)
    meta = {
        "seed": seed,
        "n": int(n),
        "intercept": intercept,
        "fraud_rate": float(df["fraud"].mean()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path.name),
        "quick": quick,
    }
    (out_dir / ("meta_quick.json" if quick else "meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.json"))
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = _load_cfg(Path(args.config))
    df = generate(cfg, Path(args.out), quick=args.quick)
    print(f"rows={len(df)} fraud_rate={df['fraud'].mean():.4f}")


if __name__ == "__main__":
    main()
