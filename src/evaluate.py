"""Офлайн-оценка: лестница, бутстрэп разности, срезы, калибровка, стоимость."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.features import matrix_from_frame
from src.train import cost_at_threshold, load_json, split_by_days

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_cost_diff(y, p_base, p_cand, thr_base, thr_cand, day, n_iter=200, seed=2026, fn=8000, fp=150):
    """Бутстрэп по дневным блокам: Δcost = cost_cand - cost_base."""
    rng = np.random.default_rng(seed)
    days = np.unique(day)
    diffs = []
    for _ in range(n_iter):
        pick = rng.choice(days, size=len(days), replace=True)
        mask = np.isin(day, pick)
        if mask.sum() == 0 or y[mask].sum() == 0:
            continue
        cb = cost_at_threshold(y[mask], p_base[mask], thr_base, fn, fp)["cost"]
        cc = cost_at_threshold(y[mask], p_cand[mask], thr_cand, fn, fp)["cost"]
        diffs.append(cc - cb)
    if not diffs:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    arr = np.asarray(diffs, dtype=float)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(arr, 0.025)),
        "hi": float(np.quantile(arr, 0.975)),
        "n": int(len(arr)),
    }


def slice_metrics(df, p, thr, name, mask):
    part = df.loc[mask]
    if len(part) == 0:
        return {"slice": name, "n": 0}
    y = part["fraud"].to_numpy(dtype=int)
    pp = p[mask.to_numpy()]
    c = cost_at_threshold(y, pp, thr)
    auc = float(roc_auc_score(y, pp)) if y.sum() and (1 - y).sum() else None
    return {
        "slice": name,
        "n": int(len(part)),
        "positives": int(y.sum()),
        "auc": auc,
        "reviews": c["reviews"],
        "cost": c["cost"],
        "fn": c["fn"],
        "fp": c["fp"],
    }


def run(data_path: Path, art_dir: Path):
    data_cfg = load_json(ROOT / "configs" / "data.json")
    train_cfg = load_json(ROOT / "configs" / "train.json")
    df = pd.read_csv(data_path)
    train, valid, test, info = split_by_days(df, train_cfg)
    bundle = joblib.load(art_dir / "model.joblib")
    thr = float(bundle["threshold"])

    X_te = matrix_from_frame(test)
    y = test["fraud"].to_numpy(dtype=int)
    Xs = bundle["scaler"].transform(X_te)
    p_lin = bundle["linear"].predict_proba(Xs)[:, 1]
    p_tree = bundle["tree"].predict_proba(X_te)[:, 1]
    p_rule = ((test["amount_rub"] >= 20000) & (test["account_age_days"] < 7)).astype(float).to_numpy()

    # калибратор на валидации (отдельная часть)
    X_va = matrix_from_frame(valid)
    p_va = bundle["tree"].predict_proba(X_va)[:, 1]
    cal = LogisticRegression(C=1.0, max_iter=200)
    cal.fit(p_va.reshape(-1, 1), valid["fraud"].to_numpy(dtype=int))
    p_cal = cal.predict_proba(p_tree.reshape(-1, 1))[:, 1]

    metrics = {
        "test_n": int(len(test)),
        "test_positives": int(y.sum()),
        "split": info,
        "models": {
            "rule": {"auc": float(roc_auc_score(y, p_rule)), **cost_at_threshold(y, p_rule, 0.5)},
            "linear": {"auc": float(roc_auc_score(y, p_lin)), **cost_at_threshold(y, p_lin, thr)},
            "tree": {"auc": float(roc_auc_score(y, p_tree)), **cost_at_threshold(y, p_tree, thr)},
            "tree_calibrated": {
                "auc": float(roc_auc_score(y, p_cal)),
                "brier": float(brier_score_loss(y, p_cal)),
                **cost_at_threshold(y, p_cal, thr),
            },
        },
        "delta_cost_tree_minus_rule": bootstrap_cost_diff(
            y,
            p_rule,
            p_tree,
            0.5,
            thr,
            test["day"].to_numpy(),
            n_iter=int(train_cfg["bootstrap_iters"]),
            seed=int(train_cfg["seed"]),
            fn=int(data_cfg["fn_cost"]),
            fp=int(data_cfg["fp_cost"]),
        ),
    }

    # калибровочная кривая
    frac_pos, mean_pred = calibration_curve(y, p_cal, n_bins=8, strategy="quantile")
    metrics["calibration"] = {
        "fraction_positives": [float(x) for x in frac_pos],
        "mean_predicted": [float(x) for x in mean_pred],
        "brier": float(brier_score_loss(y, p_cal)),
    }

    # срезы: новые аккаунты
    mask_new = test["account_age_days"] < 30
    mask_old = test["account_age_days"] >= 30
    mask_high = test["device_risk"] >= 0.5
    metrics["slices"] = [
        slice_metrics(test, p_tree, thr, "new_accounts_lt_30d", mask_new),
        slice_metrics(test, p_tree, thr, "old_accounts_ge_30d", mask_old),
        slice_metrics(test, p_tree, thr, "high_device_risk", mask_high),
    ]

    no_check = int(y.sum()) * int(data_cfg["fn_cost"])
    chosen = metrics["models"]["tree"]
    metrics["business"] = {
        "no_checks_loss": no_check,
        "with_model_cost": chosen["cost"],
        "reviews": chosen["reviews"],
        "review_share": chosen["reviews"] / max(len(test), 1),
        "fn_cost": data_cfg["fn_cost"],
        "fp_cost": data_cfg["fp_cost"],
        "threshold": thr,
    }

    out = ROOT / "reports" / "offline_eval.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(cal, art_dir / "calibrator.joblib")
    print(json.dumps({"test_auc_tree": metrics["models"]["tree"]["auc"], "cost": chosen["cost"]}, ensure_ascii=False))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    data = Path(args.data) if args.data else ROOT / "data" / ("orders_quick.csv" if args.quick else "orders.csv")
    run(data, Path(args.artifacts))


if __name__ == "__main__":
    main()
