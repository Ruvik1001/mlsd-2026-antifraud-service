"""Один цикл переобучения: кандидат vs действующая, допуск по ограничениям."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from src.features import matrix_from_frame
from src.train import cost_at_threshold, load_json, pick_threshold, split_by_days

ROOT = Path(__file__).resolve().parents[1]


def run(data_path: Path, art_dir: Path):
    data_cfg = load_json(ROOT / "configs" / "data.json")
    train_cfg = load_json(ROOT / "configs" / "train.json")
    df = pd.read_csv(data_path)

    # расширяем train: дни 0..18 (зрелые на конце теста)
    cfg = dict(train_cfg)
    cfg["train_days"] = [0, 18]
    cfg["valid_days"] = [18, 21]
    cfg["test_days"] = [21, 23]
    train, valid, test, info = split_by_days(df, cfg)

    current = joblib.load(art_dir / "model.joblib")
    thr_cur = float(current["threshold"])

    X_tr = matrix_from_frame(train)
    y_tr = train["fraud"].to_numpy(dtype=int)
    X_va = matrix_from_frame(valid)
    y_va = valid["fraud"].to_numpy(dtype=int)
    X_te = matrix_from_frame(test)
    y_te = test["fraud"].to_numpy(dtype=int)

    cand = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=60, random_state=2027, class_weight="balanced"
    )
    cand.fit(X_tr, y_tr)
    p_va = cand.predict_proba(X_va)[:, 1]
    thr_c, st = pick_threshold(
        y_va, p_va, float(train_cfg["review_share"]), int(data_cfg["fn_cost"]), int(data_cfg["fp_cost"])
    )

    p_cur = current["tree"].predict_proba(X_te)[:, 1]
    p_new = cand.predict_proba(X_te)[:, 1]
    cur_m = cost_at_threshold(y_te, p_cur, thr_cur, int(data_cfg["fn_cost"]), int(data_cfg["fp_cost"]))
    new_m = cost_at_threshold(y_te, p_new, thr_c, int(data_cfg["fn_cost"]), int(data_cfg["fp_cost"]))
    cur_auc = float(roc_auc_score(y_te, p_cur))
    new_auc = float(roc_auc_score(y_te, p_new))

    # допуск: стоимость не хуже и review_share в бюджете, auc не просел сильно
    review_ok = new_m["reviews"] / max(len(test), 1) <= float(train_cfg["review_share"]) * 1.05
    accept = (new_m["cost"] <= cur_m["cost"]) and review_ok and (new_auc + 1e-9 >= cur_auc - 0.01)

    decision = {
        "accept": bool(accept),
        "reason": "candidate_better_or_equal" if accept else "keep_current",
        "current": {"auc": cur_auc, **cur_m},
        "candidate": {"auc": new_auc, **new_m, "threshold": thr_c},
        "split": info,
        "owner": "oncall-ml",
        "schedule": "weekly or PSI alert + mature labels ready",
        "rollback": "restore artifacts/ from artifacts/previous/",
    }

    prev = art_dir / "previous"
    prev.mkdir(parents=True, exist_ok=True)
    for name in ("model.joblib", "manifest.json"):
        src = art_dir / name
        if src.exists():
            shutil.copy2(src, prev / name)

    if accept:
        payload = dict(current)
        payload["tree"] = cand
        payload["threshold"] = thr_c
        payload["model_kind"] = "tree"
        joblib.dump(payload, art_dir / "model.joblib")
        man = json.loads((art_dir / "manifest.json").read_text(encoding="utf-8"))
        man["model_version"] = "antifraud-2026.2"
        man["threshold"] = thr_c
        man["retrain"] = decision
        (art_dir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    (ROOT / "reports" / "retrain.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"accept": accept, "cur_cost": cur_m["cost"], "new_cost": new_m["cost"]}, ensure_ascii=False))
    return decision


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
