"""Лестница моделей и сохранение артефактов."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import FEATURE_NAMES, matrix_from_frame

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_by_days(df: pd.DataFrame, cfg: dict, asof_ts: int | None = None):
    """Разбиение по дням с учётом зрелости меток."""
    train_lo, train_hi = cfg["train_days"]
    valid_lo, valid_hi = cfg["valid_days"]
    test_lo, test_hi = cfg["test_days"]

    def slice_days(lo, hi):
        part = df[(df["day"] >= lo) & (df["day"] < hi)].copy()
        if asof_ts is not None:
            mature = part["label_ready_ts"] <= asof_ts
            excluded = int((~mature).sum())
            part = part.loc[mature]
        else:
            # на «конце» окна считаем метки известными, если ready_ts попал в горизонт данных
            max_ts = int(df["event_ts"].max()) + 7 * 86400
            mature = part["label_ready_ts"] <= max_ts
            excluded = int((~mature).sum())
            part = part.loc[mature]
        return part, excluded

    train, ex_tr = slice_days(train_lo, train_hi)
    valid, ex_va = slice_days(valid_lo, valid_hi)
    test, ex_te = slice_days(test_lo, test_hi)
    info = {
        "train": {"n": len(train), "excluded_immature": ex_tr, "days": [train_lo, train_hi]},
        "valid": {"n": len(valid), "excluded_immature": ex_va, "days": [valid_lo, valid_hi]},
        "test": {"n": len(test), "excluded_immature": ex_te, "days": [test_lo, test_hi]},
        "fraud_rates": {
            "train": float(train["fraud"].mean()) if len(train) else None,
            "valid": float(valid["fraud"].mean()) if len(valid) else None,
            "test": float(test["fraud"].mean()) if len(test) else None,
        },
    }
    return train, valid, test, info


def cost_at_threshold(y, p, thr, fn_cost=8000, fp_cost=150):
    pred = (p >= thr).astype(int)
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return {
        "cost": fn * fn_cost + fp * fp_cost,
        "fn": fn,
        "fp": fp,
        "tp": tp,
        "reviews": int(pred.sum()),
        "threshold": float(thr),
    }


def pick_threshold(y, p, review_share, fn_cost=8000, fp_cost=150):
    """Порог по валидации: не больше review_share, минимум стоимости."""
    if len(y) == 0:
        return 1.0, cost_at_threshold(y, p, 1.0, fn_cost, fp_cost)
    qs = np.linspace(0.0, 1.0, 201)
    cands = np.unique(np.quantile(p, qs))
    best = None
    for thr in cands:
        m = cost_at_threshold(y, p, thr, fn_cost, fp_cost)
        if m["reviews"] / max(len(y), 1) > review_share + 1e-12:
            continue
        if best is None or m["cost"] < best["cost"]:
            best = m
    if best is None:
        thr = float(np.quantile(p, 1.0 - review_share))
        best = cost_at_threshold(y, p, thr, fn_cost, fp_cost)
    return best["threshold"], best


def train_ladder(train: pd.DataFrame, valid: pd.DataFrame, seed: int = 2026):
    X_tr = matrix_from_frame(train)
    y_tr = train["fraud"].to_numpy(dtype=int)
    X_va = matrix_from_frame(valid)
    y_va = valid["fraud"].to_numpy(dtype=int)

    # 1) константа / правило: review если amount>=20000 и age<7
    rule_p = (
        ((train["amount_rub"] >= 20000) & (train["account_age_days"] < 7)).astype(float).to_numpy()
    )
    rule_p_va = (
        ((valid["amount_rub"] >= 20000) & (valid["account_age_days"] < 7)).astype(float).to_numpy()
    )

    # 2) линейная
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    lin = LogisticRegression(C=1.0, max_iter=400, random_state=seed, class_weight="balanced")
    lin.fit(X_tr_s, y_tr)
    lin_p_va = lin.predict_proba(X_va_s)[:, 1]

    # 3) дерево — ловит взаимодействия orders_1h * device_risk
    tree = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=80, random_state=seed, class_weight="balanced"
    )
    tree.fit(X_tr, y_tr)
    tree_p_va = tree.predict_proba(X_va)[:, 1]

    ladder = {
        "rule": {
            "valid_auc": float(roc_auc_score(y_va, rule_p_va)) if y_va.sum() and (1 - y_va).sum() else None,
            "note": "жёсткое правило amount/age; не видит device_risk*orders_1h",
        },
        "linear": {
            "valid_auc": float(roc_auc_score(y_va, lin_p_va)),
            "note": "учитывает все признаки линейно; слабее на взаимодействии",
        },
        "tree": {
            "valid_auc": float(roc_auc_score(y_va, tree_p_va)),
            "note": "глубина 4 закрывает взаимодействие частоты и риска устройства",
        },
    }
    return {
        "scaler": scaler,
        "linear": lin,
        "tree": tree,
        "ladder": ladder,
        "valid_probs": {"rule": rule_p_va, "linear": lin_p_va, "tree": tree_p_va},
        "y_valid": y_va,
    }


def save_artifacts(bundle, thr: float, out_dir: Path, split_info: dict, serve_cfg: dict, data_meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": bundle["scaler"],
            "linear": bundle["linear"],
            "tree": bundle["tree"],
            "feature_names": FEATURE_NAMES,
            "threshold": thr,
            "model_kind": "tree",
        },
        out_dir / "model.joblib",
    )
    # компактный JSON для сервиса без sklearn на горячем пути (дерево через joblib)
    manifest = {
        "model_version": serve_cfg.get("model_version", "antifraud-2026.1"),
        "feature_names": FEATURE_NAMES,
        "threshold": thr,
        "model_kind": "tree",
        "split": split_info,
        "ladder": bundle["ladder"],
        "data_sha256": data_meta.get("sha256"),
        "data_n": data_meta.get("n"),
        "code_id": "lab_7_62-antifraud-v1",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def run(data_path: Path, out_dir: Path, quick: bool = False):
    data_cfg = load_json(ROOT / "configs" / "data.json")
    train_cfg = load_json(ROOT / "configs" / "train.json")
    serve_cfg = load_json(ROOT / "configs" / "serve.json")

    if not data_path.exists():
        from data.generate import generate

        generate(data_cfg, ROOT / "data", quick=quick)
        data_path = ROOT / "data" / ("orders_quick.csv" if quick else "orders.csv")

    df = pd.read_csv(data_path)
    meta_path = data_path.with_name("meta_quick.json" if "quick" in data_path.name else "meta.json")
    data_meta = load_json(meta_path) if meta_path.exists() else {}

    train, valid, test, info = split_by_days(df, train_cfg)
    bundle = train_ladder(train, valid, seed=int(train_cfg["seed"]))
    thr, thr_stats = pick_threshold(
        bundle["y_valid"],
        bundle["valid_probs"]["tree"],
        review_share=float(train_cfg["review_share"]),
        fn_cost=int(data_cfg["fn_cost"]),
        fp_cost=int(data_cfg["fp_cost"]),
    )
    manifest = save_artifacts(bundle, thr, out_dir, info, serve_cfg, data_meta)
    manifest["valid_threshold_stats"] = thr_stats
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # быстрый отчёт по валидации
    report = {
        "split": info,
        "ladder": bundle["ladder"],
        "threshold": thr,
        "valid_cost": thr_stats,
        "data_sha256": data_meta.get("sha256"),
    }
    (ROOT / "reports" / "train_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"threshold": thr, "valid_auc_tree": bundle["ladder"]["tree"]["valid_auc"]}, ensure_ascii=False))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    data = Path(args.data) if args.data else ROOT / "data" / ("orders_quick.csv" if args.quick else "orders.csv")
    run(data, Path(args.out), quick=args.quick)


if __name__ == "__main__":
    main()
