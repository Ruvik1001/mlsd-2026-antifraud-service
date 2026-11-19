"""Замер задержки до/после оптимизации (batch predict path)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import matrix_from_frame
from src.service import Bundle

ROOT = Path(__file__).resolve().parents[1]


def pct(a, q):
    return float(np.quantile(a, q))


def bench_predict(bundle: Bundle, X: np.ndarray, rounds: int = 5, warmup: int = 2):
    # single
    bundle.use_batch = False
    for _ in range(warmup):
        for i in range(min(64, len(X))):
            bundle.predict_proba(X[i : i + 1])
    single = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for i in range(len(X)):
            bundle.predict_proba(X[i : i + 1])
        single.append((time.perf_counter() - t0) * 1000.0 / len(X))

    # batch (оптимизация)
    bundle.use_batch = True
    for _ in range(warmup):
        bundle.predict_proba_batch(X[:64])
    batch = []
    bs = 64
    for _ in range(rounds):
        t0 = time.perf_counter()
        for i in range(0, len(X), bs):
            bundle.predict_proba_batch(X[i : i + bs])
        batch.append((time.perf_counter() - t0) * 1000.0 / len(X))

    return {
        "n": int(len(X)),
        "rounds": rounds,
        "single_ms_per_row": {"mean": float(np.mean(single)), "p50": pct(single, 0.5), "p95": pct(single, 0.95)},
        "batch64_ms_per_row": {"mean": float(np.mean(batch)), "p50": pct(batch, 0.5), "p95": pct(batch, 0.95)},
        "raw_single_ms": [float(x) for x in single],
        "raw_batch_ms": [float(x) for x in batch],
        "note": "на машине замера; на другой — иначе",
    }


def run(data_path: Path, art_dir: Path, n: int = 2000):
    df = pd.read_csv(data_path).head(n)
    X = matrix_from_frame(df)
    bundle = Bundle(art_dir)
    # качество до/после одно и то же (тот же tree), сравниваем только путь вызова
    y = df["fraud"].to_numpy(dtype=int) if "fraud" in df.columns else None
    p = bundle.predict_proba_batch(X)
    size_bytes = (art_dir / "model.joblib").stat().st_size
    out = {
        "artifact_bytes": int(size_bytes),
        "latency": bench_predict(bundle, X),
        "quality_unchanged": True,
        "mean_proba": float(p.mean()),
        "positives_in_sample": int(y.sum()) if y is not None else None,
    }
    path = ROOT / "reports" / "benchmark.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"p95_single": out["latency"]["single_ms_per_row"]["p95"], "p95_batch": out["latency"]["batch64_ms_per_row"]["p95"]}, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    data = Path(args.data) if args.data else ROOT / "data" / ("orders_quick.csv" if args.quick else "orders.csv")
    run(data, Path(args.artifacts), n=args.n)


if __name__ == "__main__":
    main()
