"""Shadow → canary → rollback на воспроизводимом потоке."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import matrix_from_frame
from src.train import load_json

ROOT = Path(__file__).resolve().parents[1]


def route(request_id: str, share: float) -> str:
    digest = hashlib.sha256(request_id.encode("utf-8")).digest()[:8]
    bucket = int.from_bytes(digest, "big")
    return "canary" if bucket < share * (2**64) else "stable"


def window_stats(rows: list[dict]):
    if not rows:
        return {"n": 0, "error_rate": None, "p95_ms": None}
    n = len(rows)
    errors = sum(1 for r in rows if r["error"])
    lat = np.asarray([r["latency_ms"] for r in rows], dtype=float)
    return {
        "n": n,
        "error_rate": errors / n,
        "p95_ms": float(np.quantile(lat, 0.95)),
        "errors": errors,
    }


def simulate(df: pd.DataFrame, art_dir: Path, cfg: dict, inject_fault: bool = True):
    bundle = joblib.load(art_dir / "model.joblib")
    X = matrix_from_frame(df)
    p_good = bundle["tree"].predict_proba(X)[:, 1]
    thr = float(bundle["threshold"])

    # «плохая» канарейка: ломаем порог и добавляем ошибки/задержку
    p_bad = np.clip(p_good * 0.1, 0, 1)

    events = []
    journal = []
    phase = "shadow"
    canary_share = 0.0
    bad_windows = 0
    served_version = "v1"
    window = []
    window_id = 0

    n_rows = len(df)
    win_size = max(int(cfg.get("min_requests_per_window", 200)), 200)
    min_canary_n = max(20, win_size // 20)

    for i, (_, row) in enumerate(df.iterrows()):
        rid = f"req-{row['order_id']}"
        frac = i / max(n_rows - 1, 1)
        if phase == "shadow" and frac > 0.15:
            phase = "canary"
            canary_share = float(cfg["canary_share"])
            served_version = "v1+v2"
            journal.append({"t": int(row["event_ts"]), "action": "start_canary", "share": canary_share})
        assigned = route(rid, canary_share) if phase == "canary" and canary_share > 0 else "stable"

        error = 0
        latency = 8.0 + float(np.random.default_rng(i).random()) * 4.0
        if assigned == "canary" and inject_fault and frac > 0.25:
            # технический сбой канарейки (~15% ошибок + высокая задержка)
            if np.random.default_rng(i + 7).random() < 0.15:
                error = 1
                latency = 150.0
            p = float(p_bad[i])
            version = "v2"
        else:
            p = float(p_good[i])
            version = "v1"
            if phase == "shadow":
                _ = float(p_bad[i])

        decision = "review" if (p >= thr and error == 0) else ("approve" if error == 0 else "error")
        ev = {
            "request_id": rid,
            "order_id": row["order_id"],
            "phase": phase,
            "assigned": assigned,
            "served_version": version,
            "error": error,
            "latency_ms": latency,
            "score": p,
            "decision": decision,
            "event_ts": int(row["event_ts"]),
        }
        events.append(ev)
        window.append(ev)

        if len(window) >= win_size or i == n_rows - 1:
            stats = window_stats(window)
            stats["window_id"] = window_id
            stats["phase"] = phase
            canary_rows = [r for r in window if r["served_version"] == "v2"]
            cstats = window_stats(canary_rows)
            rollback = False
            if (
                phase == "canary"
                and cstats["n"] >= min_canary_n
                and cstats["error_rate"] is not None
                and cstats["error_rate"] > float(cfg["error_rate_limit"])
            ):
                bad_windows += 1
            elif cstats["n"] >= min_canary_n:
                bad_windows = 0
            if bad_windows >= int(cfg["consecutive_bad_windows"]):
                rollback = True
                phase = "rolled_back"
                canary_share = 0.0
                served_version = "v1"
                journal.append(
                    {
                        "t": int(row["event_ts"]),
                        "action": "rollback",
                        "reason": "error_rate",
                        "window_id": window_id,
                        "canary_error_rate": cstats["error_rate"],
                        "canary_n": cstats["n"],
                    }
                )
                bad_windows = 0
            if (
                phase == "canary"
                and cstats["p95_ms"] is not None
                and cstats["n"] >= min_canary_n
                and cstats["p95_ms"] > float(cfg["latency_p95_limit_ms"])
            ):
                phase = "rolled_back"
                canary_share = 0.0
                journal.append(
                    {
                        "t": int(row["event_ts"]),
                        "action": "rollback",
                        "reason": "latency_p95",
                        "window_id": window_id,
                        "canary_p95_ms": cstats["p95_ms"],
                    }
                )
            journal.append(
                {
                    "t": int(row["event_ts"]),
                    "action": "window",
                    "stats": stats,
                    "canary": cstats,
                    "rollback": rollback,
                }
            )
            window = []
            window_id += 1

    return events, journal


def run(data_path: Path, art_dir: Path, quick: bool = False):
    cfg = load_json(ROOT / "configs" / "release.json")
    train_cfg = load_json(ROOT / "configs" / "train.json")
    df = pd.read_csv(data_path)
    lo, hi = train_cfg["replay_days"]
    stream = df[(df["day"] >= lo) & (df["day"] < hi)].reset_index(drop=True)
    if quick:
        stream = stream.head(3000)
    # для демо отката уменьшаем порог окна
    cfg = dict(cfg)
    cfg["min_requests_per_window"] = 200

    events, journal = simulate(stream, art_dir, cfg, inject_fault=True)
    (ROOT / "reports" / "replay_events.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8"
    )
    (ROOT / "reports" / "release_journal.json").write_text(
        json.dumps(journal, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rollbacks = [j for j in journal if j.get("action") == "rollback"]
    summary = {
        "n_events": len(events),
        "n_rollbacks": len(rollbacks),
        "rollbacks": rollbacks,
        "final_phase": events[-1]["phase"] if events else None,
    }
    (ROOT / "reports" / "replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    data = Path(args.data) if args.data else ROOT / "data" / ("orders_quick.csv" if args.quick else "orders.csv")
    run(data, Path(args.artifacts), quick=args.quick)


if __name__ == "__main__":
    main()
