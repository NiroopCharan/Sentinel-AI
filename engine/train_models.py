"""Train the network vectors (RF, AS), tune tau, and - if their self-collected CSVs exist - fit RDS/ICS.

    python -m engine.train_models                      # real NSL-KDD (run scripts/download_nsl_kdd.sh first)
    python -m engine.train_models --allow-synthetic    # smoke test only; models are stamped trained_on="synthetic"

Outputs in models/: preprocessor.joblib, rf.joblib, iso.joblib, meta.json (+ rds_weights.json / identity_model.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, DEFAULT_MARGIN, INITIAL_WEIGHTS, MODELS_DIR  # noqa: E402
from engine.data_loader import (build_preprocessor, class_balance, feature_columns, load_train_test,  # noqa: E402
                                split_train_val)
from engine.fusion import fuse, normalize_anomaly  # noqa: E402
from utils.log import logger  # noqa: E402


def ts_network_only(rf_p, as_s):
    """Fused TS from the two vectors that exist for NSL-KDD rows (weights renormalised over rf+as)."""
    out = np.empty(len(rf_p))
    for i, (a, b) in enumerate(zip(rf_p, as_s)):
        out[i] = fuse({"rf": a, "as": b, "rds": None, "ics": None}, INITIAL_WEIGHTS)[0]
    return out


def tune_tau(ts: np.ndarray, y: np.ndarray):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [f1_score(y, (ts >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(f1s))
    return float(grid[best]), float(f1s[best])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["core", "full"], default="core",
                    help="core = 7 live-computable features + protocol (default, deployable); full = all 41")
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    args = ap.parse_args(argv)
    out = Path(args.models_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, _test, provenance = load_train_test(args.allow_synthetic)
    if provenance == "synthetic":
        logger.warning("Training on SYNTHETIC data - models are stamped trained_on='synthetic' and must not be reported as results.")
    fit_df, val_df = split_train_val(train, 0.2, args.seed)
    cols = feature_columns(args.feature_set)
    logger.info(f"train-fit: {class_balance(fit_df)}  train-val: {class_balance(val_df)}  features: {args.feature_set}")

    pre = build_preprocessor(args.feature_set).fit(fit_df[cols])
    X_fit, X_val = pre.transform(fit_df[cols]), pre.transform(val_df[cols])

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=args.seed).fit(X_fit, fit_df["y"])

    iso = IsolationForest(n_estimators=200, random_state=args.seed, n_jobs=-1).fit(X_fit[fit_df["y"].to_numpy() == 0])
    neg_normal = -iso.score_samples(X_fit[fit_df["y"].to_numpy() == 0])
    as_min, as_max = float(np.percentile(neg_normal, 1)), float(np.percentile(neg_normal, 99))

    rf_val = rf.predict_proba(X_val)[:, 1]
    as_val = normalize_anomaly(iso.score_samples(X_val), as_min, as_max)
    tau, f1 = tune_tau(ts_network_only(rf_val, as_val), val_df["y"].to_numpy())

    meta = {
        "trained_on": provenance, "feature_set": args.feature_set, "seed": args.seed,
        "as_min": as_min, "as_max": as_max, "tau": tau, "margin": DEFAULT_MARGIN,
        "tau_val_f1": f1, "tau_note": "F1-optimal on the held-out 20% of the TRAIN partition (RF+AS fusion only)",
        "n_fit": len(fit_df), "n_val": len(val_df),
        "class_balance_fit": class_balance(fit_df), "class_balance_val": class_balance(val_df),
        "sklearn_version": sklearn.__version__, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    joblib.dump(pre, out / "preprocessor.joblib")
    joblib.dump(rf, out / "rf.joblib")
    joblib.dump(iso, out / "iso.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Saved models to {out}. tau={tau:.2f} (val F1={f1:.4f}), margin={DEFAULT_MARGIN}")

    # Vectors that need self-collected data - trained only if that data really exists.
    res_csv, id_csv = DATA_DIR / "resource_baseline.csv", DATA_DIR / "identity_sessions.csv"
    if res_csv.exists():
        from engine.resource_model import fit_weights
        fit_weights(res_csv, out / "rds_weights.json")
    else:
        logger.info("RDS: data/resource_baseline.csv not found - run utils/collect_resource_baseline.py (equal weights until then).")
    if id_csv.exists():
        from engine.identity_model import train as train_ics
        train_ics(id_csv, out / "identity_model.json")
    else:
        logger.info("ICS: data/identity_sessions.csv not found - collect owner/intruder sessions (utils/session_logger.py).")
    return meta


if __name__ == "__main__":
    main()
