"""Benign-only anomaly baselines for the LID-DS window CSV.

Training is fit on training benign windows only. Thresholds are selected from
benign validation windows at a predeclared target false-positive rate. Test is
scored once and never used for fitting or threshold selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


META_COLUMNS = {
    "recording_id", "split", "category", "exploit", "window_start",
    "window_end", "label", "label_source", "features_json",
}


def load_csv(path: str):
    rows = []
    feature_names = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "":
                continue
            features = json.loads(row["features_json"])
            feature_names.update(features)
            rows.append({
                **row,
                "label_int": int(row["label"]),
                "window_start_float": float(row["window_start"]),
                "exploit_time": row.get("exploit_time", ""),
                "features": features,
            })
    if not rows:
        raise ValueError("CSV contains no labelled windows")
    names = sorted(feature_names)
    X = np.zeros((len(rows), len(names)), dtype=float)
    for i, row in enumerate(rows):
        for j, name in enumerate(names):
            value = row["features"].get(name, 0.0)
            X[i, j] = float(value)
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values")
    # Count/rate features are nonnegative. log1p reduces heavy-tailed syscall
    # counts while preserving zero. Latency is nonnegative too.
    if np.any(X < 0):
        raise ValueError("Expected nonnegative LID-DS feature values")
    X = np.log1p(X)
    return rows, X, names


def _subset(rows, X, split, require_benign=False):
    idx = np.array([i for i, r in enumerate(rows)
                    if r["split"] == split and
                    (not require_benign or r["label_int"] == 0)], dtype=int)
    if len(idx) == 0:
        raise ValueError(f"No rows available for split={split!r}")
    return idx, X[idx]


def _higher_quantile_threshold(scores, target_fpr):
    """Finite-sample order-statistic threshold; alarms use strict score >."""
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1")
    s = np.sort(np.asarray(scores, dtype=float))
    if len(s) == 0 or not np.isfinite(s).all():
        raise ValueError("Calibration scores must be non-empty and finite")
    # Choose a conservative order statistic so the calibration exceedance
    # fraction does not exceed the requested rate (ties reduce it further).
    rank = int(math.ceil((len(s) + 1) * (1.0 - target_fpr)))
    if rank > len(s):
        return float("inf")
    return float(s[rank - 1])


def _metrics(y, pred, scores):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {
        "n": int(len(y)), "benign": int(np.sum(y == 0)),
        "attack": int(np.sum(y == 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "attack_recall": float(recall_score(y, pred, zero_division=0)) if np.any(y == 1) else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, scores))
        result["auprc"] = float(average_precision_score(y, scores))
    else:
        result["auroc"] = None
        result["auprc"] = None
    return result


def _recording_metrics(rows, indices, pred):
    """Evaluate attack phases only after their exact exploit marker."""
    grouped = defaultdict(list)
    for local_i, global_i in enumerate(indices):
        grouped[rows[global_i]["recording_id"]].append(
            (rows[global_i], int(pred[local_i]))
        )

    benign_fp = attack_tp = benign_n = attack_n = 0
    delays = []
    for _, items in grouped.items():
        is_attack_recording = any(
            item["category"] == "normal_and_attack"
            and item["exploit"].strip().lower() == "true"
            for item, _ in items
        )
        if is_attack_recording:
            attack_n += 1
            post_marker_alarms = [
                item for item, alarm in items
                if alarm and item["label_int"] == 1
            ]
            if post_marker_alarms:
                attack_tp += 1
                exploit_times = [
                    float(item["exploit_time"])
                    for item, _ in items
                    if item.get("exploit_time") not in (None, "", "None")
                ]
                if exploit_times:
                    first_alarm = min(
                        item["window_start_float"] for item in post_marker_alarms
                    )
                    delays.append(max(0.0, first_alarm - min(exploit_times)))
        else:
            benign_n += 1
            if any(alarm for _, alarm in items):
                benign_fp += 1

    return {
        "recordings": int(len(grouped)),
        "benign_recordings": int(benign_n),
        "benign_recording_false_positive_rate": (
            float(benign_fp / benign_n) if benign_n else None
        ),
        "attack_recordings": int(attack_n),
        "attack_recording_detection_rate": (
            float(attack_tp / attack_n) if attack_n else None
        ),
        "median_detection_delay_from_exploit_seconds": (
            float(np.median(delays)) if delays else None
        ),
        "detected_attack_recordings": int(attack_tp),
        "undetected_attack_recordings": int(attack_n - attack_tp),
    }


def run_benchmark(csv_path: str, output_path: str, target_fpr: float = 0.01,
                  seed: int = 20261115):
    rows, X, feature_names = load_csv(csv_path)
    train_idx, X_train = _subset(rows, X, "training", require_benign=True)
    val_idx, X_val = _subset(rows, X, "validation", require_benign=True)
    test_idx, X_test = _subset(rows, X, "test", require_benign=False)
    y_test = np.array([rows[i]["label_int"] for i in test_idx], dtype=int)

    # Feature preprocessing is fitted on training only.
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    Xtr = preprocessor.fit_transform(X_train)
    Xva = preprocessor.transform(X_val)
    Xte = preprocessor.transform(X_test)

    isolation_forest = IsolationForest(
        n_estimators=200, max_samples="auto", contamination="auto",
        random_state=seed, n_jobs=-1,
    )
    isolation_forest.fit(Xtr)

    # Transparent robust-distance baseline: median/MAD are estimated on
    # training only; the validation set is used solely for threshold selection.
    med = np.median(Xtr, axis=0)
    mad = np.median(np.abs(Xtr - med), axis=0) * 1.4826
    mad[mad < 1e-9] = 1.0

    def robust_distance(values):
        z = (values - med) / mad
        return np.sqrt(np.mean(np.square(np.clip(z, -25, 25)), axis=1))

    models = {
        "isolation_forest": (
            -isolation_forest.score_samples(Xva),
            -isolation_forest.score_samples(Xte),
        ),
        "robust_mad_distance": (
            robust_distance(Xva),
            robust_distance(Xte),
        ),
    }

    report = {
        "dataset": "LID-DS 2021 CVE-2018-3760",
        "protocol": {
            "fit": "training benign windows only",
            "threshold": "validation benign windows only",
            "test": "one final evaluation; not used for fitting or threshold selection",
            "target_validation_fpr": target_fpr,
            "labeling": "metadata phase labels; attack windows start at/after exploit marker",
            "warning": "Validation contains no attack windows. This is benign-only anomaly detection, not supervised attack classification.",
        },
        "counts": {
            "training_windows": int(len(train_idx)),
            "validation_benign_windows": int(len(val_idx)),
            "test_windows": int(len(test_idx)),
            "test_benign_windows": int(np.sum(y_test == 0)),
            "test_attack_windows": int(np.sum(y_test == 1)),
            "feature_count": int(len(feature_names)),
        },
        "features": feature_names,
        "models": {},
        "python": platform.python_version(),
        "seed": seed,
    }

    for name, (val_scores, test_scores) in models.items():
        threshold = _higher_quantile_threshold(val_scores, target_fpr)
        val_pred = (val_scores > threshold).astype(int)
        test_pred = (test_scores > threshold).astype(int)
        y_val = np.zeros(len(val_scores), dtype=int)
        report["models"][name] = {
            "threshold": threshold,
            "validation": _metrics(y_val, val_pred, val_scores),
            "test_window": _metrics(y_test, test_pred, test_scores),
            "test_recording": _recording_metrics(rows, test_idx, test_pred),
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nSaved report: {out}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Benign-only LID-DS anomaly baseline benchmark")
    parser.add_argument("--input", default="results/lidds_windows_1s.csv")
    parser.add_argument("--output", default="results/lidds_baseline_report.json")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    run_benchmark(args.input, args.output, args.target_fpr, args.seed)


if __name__ == "__main__":
    main()
