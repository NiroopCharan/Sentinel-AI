import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from crbac.lidds_detector import (
    load_csv, _subset, _recording_metrics, _metrics
)

INPUT = "results/debug_lidds.csv"
OUTPUT = "results/crbac_sequential_verified.json"
KAPPA = 0.5
ALPHA = 0.05
THRESHOLD = 1.0 / ALPHA

rows, X, feature_names = load_csv(INPUT)

train_idx, X_train = _subset(rows, X, "training", require_benign=True)
val_idx, X_val = _subset(rows, X, "validation", require_benign=True)
test_idx, X_test = _subset(rows, X, "test", require_benign=False)

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", RobustScaler()),
])

Xtr = preprocessor.fit_transform(X_train)
Xva = preprocessor.transform(X_val)
Xte = preprocessor.transform(X_test)

median = np.median(Xtr, axis=0)
mad = np.median(np.abs(Xtr - median), axis=0) * 1.4826
mad[mad < 1e-9] = 1.0

def score(values):
    z = (values - median) / mad
    return np.sqrt(np.mean(np.square(np.clip(z, -25, 25)), axis=1))

val_scores = score(Xva)
test_scores = score(Xte)

groups = defaultdict(list)
for local_i, global_i in enumerate(test_idx):
    groups[rows[global_i]["recording_id"]].append(local_i)

pred = np.zeros(len(test_idx), dtype=int)
p_values = np.ones(len(test_idx))
e_values = np.ones(len(test_idx))
crossings = 0

for recording_id, local_indices in groups.items():
    local_indices.sort(
        key=lambda i: rows[test_idx[i]]["window_start_float"]
    )

    log_e = 0.0
    alarmed = False

    for i in local_indices:
        p = (1 + np.sum(val_scores >= test_scores[i])) / (len(val_scores) + 1)
        p = max(float(p), 1e-15)

        log_e += math.log(KAPPA) + (KAPPA - 1) * math.log(p)
        e = math.exp(min(log_e, 700))

        p_values[i] = p
        e_values[i] = e

        if not alarmed and e >= THRESHOLD:
            alarmed = True
            crossings += 1

        if alarmed:
            pred[i] = 1

y_test = np.array([rows[i]["label_int"] for i in test_idx])
window_metrics = _metrics(y_test, pred, test_scores)
recording_metrics = _recording_metrics(rows, test_idx, pred)

report = {
    "experiment": "CRBAC sequential-conformal component",
    "dataset": "LID-DS 2021 CVE-2018-3760",
    "input": INPUT,
    "protocol": {
        "training": "benign training windows only",
        "calibration": "benign validation windows only",
        "test": "held-out test windows",
        "score": "Robust MAD distance",
        "p_value": "upper-tail split-conformal p-value",
        "sequential_rule": "Product e-process, reset per recording",
        "kappa": KAPPA,
        "alpha": ALPHA,
        "alert_rule": "E-value >= 1/alpha; alert persists for remainder of recording",
        "limitation": "Temporal conditional super-uniformity is not established; this is an experimental component, not the complete CRBAC system."
    },
    "counts": {
        "training_windows": int(len(train_idx)),
        "validation_benign_windows": int(len(val_idx)),
        "test_windows": int(len(test_idx)),
        "test_benign_windows": int(np.sum(y_test == 0)),
        "test_attack_windows": int(np.sum(y_test == 1)),
        "features": int(len(feature_names)),
        "test_recordings": int(len(groups))
    },
    "threshold_crossings": int(crossings),
    "window_metrics": window_metrics,
    "recording_metrics": recording_metrics,
    "per_window": [
        {
            "recording_id": rows[test_idx[i]]["recording_id"],
            "window_start": rows[test_idx[i]]["window_start_float"],
            "label": int(y_test[i]),
            "robust_mad_score": float(test_scores[i]),
            "conformal_p_value": float(p_values[i]),
            "e_value": float(e_values[i]),
            "prediction": int(pred[i])
        }
        for i in range(len(test_idx))
    ]
}

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
Path(OUTPUT).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

print(json.dumps({k: v for k, v in report.items() if k != "per_window"}, indent=2))
print(f"\nSaved verified experiment: {OUTPUT}")
