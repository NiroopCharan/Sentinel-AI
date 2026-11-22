import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

from crbac.lidds_detector import load_csv, _subset, _higher_quantile_threshold

rows, X, names = load_csv("results/debug_lidds.csv")

# Calculate temporal differences within each recording only.
D = np.zeros_like(X)
groups = defaultdict(list)

for i, r in enumerate(rows):
    groups[r["recording_id"]].append(i)

for rid, indices in groups.items():
    indices.sort(key=lambda i: rows[i]["window_start_float"])
    for pos in range(1, len(indices)):
        a, b = indices[pos-1], indices[pos]
        D[b] = np.abs(X[b] - X[a])

X_combined = np.concatenate([X, D], axis=1)

train_idx, _ = _subset(rows, X, "training", True)
val_idx, _ = _subset(rows, X, "validation", True)
test_idx, _ = _subset(rows, X, "test")

y = np.array([rows[i]["label_int"] for i in test_idx])

def evaluate(features, name):
    train = features[train_idx]
    val = features[val_idx]
    test = features[test_idx]

    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler())
    ])

    A = prep.fit_transform(train)
    V = prep.transform(val)
    T = prep.transform(test)

    med = np.median(A, axis=0)
    mad = np.median(np.abs(A-med), axis=0)*1.4826
    mad[mad < 1e-9] = 1.0

    def score(z):
        d = (z-med)/mad
        return np.sqrt(np.mean(np.square(np.clip(d,-25,25)),axis=1))

    val_scores = score(V)
    test_scores = score(T)

    threshold = _higher_quantile_threshold(val_scores, 0.01)
    pred = (test_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()

    return {
        "model": name,
        "features": int(features.shape[1]),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "fpr": float(fp/(fp+tn)),
        "precision": float(precision_score(y,pred,zero_division=0)),
        "recall": float(recall_score(y,pred,zero_division=0)),
        "f1": float(f1_score(y,pred,zero_division=0)),
        "auroc": float(roc_auc_score(y,test_scores)),
        "auprc": float(average_precision_score(y,test_scores))
    }

results = [
    evaluate(X, "Static Robust MAD"),
    evaluate(X_combined, "Static + Temporal Robust MAD")
]

report = {
    "experiment": "Static versus temporal feature ablation",
    "dataset": "LID-DS 2021 CVE-2018-3760",
    "split": "Existing training, validation and held-out test recordings",
    "threshold_selection": "1% target FPR on benign validation windows",
    "warning": "Exploratory test evaluation; the test set has already been inspected.",
    "results": results
}

Path("results").mkdir(exist_ok=True)
Path("results/temporal_ablation.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print(json.dumps(report, indent=2))
