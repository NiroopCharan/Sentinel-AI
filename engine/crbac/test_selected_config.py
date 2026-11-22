import json
import math
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from crbac.lidds_detector import load_csv, _subset, _metrics, _recording_metrics

KAPPA = 0.75
ALPHA = 0.01

rows, X, names = load_csv("results/debug_lidds.csv")
tr, Xt = _subset(rows, X, "training", True)
va, Xv = _subset(rows, X, "validation", True)
te, Xe = _subset(rows, X, "test")

prep = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", RobustScaler())
])
A = prep.fit_transform(Xt)
V = prep.transform(Xv)
T = prep.transform(Xe)

med = np.median(A, axis=0)
mad = np.median(np.abs(A-med), axis=0)*1.4826
mad[mad < 1e-9] = 1.0

def score(x):
    z = (x-med)/mad
    return np.sqrt(np.mean(np.square(np.clip(z,-25,25)),axis=1))

cal = score(V)
test_scores = score(T)

groups = defaultdict(list)
for j, idx in enumerate(te):
    groups[rows[idx]["recording_id"]].append(j)

pred = np.zeros(len(te), dtype=int)
crossings = 0

for rid, indices in groups.items():
    indices.sort(key=lambda j: rows[te[j]]["window_start_float"])
    log_e = 0.0
    alarm = False

    for j in indices:
        p = (1 + np.sum(cal >= test_scores[j])) / (len(cal)+1)
        p = max(float(p), 1e-15)
        log_e += math.log(KAPPA)+(KAPPA-1)*math.log(p)

        if not alarm and log_e >= math.log(1/ALPHA):
            alarm = True
            crossings += 1
        if alarm:
            pred[j] = 1

y = np.array([rows[i]["label_int"] for i in te])
report = {
    "kappa": KAPPA,
    "alpha": ALPHA,
    "threshold_crossings": crossings,
    "window_metrics": _metrics(y, pred, test_scores),
    "recording_metrics": _recording_metrics(rows, te, pred),
    "warning": "Exploratory test evaluation; not an independent final test."
}

Path("results").mkdir(exist_ok=True)
Path("results/crbac_k075_a001_exploratory.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
