
import numpy as np
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from crbac.lidds_detector import load_csv, _subset

rows, X, names = load_csv("results/debug_lidds.csv")

D = np.zeros_like(X)
groups = defaultdict(list)

for i, r in enumerate(rows):
    groups[r["recording_id"]].append(i)

for indices in groups.values():
    indices.sort(key=lambda i: rows[i]["window_start_float"])
    for j in range(1, len(indices)):
        D[indices[j]] = np.abs(
            X[indices[j]] - X[indices[j - 1]]
        )

tr, _ = _subset(rows, X, "training", True)
va, _ = _subset(rows, X, "validation", True)

def fit_score(F):
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler())
    ])

    A = prep.fit_transform(F[tr])
    V = prep.transform(F[va])

    med = np.median(A, axis=0)
    mad = np.median(np.abs(A - med), axis=0) * 1.4826
    mad[mad < 1e-9] = 1.0

    def calc(z):
        z = (z - med) / mad
        return np.sqrt(
            np.mean(np.square(np.clip(z, -25, 25)), axis=1)
        )

    return calc(V)

S = fit_score(X)
T = fit_score(D)

vg = defaultdict(list)

for j, idx in enumerate(va):
    vg[rows[idx]["recording_id"]].append(j)

ids = sorted(vg)
cal_ids, eval_ids = ids[::2], ids[1::2]

print("Validation recordings:", len(ids))
print("Calibration recordings:", len(cal_ids))
print("Evaluation recordings:", len(eval_ids))

print("\nWeight | Threshold | Cal alarms | Eval alarms | Eval FPR")

for w in (0, 0.25, 0.5, 1, 2):
    combined = S + w * T

    cal_max = np.array([
        max(combined[j] for j in vg[rid])
        for rid in cal_ids
    ])

    eval_max = np.array([
        max(combined[j] for j in vg[rid])
        for rid in eval_ids
    ])

    n = len(cal_max)

    threshold_index = int(np.ceil((n + 1) * 0.90)) - 1
    threshold_index = min(max(threshold_index, 0), n - 1)

    threshold = np.sort(cal_max)[threshold_index]

    cal_alarm = int(np.sum(cal_max > threshold))
    eval_alarm = int(np.sum(eval_max > threshold))

    eval_fpr = eval_alarm / len(eval_max) if len(eval_max) else 0

    print(
        f"{w:5.2f} | {threshold:9.4f} | "
        f"{cal_alarm:2d}/{n} | "
        f"{eval_alarm:2d}/{len(eval_max)} | "
        f"{eval_fpr:.4f}"
    )
