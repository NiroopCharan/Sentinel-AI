import csv
import math
import numpy as np
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from crbac.lidds_detector import load_csv, _subset

KAPPA = 0.75

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
y = np.array([rows[i]["label_int"] for i in te])

groups = defaultdict(list)
for j, idx in enumerate(te):
    groups[rows[idx]["recording_id"]].append(j)

e_values = np.ones(len(te))
pred = np.zeros(len(te), dtype=int)

for rid, indices in groups.items():
    indices.sort(key=lambda j: rows[te[j]]["window_start_float"])
    log_e = 0.0
    alarm = False

    for j in indices:
        p = (1 + np.sum(cal >= test_scores[j])) / (len(cal)+1)
        p = max(float(p), 1e-15)
        log_e += math.log(KAPPA)+(KAPPA-1)*math.log(p)
        e_values[j] = math.exp(max(-700, min(log_e, 700)))

        if log_e >= math.log(100):
            alarm = True
        if alarm:
            pred[j] = 1

print("EXPLORATORY TEST RANKING COMPARISON")
print("-----------------------------------")
for name, values in [
    ("Robust MAD", test_scores),
    ("Sequential e-value", e_values)
]:
    print(name)
    print(" AUROC:", round(roc_auc_score(y, values), 6))
    print(" AUPRC:", round(average_precision_score(y, values), 6))

print("\nSequential threshold metrics (E >= 100)")
print("TP:", int(np.sum((pred == 1) & (y == 1))))
print("FP:", int(np.sum((pred == 1) & (y == 0))))
print("FN:", int(np.sum((pred == 0) & (y == 1))))
print("TN:", int(np.sum((pred == 0) & (y == 0))))

with open("results/sequential_vs_mad_per_window.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["recording_id","window_start","label","robust_mad_score","e_value","prediction"])
    for j, idx in enumerate(te):
        w.writerow([
            rows[idx]["recording_id"],
            rows[idx]["window_start_float"],
            y[j],
            test_scores[j],
            e_values[j],
            pred[j]
        ])

print("\nSaved: results/sequential_vs_mad_per_window.csv")
