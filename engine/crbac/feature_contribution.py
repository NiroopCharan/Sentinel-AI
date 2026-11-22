import numpy as np
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from crbac.lidds_detector import load_csv, _subset

rows, X, names = load_csv("results/debug_lidds.csv")

tr, _ = _subset(rows, X, "training", True)
va, _ = _subset(rows, X, "validation", True)

prep = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", RobustScaler())
])

A = prep.fit_transform(X[tr])
V = prep.transform(X[va])

med = np.median(A, axis=0)
mad = np.median(np.abs(A - med), axis=0) * 1.4826
mad[mad < 1e-9] = 1.0

Z = (V - med) / mad
C = np.square(np.clip(Z, -25, 25))
scores = np.sqrt(np.mean(C, axis=1))

groups = defaultdict(list)
for j, idx in enumerate(va):
    groups[rows[idx]["recording_id"]].append(j)

ids = sorted(groups)
cal_ids, eval_ids = ids[::2], ids[1::2]

cal_max = np.array([
    max(scores[j] for j in groups[rid]) for rid in cal_ids
])

threshold = np.sort(cal_max)[
    min(len(cal_max)-1, int(np.ceil((len(cal_max)+1)*0.90))-1)
]

print("\n========== CONSTANT FEATURES ==========")
for i, name in enumerate(names):
    if np.std(X[tr, i]) == 0:
        print(
            name,
            "| training value:", X[tr[0], i],
            "| validation unique:", len(np.unique(X[va, i]))
        )

print("\n========== CLIPPING ANALYSIS ==========")
for rid in eval_ids:
    indices = groups[rid]
    j = max(indices, key=lambda k: scores[k])
    clipped = int(np.sum(np.abs(Z[j]) >= 25))

    if scores[j] > threshold - 0.15:
        print(
            f"{rid} | score={scores[j]:.6f} | "
            f"clipped_features={clipped}/ {len(names)}"
        )

        top = np.argsort(C[j])[::-1][:10]
        print("  Top contributing features:")
        for i in top:
            print(
                f"    {names[i]} | raw={V[j, i]:.4f} | "
                f"standardized={Z[j, i]:.4f} | "
                f"contribution={C[j, i]:.4f}"
            )

print("\n========== FEATURE SPARSITY ==========")
for i, name in enumerate(names):
    zero_rate = np.mean(X[tr, i] == 0)
    if zero_rate > 0.90:
        print(f"{name}: {zero_rate:.2%} zeros")

print("\n========== SUMMARY ==========")
print("Threshold:", round(float(threshold), 6))
print("Evaluation recordings:", len(eval_ids))
print("Evaluation alarms:", sum(
    max(scores[j] for j in groups[rid]) > threshold
    for rid in eval_ids
))
