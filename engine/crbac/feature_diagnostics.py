import numpy as np
from collections import defaultdict, Counter
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
scores = np.sqrt(np.mean(np.square(np.clip(Z, -25, 25)), axis=1))

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

eval_max = np.array([
    max(scores[j] for j in groups[rid]) for rid in eval_ids
])

print("\n========== DATA QUALITY ==========")
print("Training windows:", len(tr))
print("Validation windows:", len(va))
print("Feature count:", X.shape[1])
print("Exact duplicate validation rows:", len(va)-len(np.unique(X[va], axis=0)))
print("Unique validation rows:", len(np.unique(X[va], axis=0)))
print("Constant features:", np.sum(np.std(X[tr], axis=0) == 0))
print("Features with >90% zeros:", np.sum(np.mean(X[tr] == 0, axis=0) > 0.90))

print("\n========== SCORE DISTRIBUTION ==========")
print("Unique scores (rounded 6 decimals):", len(np.unique(np.round(scores, 6))))
print("Minimum:", round(float(scores.min()), 6))
print("Median:", round(float(np.median(scores)), 6))
print("95th percentile:", round(float(np.percentile(scores, 95)), 6))
print("Maximum:", round(float(scores.max()), 6))
print("Calibration threshold:", round(float(threshold), 6))
print("Evaluation recording FPR:", round(float(np.mean(eval_max > threshold)), 4))

print("\n========== TOP FEATURE VARIATION ==========")
std = np.std(X[tr], axis=0)
for i in np.argsort(std)[:15]:
    print(f"{names[i]} | std={std[i]:.8f} | unique={len(np.unique(X[tr, i]))}")

print("\n========== RECORDING CHECK ==========")
for rid in eval_ids:
    indices = groups[rid]
    vals = scores[indices]
    print(
        f"{rid} | windows={len(indices)} | "
        f"max={max(vals):.6f} | mean={np.mean(vals):.6f} | "
        f"alarm={max(vals) > threshold}"
    )
