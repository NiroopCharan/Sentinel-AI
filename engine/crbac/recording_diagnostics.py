
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
scores = np.sqrt(
    np.mean(np.square(np.clip(Z, -25, 25)), axis=1)
)

groups = defaultdict(list)

for j, idx in enumerate(va):
    groups[rows[idx]["recording_id"]].append(j)

recordings = []

for rid, indices in groups.items():
    values = scores[indices]

    recordings.append({
        "id": rid,
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "windows": len(indices)
    })

recordings.sort(key=lambda x: x["max"], reverse=True)

cal = recordings[::2]
threshold = np.sort([r["max"] for r in cal])[
    min(len(cal)-1, int(np.ceil((len(cal)+1)*0.90))-1)
]

print("Recording-level diagnostic")
print("Threshold:", round(threshold, 4))
print("\nTop 15 recordings by maximum anomaly score")
print("Recording ID | Max Score | Mean Score | P95 Score | Windows | Alarm")

for r in recordings[:15]:
    print(
        f"{r['id']} | {r['max']:.4f} | "
        f"{r['mean']:.4f} | {r['p95']:.4f} | "
        f"{r['windows']} | "
        f"{'YES' if r['max'] > threshold else 'NO'}"
    )

print("\nTotal recordings:", len(recordings))
print("Threshold:", round(threshold, 4))
print("Alarmed:", sum(r["max"] > threshold for r in recordings))
