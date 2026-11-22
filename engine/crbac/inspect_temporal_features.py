import numpy as np
from collections import defaultdict
from crbac.lidds_detector import load_csv

rows, X, names = load_csv("results/debug_lidds.csv")

groups = defaultdict(list)
for i, r in enumerate(rows):
    groups[r["recording_id"]].append(i)

D = np.zeros_like(X)

for rid, indices in groups.items():
    indices.sort(key=lambda i: rows[i]["window_start_float"])
    for pos in range(1, len(indices)):
        a, b = indices[pos-1], indices[pos]
        D[b] = np.abs(X[b] - X[a])

print("Original features:", X.shape[1])
print("Temporal delta features:", D.shape[1])
print("Combined features:", X.shape[1] + D.shape[1])
print("Training windows:", sum(r["split"] == "training" for r in rows))
print("Validation windows:", sum(r["split"] == "validation" for r in rows))
print("Test windows:", sum(r["split"] == "test" for r in rows))
print("Temporal feature matrix finite:", bool(np.isfinite(D).all()))
