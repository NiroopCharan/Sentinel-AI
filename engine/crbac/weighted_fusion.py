import numpy as np
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from crbac.lidds_detector import load_csv, _subset, _higher_quantile_threshold

rows, X, names = load_csv("results/debug_lidds.csv")

# Temporal differences, calculated within each recording.
D = np.zeros_like(X)
groups = defaultdict(list)

for i, r in enumerate(rows):
    groups[r["recording_id"]].append(i)

for indices in groups.values():
    indices.sort(key=lambda i: rows[i]["window_start_float"])
    for j in range(1, len(indices)):
        D[indices[j]] = np.abs(X[indices[j]] - X[indices[j-1]])

train_idx, _ = _subset(rows, X, "training", True)
val_idx, _ = _subset(rows, X, "validation", True)

# Fit separate preprocessors using training data only.
def fit_score(A, V):
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler())
    ])
    A = prep.fit_transform(A)
    V = prep.transform(V)

    med = np.median(A, axis=0)
    mad = np.median(np.abs(A-med), axis=0)*1.4826
    mad[mad < 1e-9] = 1.0

    def calc(z):
        z = (z-med)/mad
        return np.sqrt(np.mean(np.square(np.clip(z,-25,25)), axis=1))

    return calc(A), calc(V)

_, static_val = fit_score(X[train_idx], X[val_idx])
_, temporal_val = fit_score(D[train_idx], D[val_idx])

# Split validation by recording.
val_groups = defaultdict(list)
for j, idx in enumerate(val_idx):
    val_groups[rows[idx]["recording_id"]].append(j)

ids = sorted(val_groups)
cal_ids, eval_ids = ids[::2], ids[1::2]

cal_indices = [j for rid in cal_ids for j in val_groups[rid]]
eval_indices = [j for rid in eval_ids for j in val_groups[rid]]

print("Weight | Window FPR | Recording FPR | False-alarm recordings")

for weight in (0, 0.25, 0.5, 1, 2):
    combined = static_val + weight * temporal_val

    threshold = _higher_quantile_threshold(
        combined[cal_indices], 0.01
    )

    pred = combined[eval_indices] > threshold

    fp_windows = int(np.sum(pred))
    total_windows = len(eval_indices)

    alarmed = 0
    for rid in eval_ids:
        local = val_groups[rid]
        if any(combined[j] > threshold for j in local):
            alarmed += 1

    print(
        f"{weight:5.2f} | "
        f"{fp_windows/total_windows:.4f}     | "
        f"{alarmed/len(eval_ids):.4f}        | "
        f"{alarmed}/{len(eval_ids)}"
    )
