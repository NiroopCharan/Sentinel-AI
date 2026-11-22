import numpy as np
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from crbac.lidds_detector import load_csv, _subset

rows, X, names = load_csv("results/debug_lidds.csv")
names = list(names)

tr, _ = _subset(rows, X, "training", True)
va, _ = _subset(rows, X, "validation", True)

groups = defaultdict(list)
for j, idx in enumerate(va):
    groups[rows[idx]["recording_id"]].append(j)

ids = sorted(groups)
cal_ids, eval_ids = ids[::2], ids[1::2]

constant = np.std(X[tr], axis=0) < 1e-12
sparse = np.mean(X[tr] == 0, axis=0) > 0.90

latency = np.array([
    "syscall_latency_ms_sum" in str(name)
    for name in names
])

configs = {
    "Original": np.ones(len(names), dtype=bool),
    "Remove constants": ~constant,
    "Remove constants+sparse": ~(constant | sparse),
    "Remove latency": ~latency,
    "Remove latency+constants": ~(latency | constant),
}

print("\n========== FEATURE ABLATION ==========")
print("Configuration | Features | Threshold | Cal alarms | Eval alarms | Eval FPR")

for label, mask in configs.items():
    if not np.any(mask):
        continue

    F = X[:, mask]

    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler())
    ])

    A = prep.fit_transform(F[tr])
    V = prep.transform(F[va])

    med = np.median(A, axis=0)
    mad = np.median(np.abs(A - med), axis=0) * 1.4826
    mad[mad < 1e-9] = 1.0

    Z = (V - med) / mad
    scores = np.sqrt(
        np.mean(np.square(np.clip(Z, -25, 25)), axis=1)
    )

    cal_max = np.array([
        max(scores[j] for j in groups[rid])
        for rid in cal_ids
    ])

    eval_max = np.array([
        max(scores[j] for j in groups[rid])
        for rid in eval_ids
    ])

    n = len(cal_max)
    k = min(n - 1, int(np.ceil((n + 1) * 0.90)) - 1)
    threshold = np.sort(cal_max)[k]

    cal_alarm = int(np.sum(cal_max > threshold))
    eval_alarm = int(np.sum(eval_max > threshold))
    fpr = eval_alarm / len(eval_max)

    print(
        f"{label:25s} | {np.sum(mask):3d} | "
        f"{threshold:.6f} | {cal_alarm}/{n} | "
        f"{eval_alarm}/{len(eval_max)} | {fpr:.4f}"
    )

print("\n========== FEATURE SELECTION ==========")
print("Constant features:")
print([names[i] for i in np.where(constant)[0]])

print("\nHighly sparse features:")
print([names[i] for i in np.where(sparse)[0]])

print("\nLatency features:")
print([names[i] for i in np.where(latency)[0]])

print("\nExperiment completed.")
