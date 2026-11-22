import math
import numpy as np
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from crbac.lidds_detector import load_csv, _subset

rows, X, names = load_csv("results/debug_lidds.csv")

train_idx, X_train = _subset(rows, X, "training", True)
pre = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", RobustScaler())
])
Xtr = pre.fit_transform(X_train)

median = np.median(Xtr, axis=0)
mad = np.median(np.abs(Xtr - median), axis=0) * 1.4826
mad[mad < 1e-9] = 1.0

def score(x):
    z = (x - median) / mad
    return np.sqrt(np.mean(np.square(np.clip(z, -25, 25)), axis=1))

val_idx, X_val = _subset(rows, X, "validation", True)
val_scores = score(pre.transform(X_val))

groups = defaultdict(list)
for idx, s in zip(val_idx, val_scores):
    r = rows[idx]
    groups[r["recording_id"]].append(
        (r["window_start_float"], float(s))
    )

ids = sorted(groups)
cal_ids, eval_ids = ids[::2], ids[1::2]

cal_scores = np.array([
    s for rid in cal_ids for _, s in groups[rid]
])

print("Calibration windows:", len(cal_scores))
print("Evaluation windows:", sum(len(groups[r]) for r in eval_ids))
print()
print("Kappa | Alpha | Window FPR | Recording FPR | Alarmed recordings")

for kappa in (0.25, 0.5, 0.75):
    for alpha in (0.05, 0.01):
        alarmed_recordings = 0
        alarmed_windows = 0
        total_windows = 0

        for rid in eval_ids:
            log_e = 0.0
            alarmed = False
            ordered = sorted(groups[rid])

            for _, s in ordered:
                p = (1 + np.sum(cal_scores >= s)) / (len(cal_scores) + 1)
                p = max(float(p), 1e-15)
                log_e += math.log(kappa) + (kappa - 1) * math.log(p)

                if log_e >= math.log(1 / alpha):
                    alarmed = True

                if alarmed:
                    alarmed_windows += 1

            total_windows += len(ordered)
            alarmed_recordings += int(alarmed)

        print(
            f"{kappa:.2f}  | {alpha:.2f}  | "
            f"{alarmed_windows / total_windows:.4f}     | "
            f"{alarmed_recordings / len(eval_ids):.4f}        | "
            f"{alarmed_recordings}/{len(eval_ids)}"
        )
