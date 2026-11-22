import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

seq = json.load(open("results/crbac_sequential_verified.json", encoding="utf-8"))
base = json.load(open("results/lidds_baseline_verified.json", encoding="utf-8"))

out = Path("results/graphs")
out.mkdir(parents=True, exist_ok=True)

m = seq["window_metrics"]
y = np.array([r["label"] for r in seq["per_window"]])
scores = np.array([r["robust_mad_score"] for r in seq["per_window"]])
evalues = np.array([r["e_value"] for r in seq["per_window"]])

# 1. Confusion matrix
cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
fig, ax = plt.subplots()
im = ax.imshow(cm)
ax.set_xticks([0, 1], ["Benign", "Attack"])
ax.set_yticks([0, 1], ["Benign", "Attack"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Sequential-Conformal Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(out / "01_confusion_matrix.png", dpi=300)
plt.close(fig)

# 2. Window-level metric comparison
models = {
    "Isolation Forest": base["models"]["isolation_forest"]["test_window"],
    "Robust MAD": base["models"]["robust_mad_distance"]["test_window"],
    "Sequential-Conformal": m
}
metrics = ["precision", "attack_recall", "f1", "false_positive_rate"]
labels = ["Precision", "Recall", "F1-score", "FPR"]
x = np.arange(len(metrics))
width = 0.24

fig, ax = plt.subplots(figsize=(9, 5))
for i, (name, values) in enumerate(models.items()):
    vals = [values[k] or 0 for k in metrics]
    ax.bar(x + (i-1)*width, vals, width, label=name)
ax.set_xticks(x, labels)
ax.set_ylabel("Score")
ax.set_title("Window-Level Performance Comparison")
ax.legend()
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(out / "02_metric_comparison.png", dpi=300)
plt.close(fig)

# 3. Recording-level detection
recording = seq["recording_metrics"]
names = ["Attack recordings\ndetected", "Attack recordings\nmissed"]
values = [
    recording["detected_attack_recordings"],
    recording["undetected_attack_recordings"]
]
fig, ax = plt.subplots()
ax.bar(names, values)
ax.set_ylabel("Number of recordings")
ax.set_title("Sequential-Conformal Recording Detection")
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(out / "03_recording_detection.png", dpi=300)
plt.close(fig)

# 4. ROC curves: robust score and sequential e-value
fig, ax = plt.subplots()
for name, value in [
    ("Robust MAD score", scores),
    ("Sequential e-value", evalues)
]:
    fpr, tpr, _ = roc_curve(y, value)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.4f})")
ax.plot([0, 1], [0, 1], "--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves on Held-Out Test Windows")
ax.legend()
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(out / "04_roc_curve.png", dpi=300)
plt.close(fig)

# 5. Precision-recall curves
fig, ax = plt.subplots()
for name, value in [
    ("Robust MAD score", scores),
    ("Sequential e-value", evalues)
]:
    precision, recall, _ = precision_recall_curve(y, value)
    ax.plot(recall, precision, label=name)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves on Held-Out Test Windows")
ax.legend()
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(out / "05_precision_recall.png", dpi=300)
plt.close(fig)

print("Generated graphs:")
for p in sorted(out.glob("*.png")):
    print(p)
