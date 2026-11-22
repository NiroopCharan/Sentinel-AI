from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "ieee_package"
GRAPHS = OUT / "graphs"
GRAPHS.mkdir(parents=True, exist_ok=True)

def load_json(name):
    path = RESULTS / name
    if not path.exists():
        print("MISSING:", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

baseline = load_json("lidds_baseline_report_verified.json")
sequential = load_json("CRBAC_sequential_conformal_experimental_result.json")
exploratory = load_json("crbac_k075_a001_exploratory.json")
ablation = load_json("temporal_ablation.json")

# Dataset distribution
data_path = RESULTS / "debug_lidds.csv"
df = pd.read_csv(data_path)
distribution = (
    df.groupby(["split", "label"]).size()
    .unstack(fill_value=0).reset_index()
)
distribution.to_csv(OUT / "dataset_distribution.csv", index=False)

recording_counts = (
    df.groupby(["split", "recording_id"])
    .agg(label=("label", "max"), windows=("label", "size"))
    .reset_index()
)
recording_summary = (
    recording_counts.groupby(["split", "label"])
    .agg(recordings=("recording_id", "count"), windows=("windows", "sum"))
    .reset_index()
)
recording_summary.to_csv(OUT / "recording_distribution.csv", index=False)

# Unified comparison table
rows = []

def add_model(name, window, recording, note):
    if not window:
        return
    rows.append({
        "Algorithm": name,
        "TN": window.get("tn"),
        "FP": window.get("fp"),
        "FN": window.get("fn"),
        "TP": window.get("tp"),
        "Accuracy": window.get("accuracy"),
        "Precision": window.get("precision"),
        "Recall": window.get("recall", window.get("attack_recall")),
        "F1": window.get("f1"),
        "Window_FPR": window.get("false_positive_rate", window.get("fpr")),
        "AUROC": window.get("auroc"),
        "AUPRC": window.get("auprc"),
        "Benign_Recording_FPR": recording.get("benign_recording_false_positive_rate"),
        "Attack_Recording_Detection": recording.get("attack_recording_detection_rate"),
        "Detected_Attack_Recordings": recording.get("detected_attack_recordings"),
        "Median_Detection_Delay_s": recording.get(
            "median_detection_delay_seconds",
            recording.get("median_detection_delay_from_exploit_seconds")
        ),
        "Notes": note
    })

models = baseline.get("models", {})
add_model(
    "Isolation Forest",
    models.get("isolation_forest", {}).get("test_window", {}),
    models.get("isolation_forest", {}).get("test_recording", {}),
    "Benign-only training; validation threshold"
)
add_model(
    "Robust MAD",
    models.get("robust_mad_distance", {}).get("test_window", {}),
    models.get("robust_mad_distance", {}).get("test_recording", {}),
    "Static features; benign-only calibration"
)
add_model(
    "CRBAC Sequential k=0.5 alpha=0.05",
    sequential.get("window_metrics", {}),
    sequential.get("recording_metrics", {}),
    "Experimental; conditional super-uniformity not established"
)
add_model(
    "CRBAC Exploratory k=0.75 alpha=0.01",
    exploratory.get("window_metrics", {}),
    exploratory.get("recording_metrics", {}),
    "Exploratory operating point; test set already inspected"
)

for item in ablation.get("results", []):
    rows.append({
        "Algorithm": item.get("model"),
        "TN": item.get("tn"),
        "FP": item.get("fp"),
        "FN": item.get("fn"),
        "TP": item.get("tp"),
        "Accuracy": None,
        "Precision": item.get("precision"),
        "Recall": item.get("recall"),
        "F1": item.get("f1"),
        "Window_FPR": item.get("fpr"),
        "AUROC": item.get("auroc"),
        "AUPRC": item.get("auprc"),
        "Benign_Recording_FPR": None,
        "Attack_Recording_Detection": None,
        "Detected_Attack_Recordings": None,
        "Median_Detection_Delay_s": None,
        "Notes": "Feature ablation; window-level evaluation"
    })

comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "algorithm_comparison.csv", index=False)

# Recording-level temporal calibration experiment
calibration = pd.DataFrame([
    [0.00, 2.2311, 2, 4, 0.1333],
    [0.25, 2.7085, 2, 8, 0.2667],
    [0.50, 3.5393, 2, 7, 0.2333],
    [1.00, 5.3713, 2, 5, 0.1667],
    [2.00, 8.8510, 2, 5, 0.1667],
], columns=[
    "Temporal_Weight", "Threshold", "Calibration_Alarms",
    "Evaluation_Alarms", "Evaluation_Recording_FPR"
])
calibration["Calibration_Recordings"] = 30
calibration["Evaluation_Recordings"] = 30
calibration.to_csv(OUT / "recording_calibration.csv", index=False)

# Plot 1: dataset windows
dist_plot = distribution.set_index("split")
ax = dist_plot.plot(kind="bar", figsize=(8, 5))
ax.set_title("LID-DS Dataset Window Distribution")
ax.set_xlabel("Dataset split")
ax.set_ylabel("Number of windows")
ax.legend(["Benign", "Attack"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(GRAPHS / "01_dataset_distribution.png", dpi=300)
plt.close()

# Plot 2: window-level metrics
plot_df = comparison.dropna(subset=["Precision", "Recall", "F1"])
if not plot_df.empty:
    ax = plot_df.set_index("Algorithm")[["Precision", "Recall", "F1"]].plot(
        kind="bar", figsize=(11, 6)
    )
    ax.set_title("Window-Level Detection Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(GRAPHS / "02_window_metrics.png", dpi=300)
    plt.close()

# Plot 3: recording-level detection and false alarms
rec = comparison.dropna(
    subset=["Benign_Recording_FPR", "Attack_Recording_Detection"]
)
if not rec.empty:
    ax = rec.set_index("Algorithm")[
        ["Benign_Recording_FPR", "Attack_Recording_Detection"]
    ].plot(kind="bar", figsize=(11, 6))
    ax.set_title("Recording-Level False Alarm and Attack Detection")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(GRAPHS / "03_recording_metrics.png", dpi=300)
    plt.close()

# Plot 4: confusion matrices
conf = comparison.dropna(subset=["TN", "FP", "FN", "TP"])
for _, r in conf.iterrows():
    matrix = np.array([
        [r["TN"], r["FP"]],
        [r["FN"], r["TP"]]
    ], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix)
    ax.set_title(str(r["Algorithm"]) + "\nConfusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["Benign", "Attack"])
    ax.set_yticks([0, 1], ["Benign", "Attack"])
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{int(value):,}", ha="center", va="center")
    plt.tight_layout()
    safe = "".join(c if c.isalnum() else "_" for c in str(r["Algorithm"]))[:55]
    plt.savefig(GRAPHS / f"confusion_{safe}.png", dpi=300)
    plt.close()

# Plot 5: temporal weight calibration
plt.figure(figsize=(8, 5))
plt.plot(
    calibration["Temporal_Weight"],
    calibration["Evaluation_Recording_FPR"],
    marker="o"
)
plt.xlabel("Temporal weight")
plt.ylabel("Evaluation recording FPR")
plt.title("Temporal Weight vs Recording-Level False Positive Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig(GRAPHS / "04_temporal_weight_fpr.png", dpi=300)
plt.close()

# Plot 6: static vs temporal feature ablation
if ablation.get("results"):
    abl = pd.DataFrame(ablation["results"])
    ax = abl.set_index("model")[["precision", "recall", "f1", "fpr"]].plot(
        kind="bar", figsize=(9, 5)
    )
    ax.set_title("Static vs Temporal Feature Ablation")
    ax.set_ylabel("Score")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(GRAPHS / "05_feature_ablation.png", dpi=300)
    plt.close()

# Plot 7: ROC and PR curves from stored per-window scores, if available
score_path = RESULTS / "sequential_vs_mad_per_window.csv"
curve_note = "ROC/PR curves not generated: per-window score file unavailable."
if score_path.exists():
    try:
        from sklearn.metrics import (
            roc_curve, precision_recall_curve, auc,
            average_precision_score, roc_auc_score
        )
        scores = pd.read_csv(score_path)
        if {"label", "robust_mad_score"}.issubset(scores.columns):
            y = scores["label"].astype(int).to_numpy()
            fig, ax = plt.subplots(figsize=(7, 6))
            curve_names = []
            for col, label in [
                ("robust_mad_score", "Robust MAD"),
                ("e_value", "Sequential e-value")
            ]:
                if col not in scores.columns:
                    continue
                s = pd.to_numeric(scores[col], errors="coerce")
                valid = s.notna()
                if y[valid].min() == y[valid].max():
                    continue
                fpr, tpr, _ = roc_curve(y[valid], s[valid])
                score_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{label} (AUC={score_auc:.4f})")
                curve_names.append((label, score_auc))
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Exploratory ROC Curves")
            ax.legend()
            plt.tight_layout()
            plt.savefig(GRAPHS / "06_roc_curves_exploratory.png", dpi=300)
            plt.close()

            fig, ax = plt.subplots(figsize=(7, 6))
            for col, label in [
                ("robust_mad_score", "Robust MAD"),
                ("e_value", "Sequential e-value")
            ]:
                if col not in scores.columns:
                    continue
                s = pd.to_numeric(scores[col], errors="coerce")
                valid = s.notna()
                if y[valid].min() == y[valid].max():
                    continue
                precision, recall, _ = precision_recall_curve(y[valid], s[valid])
                ap = average_precision_score(y[valid], s[valid])
                ax.plot(recall, precision, label=f"{label} (AP={ap:.4f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Exploratory Precision-Recall Curves")
            ax.legend()
            plt.tight_layout()
            plt.savefig(GRAPHS / "07_pr_curves_exploratory.png", dpi=300)
            plt.close()
            curve_note = "ROC/PR curves generated from stored per-window scores; exploratory test data."
    except Exception as exc:
        curve_note = f"ROC/PR generation skipped: {exc}"

# Save source result snapshots into package
for name in [
    "lidds_baseline_report_verified.json",
    "CRBAC_sequential_conformal_experimental_result.json",
    "crbac_k075_a001_exploratory.json",
    "temporal_ablation.json"
]:
    source = RESULTS / name
    if source.exists():
        (OUT / name).write_bytes(source.read_bytes())

# IEEE-oriented report
report = f"""# CRBAC Experimental Evaluation — IEEE Research Draft

## 1. Dataset
Dataset: LID-DS 2021, CVE-2018-3760.

Window counts:
{distribution.to_string(index=False)}

Recording counts:
{recording_summary.to_string(index=False)}

## 2. Experimental protocol
- Training uses benign training windows only.
- Calibration uses benign validation windows only.
- The baseline threshold is selected using the validation benign data.
- The test split contains both benign and attack windows.
- Test results are exploratory because the test set has already been inspected during development.
- Validation contains no attack windows; this is a benign-only anomaly-detection setting, not supervised attack classification.

## 3. Algorithm comparison
{comparison.to_string(index=False)}

## 4. Recording-level temporal calibration
{calibration.to_string(index=False)}

The calibration experiment contains 30 calibration recordings and 30 evaluation recordings for each temporal weight. The lowest observed evaluation recording FPR in this sweep is associated with temporal weight 0.00. These results do not establish attack detection performance because the supplied sweep reports false alarms only.

## 5. Feature ablation
{json.dumps(ablation.get("results", []), indent=2)}

## 6. Sequential method limitations
The sequential-conformal experiment is experimental. Conditional super-uniformity of sequential p-values has not been established, so formal sequential error-control guarantees must not be claimed. The evaluated component is not the complete integrated CRBAC architecture.

## 7. ROC and precision-recall curves
{curve_note}

AUROC and AUPRC are ranking metrics. They do not by themselves demonstrate a useful operating threshold. ROC/PR curves generated from test scores are exploratory because the test set has already been inspected.

## 8. Interpretation
The results show a trade-off between attack detection and false alarms. The sequential experiment reports higher attack-recording detection than the static baselines, but also a different false-alarm operating point. The exploratory kappa/alpha setting has a lower benign recording false-positive rate than the sequential kappa=0.5, alpha=0.05 experiment, but also a lower attack-recording detection rate.

Do not claim universal superiority. Report window-level and recording-level metrics separately. Clearly identify exploratory results and limitations.

## 9. Reproducibility and further work
- Freeze a new, untouched test split before final claims.
- Repeat experiments across multiple attack scenarios and random seeds.
- Report confidence intervals, preferably bootstrapped by recording rather than by window.
- Compare methods at matched recording-level false-positive rates.
- Evaluate detection delay and attack-recording detection consistently.
- Establish the assumptions required for any sequential validity guarantee.
- Evaluate the complete CRBAC system, including provenance and interdiction, separately from the detector component.

## 10. Generated artifacts
See `algorithm_comparison.csv`, `dataset_distribution.csv`, `recording_distribution.csv`, `recording_calibration.csv`, and the `graphs` directory.
"""
(OUT / "IEEE_experimental_report.md").write_text(report, encoding="utf-8")

print("\nPACKAGE CREATED")
print("Location:", OUT)
print("\nCSV files:")
for p in sorted(OUT.glob("*.csv")):
    print(" -", p.name)
print("\nGraphs:")
for p in sorted(GRAPHS.glob("*.png")):
    print(" -", p.name)
print("\nReport: IEEE_experimental_report.md")
print("\nNOTE:", curve_note)
