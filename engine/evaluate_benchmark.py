"""Honest benchmark (build prompt section 12). Every number printed/saved comes from running this file.

    python -m engine.evaluate_benchmark                    # real KDDTest+.txt
    python -m engine.evaluate_benchmark --allow-synthetic  # smoke test; outputs are suffixed *_synthetic and watermarked

What is and is not evaluated:
  * HMR here = fusion of RF + AS (weights renormalised). RDS and ICS need container resource / SSH-session
    telemetry that NSL-KDD does not contain, so they CANNOT be evaluated on this dataset.
  * Baselines (GaussianNB, LinearSVC, LogisticRegression, KNN) use the identical train partition and preprocessing.
  * Ablations RF-only and AS-only are reported from the same trained models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (balanced_accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             matthews_corrcoef, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MODELS_DIR  # noqa: E402
from engine.data_loader import class_balance, feature_columns, load_train_test, split_train_val  # noqa: E402
from engine.fusion import normalize_anomaly  # noqa: E402
from engine.train_models import ts_network_only  # noqa: E402
from utils.log import logger  # noqa: E402


def metrics(name, y, pred, score):
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y, score)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "report": classification_report(y, pred, target_names=["normal", "attack"], output_dict=True, zero_division=0),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    args = ap.parse_args(argv)
    models, out = Path(args.models_dir), Path(args.out_dir)

    meta = json.loads((models / "meta.json").read_text())
    train, test, provenance = load_train_test(args.allow_synthetic)
    if meta["trained_on"] != provenance:
        raise SystemExit(f"Models were trained on '{meta['trained_on']}' data but evaluation data is '{provenance}'. "
                         "Re-run engine.train_models on the same data source.")
    suffix = "real" if provenance == "real" else "synthetic"
    mark = "" if provenance == "real" else "SYNTHETIC — NOT NSL-KDD\n"
    if provenance != "real":
        logger.warning("SYNTHETIC DEMO DATA — not NSL-KDD. These outputs are a pipeline smoke test, not results.")

    fit_df, val_df = split_train_val(train, 0.2, meta["seed"])  # identical partition to train_models
    cols = feature_columns(meta["feature_set"])
    pre, rf, iso = (joblib.load(models / f) for f in ("preprocessor.joblib", "rf.joblib", "iso.joblib"))
    X_fit, X_test = pre.transform(fit_df[cols]), pre.transform(test[cols])
    y_fit, y_test = fit_df["y"].to_numpy(), test["y"].to_numpy()

    print("\n=== Data used ===")
    print(f"provenance: {provenance} | feature set: {meta['feature_set']} | tau={meta['tau']:.2f} margin={meta['margin']}")
    print(f"train-fit : {class_balance(fit_df)}")
    print(f"train-val : {class_balance(val_df)} (used only to tune tau)")
    print(f"test      : {class_balance(test)}")

    rf_p = rf.predict_proba(X_test)[:, 1]
    as_s = normalize_anomaly(iso.score_samples(X_test), meta["as_min"], meta["as_max"])
    ts = ts_network_only(rf_p, as_s)

    results = [
        metrics("HMR (RF+AS fusion)", y_test, (ts >= meta["tau"]).astype(int), ts),
        metrics("Ablation: RF only", y_test, (rf_p >= 0.5).astype(int), rf_p),
        metrics("Ablation: AS only", y_test, (as_s >= meta["tau"]).astype(int), as_s),
    ]
    baselines = {
        "Naive Bayes": GaussianNB(),
        "Linear SVM": LinearSVC(dual="auto", max_iter=5000, random_state=meta["seed"]),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=meta["seed"]),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    for name, clf in baselines.items():
        clf.fit(X_fit, y_fit)
        pred = clf.predict(X_test)
        score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
        results.append(metrics(name, y_test, pred, score))

    print("\n=== Results (test partition, binary normal vs attack) ===")
    print(f"{'model':<24}{'ROC-AUC':>9}{'MCC':>8}{'BalAcc':>8}{'Prec(att)':>11}{'Rec(att)':>10}{'F1(att)':>9}")
    for r in results:
        a = r["report"]["attack"]
        print(f"{r['model']:<24}{r['roc_auc']:>9.4f}{r['mcc']:>8.4f}{r['balanced_accuracy']:>8.4f}"
              f"{a['precision']:>11.4f}{a['recall']:>10.4f}{a['f1-score']:>9.4f}")
    print("\nFull classification reports and confusion matrices are in the JSON file below.")
    print("Reminder: RDS and ICS are not evaluable on NSL-KDD; HMR here is RF+AS only.\n")

    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, r in zip(axes.ravel(), results):
        cm = np.array(r["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", color="black" if v < cm.max() / 2 else "white")
        ax.set_xticks([0, 1], ["normal", "attack"]); ax.set_yticks([0, 1], ["normal", "attack"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(r["model"], fontsize=10)
    for ax in axes.ravel()[len(results):]:
        ax.axis("off")
    fig.suptitle(f"{mark}Confusion matrices — NSL-KDD test partition ({provenance})")
    fig.tight_layout(); fig.savefig(out / f"confusion_matrix_{suffix}.png", dpi=150); plt.close(fig)

    plt.figure(figsize=(7, 6))
    scores_by_name = {"HMR (RF+AS fusion)": ts, "Ablation: RF only": rf_p, "Ablation: AS only": as_s}
    for name, clf in baselines.items():
        scores_by_name[name] = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, scores_by_name[r["model"]])
        plt.plot(fpr, tpr, label=f"{r['model']} (AUC {r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"{mark}ROC comparison — NSL-KDD test ({provenance})"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / f"roc_comparison_{suffix}.png", dpi=150); plt.close()

    (out / f"benchmark_results_{suffix}.json").write_text(json.dumps(
        {"provenance": provenance, "meta": meta, "test_balance": class_balance(test), "results": results}, indent=2))
    logger.info(f"Saved confusion_matrix_{suffix}.png, roc_comparison_{suffix}.png, benchmark_results_{suffix}.json to {out}")


if __name__ == "__main__":
    main()
