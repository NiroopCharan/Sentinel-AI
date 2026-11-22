"""Generate a reproducible synthetic smoke-test report; not a real dataset benchmark."""
import json, platform
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    rng = np.random.default_rng(20261113)
    y = np.array([0]*500 + [1]*100)
    # Deliberately separable toy score distribution to test the reporting pipeline.
    scores = np.r_[rng.normal(0, 1, 500), rng.normal(2.5, 1, 100)]
    pred = (scores > 1.25).astype(int)
    report = {"dataset": "SYNTHETIC_SMOKE_TEST_NOT_LID_DS", "seed": 20261113,
      "python": platform.python_version(), "n": int(len(y)),
      "accuracy": float(accuracy_score(y,pred)), "precision": float(precision_score(y,pred,zero_division=0)),
      "recall": float(recall_score(y,pred,zero_division=0)), "f1": float(f1_score(y,pred,zero_division=0)),
      "warning": "These numbers validate code execution only; do not report as CRBAC results."}
    out = Path("results/synthetic_smoke_test.json"); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2)); print(f"Saved: {out}")

if __name__ == "__main__": main()
