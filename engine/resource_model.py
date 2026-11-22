"""Resource Drift Score (RDS) - build prompt section 5.

RDS = X_r . W_rds  where X_r are per-feature drift magnitudes (0..1) of the live
(cpu_util, ram_util, net_io_rate) against a rolling per-container healthy baseline, and W_rds are
correlation-derived weights fitted on the self-collected data/resource_baseline.csv (section 8.2).

Note this is a weighted *sum*: one saturated feature (e.g. a CPU-only cryptominer) yields RDS close to
that feature's weight, not 1.0. That is the algorithm as specified and is reported as such.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MODELS_DIR  # noqa: E402
from utils.log import logger  # noqa: E402

FEATURES = ("cpu_util", "ram_util", "net_io_rate")
# Absolute floors on the baseline std so a perfectly flat baseline doesn't make tiny wobbles look huge.
STD_FLOOR = {"cpu_util": 1.0, "ram_util": 1.0, "net_io_rate": 2048.0}
Z_CAP = 6.0
WEIGHTS_FILE = MODELS_DIR / "rds_weights.json"


class ResourceDriftModel:
    def __init__(self, weights: Optional[Dict[str, float]] = None, global_baseline: Optional[dict] = None,
                 window: int = 120, min_baseline: int = 10):
        self.window, self.min_baseline = window, min_baseline
        self.global_baseline = global_baseline          # {feature: {"mean":..,"std":..}} from benign collected rows
        if weights is None:
            logger.warning("RDS: no fitted W_rds found (run engine.resource_model --train after collecting "
                           "data/resource_baseline.csv). Using equal weights.")
            weights = {f: 1 / len(FEATURES) for f in FEATURES}
        self.weights = weights
        self._hist: Dict[str, Dict[str, deque]] = defaultdict(lambda: {f: deque(maxlen=window) for f in FEATURES})
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: Path = WEIGHTS_FILE) -> "ResourceDriftModel":
        if path.exists():
            blob = json.loads(path.read_text())
            return cls(blob["weights"], blob.get("benign_baseline"))
        return cls()

    def observe_healthy(self, container: str, x: Dict[str, float]):
        """Feed a sample into the rolling baseline. The engine only calls this when the container is at Monitor tier,
        so an attacker cannot slowly poison the baseline."""
        with self._lock:
            for f in FEATURES:
                self._hist[container][f].append(float(x[f]))

    def _baseline(self, container: str, f: str):
        h = self._hist[container][f]
        if len(h) >= self.min_baseline:
            arr = np.asarray(h)
            return float(arr.mean()), float(arr.std())
        if self.global_baseline:
            g = self.global_baseline[f]
            return g["mean"], g["std"]
        return None

    def score(self, container: str, x: Dict[str, float]) -> Optional[float]:
        """RDS in [0,1], or None while no baseline (per-container or global) exists yet."""
        with self._lock:
            total = 0.0
            for f in FEATURES:
                base = self._baseline(container, f)
                if base is None:
                    return None
                mean, std = base
                sigma = max(std, 0.05 * abs(mean), STD_FLOOR[f])
                drift = min(abs(float(x[f]) - mean) / sigma / Z_CAP, 1.0)
                total += self.weights[f] * drift
            return float(np.clip(total, 0.0, 1.0))

    def ready(self, container: str) -> bool:
        with self._lock:
            return all(self._baseline(container, f) is not None for f in FEATURES)


def fit_weights(csv_path: Path = DATA_DIR / "resource_baseline.csv", out: Path = WEIGHTS_FILE) -> dict:
    """W_rds_i = |corr(feature_i, label)| / sum_j |corr(feature_j, label)|, from REAL collected rows."""
    df = pd.read_csv(csv_path).dropna(subset=list(FEATURES) + ["label"])
    if df["label"].nunique() < 2:
        raise ValueError("resource_baseline.csv needs both benign (0) and attack (1) rows.")
    corr = {f: abs(float(np.corrcoef(df[f], df["label"])[0, 1])) if df[f].std() > 0 else 0.0 for f in FEATURES}
    total = sum(corr.values())
    weights = {f: (corr[f] / total if total > 0 else 1 / len(FEATURES)) for f in FEATURES}
    benign = df[df["label"] == 0]
    blob = {"weights": weights, "abs_corr": corr, "n_rows": int(len(df)), "n_benign": int(len(benign)),
            "benign_baseline": {f: {"mean": float(benign[f].mean()), "std": float(benign[f].std(ddof=0))} for f in FEATURES}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2))
    logger.info(f"RDS weights fitted on {len(df)} real rows: { {k: round(v, 3) for k, v in weights.items()} }")
    return blob


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit W_rds from data/resource_baseline.csv")
    ap.add_argument("--csv", default=str(DATA_DIR / "resource_baseline.csv"))
    ap.add_argument("--train", action="store_true", required=True)
    a = ap.parse_args()
    print(json.dumps(fit_weights(Path(a.csv)), indent=2))
