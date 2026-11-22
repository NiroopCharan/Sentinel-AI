"""Identity Consistency Score (ICS) - the key novelty (build prompt section 5 / 8.3).

Question answered: "is this session consistent with how THIS identity historically behaves?", independent of
whether the credentials were valid. Features per session:
  hour_dev        circular distance (hours) between session start hour and the owner's historical centre
  timing_entropy  Shannon entropy (bits) of binned inter-command gaps - humans are irregular, scripts are bursty
  lineage_dev     0 if the process lineage shape (e.g. sshd>bash) was seen for the owner, else 1
  origin_dev      0 if the source IP is a known origin, else 1

Model: diagonal Mahalanobis distance from the owner's feature distribution (variance floors keep tiny samples
and constant binary features from exploding), mapped to [0,1] using the owner's 95th-percentile distance as
"fully consistent" and the intruders' 25th-percentile distance (if intruder rows exist) as "fully inconsistent".

Train:  python -m engine.identity_model --train        (reads data/identity_sessions.csv - self-collected)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, KNOWN_ORIGINS, MODELS_DIR  # noqa: E402
from utils.log import logger  # noqa: E402

FEATURES = ("hour_dev", "timing_entropy", "lineage_dev", "origin_dev")
STD_FLOOR = np.array([1.5, 0.35, 0.30, 0.30])
GAP_BINS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, math.inf]   # seconds
MODEL_FILE = MODELS_DIR / "identity_model.json"
CSV_COLUMNS = ["session_id", "container", "label", "identity", "source_ip", "origin_known", "start_hour",
               "hour_simulated", "origin_simulated", "lineage", "cmd_intervals", "n_commands", "scenario", "recorded_at"]


def circ_dist(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 24
    return min(d, 24 - d)


def circ_mean_hour(hours: Sequence[float]) -> float:
    ang = [h / 24 * 2 * math.pi for h in hours]
    return (math.atan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) % (2 * math.pi)) / (2 * math.pi) * 24


def timing_entropy(intervals: Sequence[float]) -> float:
    if len(intervals) < 2:
        return 0.0
    hist, _ = np.histogram(np.clip(intervals, 0, 1e6), bins=GAP_BINS)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def lineage_shape(lineage: str) -> str:
    """'sshd>bash>cat' -> 'sshd>bash' (parent shape, ignoring the leaf tool)."""
    return ">".join(lineage.split(">")[:2])


class IdentityModel:
    def __init__(self):
        self.hour_center = 12.0
        self.mean = np.zeros(len(FEATURES))
        self.std = np.ones(len(FEATURES))
        self.owner_shapes: List[str] = []
        self.known_origins: List[str] = list(KNOWN_ORIGINS)
        self.d_lo, self.d_hi = 1.0, 3.0
        self.fitted = False

    # ---- features ----
    def features(self, s: dict) -> np.ndarray:
        intervals = s["intervals"] if "intervals" in s else [float(x) for x in str(s.get("cmd_intervals", "")).split(";") if x]
        if s.get("source_ip") not in (None, "") and not (isinstance(s.get("source_ip"), float)):
            origin_known = int(str(s["source_ip"]) in self.known_origins)
        else:
            origin_known = int(s.get("origin_known", 0))
        return np.array([
            circ_dist(float(s["start_hour"]), self.hour_center),
            timing_entropy(intervals),
            0.0 if lineage_shape(str(s["lineage"])) in self.owner_shapes else 1.0,
            0.0 if int(origin_known) else 1.0,
        ])

    def _distance(self, x: np.ndarray) -> float:
        z = (x - self.mean) / self.std
        return float(np.sqrt((z ** 2).sum() / len(FEATURES)))

    def score(self, session: dict) -> float:
        """ICS in [0,1]: 0 = consistent with the owner, 1 = clearly not the owner."""
        if not self.fitted:
            raise RuntimeError("IdentityModel not fitted")
        d = self._distance(self.features(session))
        return float(np.clip((d - self.d_lo) / max(self.d_hi - self.d_lo, 1e-6), 0.0, 1.0))

    # ---- training ----
    def fit(self, sessions: pd.DataFrame) -> dict:
        owners = sessions[sessions["label"] == "owner"]
        intruders = sessions[sessions["label"] == "intruder"]
        if len(owners) < 3:
            raise ValueError("Need at least 3 owner sessions in identity_sessions.csv to model an identity.")
        self.hour_center = circ_mean_hour(owners["start_hour"].astype(float))
        self.owner_shapes = sorted({lineage_shape(str(l)) for l in owners["lineage"]})
        # The owner's historical origins define "known origin" (plus any configured via SENTINEL_KNOWN_ORIGINS).
        self.known_origins = sorted(set(self.known_origins) | set(owners["source_ip"].dropna().astype(str)))
        # features need owner_shapes/known_origins set first
        self.mean, self.std = np.zeros(len(FEATURES)), np.ones(len(FEATURES))
        X = np.vstack([self.features(r) for r in owners.to_dict("records")])
        self.mean = X.mean(axis=0)
        self.std = np.maximum(X.std(axis=0), STD_FLOOR)
        owner_d = np.array([self._distance(x) for x in X])
        self.d_lo = max(float(np.percentile(owner_d, 95)), 0.5)
        info = {"n_owner": int(len(owners)), "n_intruder": int(len(intruders)), "d_lo": self.d_lo}
        if len(intruders):
            XI = np.vstack([self.features(r) for r in intruders.to_dict("records")])
            intr_d = np.array([self._distance(x) for x in XI])
            self.d_hi = max(float(np.percentile(intr_d, 25)), self.d_lo * 1.5)
            info["intruder_d_p25"] = float(np.percentile(intr_d, 25))
        else:
            self.d_hi = self.d_lo * 3.0
        info["d_hi"] = self.d_hi
        self.fitted = True
        return info

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {"hour_center": self.hour_center, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "owner_shapes": self.owner_shapes, "known_origins": self.known_origins,
                "d_lo": self.d_lo, "d_hi": self.d_hi}

    @classmethod
    def from_dict(cls, d: dict) -> "IdentityModel":
        m = cls()
        m.hour_center, m.mean, m.std = d["hour_center"], np.array(d["mean"]), np.array(d["std"])
        m.owner_shapes, m.known_origins = d["owner_shapes"], d["known_origins"]
        m.d_lo, m.d_hi, m.fitted = d["d_lo"], d["d_hi"], True
        return m

    @classmethod
    def load(cls, path: Path = MODEL_FILE) -> Optional["IdentityModel"]:
        return cls.from_dict(json.loads(path.read_text())) if path.exists() else None

    def save(self, path: Path = MODEL_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def train(csv_path: Path = DATA_DIR / "identity_sessions.csv", out: Path = MODEL_FILE) -> dict:
    df = pd.read_csv(csv_path)
    model = IdentityModel()
    info = model.fit(df)
    model.save(out)
    # Resubstitution sanity check only - NOT a generalisation estimate.
    scores = df.assign(ics=[model.score(r) for r in df.to_dict("records")])
    info["mean_ics_by_label_resubstitution"] = scores.groupby("label")["ics"].mean().round(3).to_dict()
    logger.info(f"ICS model trained: {info}")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the Identity Consistency model")
    ap.add_argument("--train", action="store_true", required=True)
    ap.add_argument("--csv", default=str(DATA_DIR / "identity_sessions.csv"))
    args = ap.parse_args()
    print(json.dumps(train(Path(args.csv)), indent=2))
