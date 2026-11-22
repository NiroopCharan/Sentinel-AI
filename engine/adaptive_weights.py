"""Novel contribution #1 - adaptive fusion weights (build prompt section 6).

Differences from the literal snippet in the prompt, both deliberate:
  * Weights are  prior_i * exp(3 * acc_i) , renormalised. With every accuracy at its 0.5 start value this
    reproduces the prescribed starting weights (0.40/0.25/0.15/0.20) instead of collapsing to 0.25 each;
    as accuracies diverge the weights move away from the prior.
  * Every update is appended to an audit trail (data/weight_audit.jsonl) with timestamp, incident and
    the verification that triggered it - the basis for the "post-remediation re-weighting" claim.
State persists in data/weights_state.json so learning survives restarts.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config import DATA_DIR, INITIAL_WEIGHTS, VECTORS
from utils.log import logger


class AdaptiveWeightTracker:
    def __init__(self, alpha: float = 0.05, prior: Optional[Dict[str, float]] = None,
                 state_path: Optional[Path] = None, audit_path: Optional[Path] = None, sharpness: float = 3.0):
        self.alpha = alpha
        self.sharpness = sharpness
        self.prior = dict(prior or INITIAL_WEIGHTS)
        self.acc = {v: 0.5 for v in VECTORS}  # rolling per-vector accuracy
        self.state_path = Path(state_path) if state_path else None
        self.audit_path = Path(audit_path) if audit_path else None
        self._lock = threading.Lock()
        if self.state_path and self.state_path.exists():
            try:
                self.acc.update(json.loads(self.state_path.read_text())["acc"])
            except Exception as exc:  # corrupt state must never stop the engine
                logger.warning(f"Could not load weight state ({exc}); starting from priors.")

    def update(self, vector_name: str, was_correct: bool, incident_id: str = "", context: Optional[dict] = None):
        """Update ONE vector from a *verified* outcome (never from an unverified prediction)."""
        with self._lock:
            before = self._weights_unlocked()
            target = 1.0 if was_correct else 0.0
            self.acc[vector_name] = (1 - self.alpha) * self.acc[vector_name] + self.alpha * target
            after = self._weights_unlocked()
            self._persist(vector_name, was_correct, incident_id, before, after, context or {})

    def current_weights(self) -> Dict[str, float]:
        with self._lock:
            return self._weights_unlocked()

    def _weights_unlocked(self) -> Dict[str, float]:
        names = list(self.acc.keys())
        raw = np.array([self.prior[n] * np.exp(self.sharpness * self.acc[n]) for n in names])
        w = raw / raw.sum()
        return dict(zip(names, (float(x) for x in w)))

    def _persist(self, vec, ok, incident_id, before, after, context):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({"acc": self.acc}))
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"ts": time.time(), "incident_id": incident_id, "vector": vec, "was_correct": ok,
                     "acc": dict(self.acc), "weights_before": before, "weights_after": after, **context}
            with self.audit_path.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
        logger.info(f"[adaptive-weights] {vec} correct={ok} -> {({k: round(v, 3) for k, v in after.items()})}")


def default_tracker() -> AdaptiveWeightTracker:
    return AdaptiveWeightTracker(state_path=DATA_DIR / "weights_state.json", audit_path=DATA_DIR / "weight_audit.jsonl")
