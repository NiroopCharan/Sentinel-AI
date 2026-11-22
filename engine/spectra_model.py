"""SPECTRA-CRBAC v2: Model architecture and pickle loader.

Defines the SpectraCrbacV2 and SpectraCrbacDetector classes required for
loading serialized model bundles without namespace errors, and provides
the load_spectra_bundle helper.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


class SpectraCrbacV2:
    """SPECTRA-CRBAC v2: Sequential Provenance-Evidential Conformal Temporal Risk Architecture.
    
    Unifies:
      Engine A: Cost-sensitive threat discriminator (HistGradientBoosting + ExtraTrees ensemble).
      Engine B: Directional robust anomaly manifold (Median + MAD z-score quantile mapping).
      Engine C: Leaky sequential CUSUM accumulator (Log-Likelihood Ratio martingale).
    """

    def __init__(
        self,
        random_state: int = 42,
        alpha_threat: float = 0.50,
        alpha_anomaly: float = 0.10,
        alpha_evidence: float = 0.40,
        lambda_decay: float = 0.88,
        beta_scale: float = 0.25,
        prior_p0: float = 0.023,
    ):
        self.random_state = random_state
        self.alpha_threat = alpha_threat
        self.alpha_anomaly = alpha_anomaly
        self.alpha_evidence = alpha_evidence
        self.lambda_decay = lambda_decay
        self.beta_scale = beta_scale
        self.prior_p0 = prior_p0

        self.clf_hgb: Optional[HistGradientBoostingClassifier] = None
        self.clf_et: Optional[ExtraTreesClassifier] = None
        self.benign_center: Optional[np.ndarray] = None
        self.benign_scale: Optional[np.ndarray] = None
        self.val_benign_anomaly_scores: Optional[np.ndarray] = None

        self.threshold_optimal: float = 0.50
        self.threshold_conformal_1pct: float = 0.50
        self.consecutive_k: int = 2

    def _threat_prob(self, X: np.ndarray) -> np.ndarray:
        p1 = self.clf_hgb.predict_proba(X)[:, 1]
        p2 = self.clf_et.predict_proba(X)[:, 1]
        return 0.60 * p1 + 0.40 * p2

    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((X - self.benign_center) / self.benign_scale)
        return np.median(z, axis=1)

    def _emp_quantile(self, raw: np.ndarray, ref: np.ndarray) -> np.ndarray:
        ref_s = np.sort(ref)
        return np.searchsorted(ref_s, raw, side="right") / len(ref_s)

    def _sequential_evidence(self, probs: np.ndarray, rec_ids: np.ndarray) -> np.ndarray:
        llr = np.log((probs + 1e-7) / (1.0 - probs + 1e-7)) - np.log(self.prior_p0 / (1.0 - self.prior_p0))
        evidence = np.zeros_like(probs)
        df_tmp = pd.DataFrame({"rid": rec_ids, "llr": llr})

        for rid, group in df_tmp.groupby("rid"):
            idx = group.index.values
            llr_vals = group["llr"].values
            s = 0.0
            ev = np.zeros(len(llr_vals))
            for i, val in enumerate(llr_vals):
                s = max(0.0, self.lambda_decay * s + val)
                ev[i] = 1.0 - np.exp(-self.beta_scale * s)
            evidence[idx] = ev
        return evidence

    def score_samples(self, X: np.ndarray, rec_ids: np.ndarray) -> np.ndarray:
        p_threat = self._threat_prob(X)
        raw_ano = self._anomaly_score(X)
        q_ano = self._emp_quantile(raw_ano, self.val_benign_anomaly_scores)
        ev = self._sequential_evidence(p_threat, rec_ids)
        risk = self.alpha_threat * p_threat + self.alpha_anomaly * q_ano + self.alpha_evidence * ev
        return np.clip(risk, 0.0, 1.0)


# Alias for backward-compatibility with older bundles
SpectraCrbacDetector = SpectraCrbacV2


def _register_classes_for_unpickling():
    """Ensure pickle can find SpectraCrbacV2 and SpectraCrbacDetector regardless of calling module."""
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        if not hasattr(main_mod, "SpectraCrbacV2"):
            setattr(main_mod, "SpectraCrbacV2", SpectraCrbacV2)
        if not hasattr(main_mod, "SpectraCrbacDetector"):
            setattr(main_mod, "SpectraCrbacDetector", SpectraCrbacDetector)


def load_spectra_bundle(model_path: str | Path) -> Dict[str, Any]:
    """Loads the serialized SPECTRA-CRBAC joblib bundle safely."""
    _register_classes_for_unpickling()
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"SPECTRA model bundle not found at: {path}")
    return joblib.load(path)
