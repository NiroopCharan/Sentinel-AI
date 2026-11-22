"""SPECTRA-CRBAC v2 Engine: Real-time 3-Engine Telemetry Risk Assessor.

Combines:
  Engine A: Cost-sensitive threat discriminator (p_threat)
  Engine B: Directional robust anomaly manifold (q_ano)
  Engine C: Leaky sequential CUSUM accumulator (ev)
into a unified SPECTRA Risk Score with conformal thresholding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from engine.spectra_features import build_feature_vector
from engine.spectra_model import load_spectra_bundle
from utils.log import logger


class SpectraEngine:
    """Evaluates host/container kernel telemetry using the calibrated SPECTRA-CRBAC v2 model."""

    def __init__(self, model_path: Union[str, Path] = "models/spectra_model.joblib"):
        self.model_path = Path(model_path)
        self.active = False
        self.model = None
        self.imputer = None
        self.scaler = None
        self.features: list[str] = []
        self.config: dict[str, Any] = {}
        self.cusum_state: Dict[str, float] = {}

        try:
            bundle = load_spectra_bundle(self.model_path)
            self.model = bundle.get("model") or bundle.get("detector")
            self.imputer = bundle.get("imputer")
            self.scaler = bundle.get("scaler")
            self.features = bundle.get("features", [])
            self.config = bundle.get("config", {})

            f_weights = self.config.get("fusion_weights", {})
            self.alpha_threat = float(f_weights.get("threat_estimator", 0.40))
            self.alpha_ano = float(f_weights.get("anomaly_manifold", 0.10))
            self.alpha_ev = float(f_weights.get("sequential_evidence", 0.50))

            self.lambda_decay = 0.88
            self.beta_scale = 0.25
            self.prior_p0 = 0.023

            thresholds = self.config.get("thresholds", {})
            self.threshold = float(thresholds.get("optimal_f1_mcc_threshold", 0.7724))
            self.conformal_threshold = float(thresholds.get("conformal_1pct_fpr_threshold", 0.7669))
            self.active = True
            logger.info(
                f"SPECTRA-CRBAC Engine loaded from {self.model_path} "
                f"({len(self.features)} features, threshold={self.threshold:.4f})"
            )
        except Exception as exc:
            logger.warning(
                f"SPECTRA-CRBAC model could not be initialized from {self.model_path} ({exc}). "
                "SPECTRA engine will be inactive."
            )

    def evaluate_vector(self, session_id: str, feature_vector: np.ndarray) -> dict:
        """Runs the 3-engine SPECTRA assessment on incoming feature window."""
        if not self.active or self.model is None:
            return {"active": False, "spectra_risk": 0.0, "is_anomalous": False}

        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Impute missing fields using feature names to avoid sklearn warnings
        if self.imputer is not None:
            if hasattr(self.imputer, "feature_names_in_"):
                df_in = pd.DataFrame(feature_vector, columns=self.features)
                X = self.imputer.transform(df_in)
            else:
                X = self.imputer.transform(feature_vector)
        else:
            X = feature_vector

        # Engine A: Dual-tree threat discriminator
        p_threat = float(self.model._threat_prob(X)[0])

        # Engine B: Directional robust anomaly manifold
        raw_ano = self.model._anomaly_score(X)
        ref_scores = getattr(self.model, "val_benign_anomaly_scores", None)
        if ref_scores is not None and len(ref_scores) > 0:
            q_ano = float(self.model._emp_quantile(raw_ano, ref_scores)[0])
        else:
            q_ano = float(np.clip(raw_ano[0], 0.0, 1.0))

        # Engine C: Leaky sequential CUSUM accumulator
        llr = np.log((p_threat + 1e-7) / (1.0 - p_threat + 1e-7)) - np.log(self.prior_p0 / (1.0 - self.prior_p0))
        prev_s = self.cusum_state.get(session_id, 0.0)
        curr_s = max(0.0, self.lambda_decay * prev_s + float(llr))
        self.cusum_state[session_id] = curr_s
        ev = float(1.0 - np.exp(-self.beta_scale * curr_s))

        # Fused internal risk score
        risk = float(np.clip(
            self.alpha_threat * p_threat + self.alpha_ano * q_ano + self.alpha_ev * ev,
            0.0, 1.0,
        ))

        return {
            "active": True,
            "spectra_risk": round(risk, 4),
            "threat_prob": round(p_threat, 4),
            "anomaly_quantile": round(q_ano, 4),
            "evidence_score": round(ev, 4),
            "is_anomalous": risk >= self.threshold,
            "threshold": round(self.threshold, 4),
        }

    def evaluate_event(self, session_id: str, event: dict) -> Optional[dict]:
        """Extracts features from the event (if available) and scores it."""
        if not self.active:
            return None
        vec = build_feature_vector(event, self.features)
        if vec is None:
            return None
        return self.evaluate_vector(session_id, vec)

    def reset_session(self, session_id: str):
        """Clears the sequential CUSUM accumulator state for a container or session."""
        self.cusum_state.pop(session_id, None)