"""Pure, dependency-light HMR helpers: score fusion, tiering, anomaly-score normalisation.

Kept free of docker / flask imports so training and benchmarking can use exactly the same
fusion code as the live engine.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from config import TIERS


def normalize_anomaly(iso_score_samples, as_min: float, as_max: float):
    """AS = clip((-score - ASmin) / (ASmax - ASmin), 0, 1).

    `iso_score_samples` is IsolationForest.score_samples() output (higher = more normal),
    so -score grows with abnormality.
    """
    neg = -np.asarray(iso_score_samples, dtype=float)
    span = max(as_max - as_min, 1e-9)
    return np.clip((neg - as_min) / span, 0.0, 1.0)


def fuse(scores: Mapping[str, Optional[float]], weights: Mapping[str, float]) -> Tuple[float, Dict[str, float]]:
    """TS = sum_i w_i * V_i over the vectors that are actually available.

    A vector is unavailable (None) when its telemetry does not exist for the event, e.g. no SSH
    session means no Identity Consistency Score; NSL-KDD rows have no RDS/ICS at all. Weights of
    the available vectors are renormalised to sum to 1 so TS stays in [0, 1]. Returns (TS, effective_weights).
    """
    present = {k: float(v) for k, v in scores.items() if v is not None}
    if not present:
        return 0.0, {}
    total = sum(weights[k] for k in present)
    if total <= 0:
        eff = {k: 1.0 / len(present) for k in present}
    else:
        eff = {k: weights[k] / total for k in present}
    ts = sum(eff[k] * present[k] for k in present)
    return float(np.clip(ts, 0.0, 1.0)), eff


def tier_for(ts: float, tau: float, margin: float) -> str:
    """Four-tier graduated response (build prompt section 5)."""
    if ts < tau / 2:
        return TIERS[0]   # Monitor
    if ts < tau:
        return TIERS[1]   # Log & Watch
    if ts < tau + margin:
        return TIERS[2]   # Isolate
    return TIERS[3]       # Kill & Restart


def tier_rank(tier: str) -> int:
    return TIERS.index(tier)

# Add to engine/fusion.py:

def normalize_anomaly_mad(value: float, center: float, mad: float) -> float:
    """Robust Directional Anomaly Normalization using Median and MAD (SPECTRA Engine B).
    
    Prevents extreme outlier distortion on heavy container workloads.
    """
    scale = max(1.4826 * mad, 1e-6)
    z = max(0.0, (value - center) / scale)
    # Map robust z-score smoothly to [0, 1]
    return float(np.clip(1.0 - np.exp(-0.5 * z), 0.0, 1.0))


def update_sequential_evidence(
    prev_s: float, 
    ts: float, 
    prior_p0: float = 0.023, 
    lambda_decay: float = 0.88, 
    beta_scale: float = 0.25
) -> Tuple[float, float]:
    """Leaky Sequential Evidential Accumulator (SPECTRA Engine C).
    
    Computes Log-Likelihood Ratio against benign prior and updates discounted evidence.
    Returns: (new_S, evidence_score in [0, 1])
    """
    prob = np.clip(ts, 1e-4, 1.0 - 1e-4)
    llr = np.log(prob / (1.0 - prob)) - np.log(prior_p0 / (1.0 - prior_p0))
    new_s = max(0.0, lambda_decay * prev_s + llr)
    evidence = float(1.0 - np.exp(-beta_scale * new_s))
    return new_s, float(np.clip(evidence, 0.0, 1.0))