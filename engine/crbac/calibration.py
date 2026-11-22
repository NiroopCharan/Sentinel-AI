"""Split-conformal p-values and bounded-loss CRC threshold selection."""
import math
import numpy as np

def conformal_p_value(calibration_scores, test_score: float) -> float:
    """Upper-tail split-conformal p-value; scores must be exchangeable under null."""
    scores = np.asarray(calibration_scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("calibration_scores must be a non-empty finite 1-D array")
    if not math.isfinite(test_score): raise ValueError("test_score must be finite")
    return float((1 + np.count_nonzero(scores >= test_score)) / (len(scores) + 1))


def crc_select_threshold(losses_by_threshold: dict[float, list[float]], epsilon: float, max_loss: float = 1.0):
    """Select least conservative feasible threshold in a nested policy family.

    Uses the finite-sample CRC upper estimate (sum(losses)+B)/(n+1).
    Caller must establish exchangeable calibration episodes, bounded losses,
    and monotonicity of the threshold-indexed policy/loss family.
    """
    if not 0 < epsilon < max_loss: raise ValueError("epsilon must be in (0, max_loss)")
    if not losses_by_threshold: raise ValueError("at least one threshold is required")
    candidates = []
    for threshold, losses in losses_by_threshold.items():
        arr = np.asarray(losses, dtype=float)
        if arr.ndim != 1 or len(arr) == 0 or not np.isfinite(arr).all():
            raise ValueError("each threshold needs a non-empty finite loss list")
        if np.any(arr < 0) or np.any(arr > max_loss): raise ValueError("loss outside [0, max_loss]")
        bound = (float(arr.sum()) + max_loss) / (len(arr) + 1)
        if bound <= epsilon: candidates.append((float(threshold), bound))
    if not candidates: return None, None
    # Explicitly choose the smallest threshold only if caller's threshold ordering
    # defines increasing sensitivity; document and test that policy convention.
    return min(candidates, key=lambda x: x[0])
