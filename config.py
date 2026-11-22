"""Central configuration for Sentinel-AI. Everything overridable via environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
FORENSICS_DIR = DATA_DIR / "forensics"

VECTORS = ("rf", "as", "rds", "ics")

# Build prompt section 5: starting weights (sum = 1.0). The adaptive tracker modulates these at runtime.
INITIAL_WEIGHTS = {"rf": 0.40, "as": 0.25, "rds": 0.15, "ics": 0.20}

# Decision thresholds. tau is re-estimated (F1-optimal) by engine/train_models.py and stored in
# models/meta.json; these are only the fallbacks (build prompt section 5: tau=0.60, m=0.20).
DEFAULT_TAU = 0.60
DEFAULT_MARGIN = 0.20

TIERS = ("MONITOR", "LOG_WATCH", "ISOLATE", "KILL_RESTART")

ENGINE_URL = os.getenv("SENTINEL_URL", "http://127.0.0.1:5000")
ENGINE_TOKEN = os.getenv("SENTINEL_TOKEN", "")  # optional shared secret for /api/ingest_telemetry
FALCO_LOG = os.getenv("FALCO_LOG", "/var/log/falco/events.log")

VICTIM_IMAGE = os.getenv("VICTIM_IMAGE", "sentinel-victim")
VICTIM_NAME = os.getenv("VICTIM_NAME", "sentinel-victim-pod")
QUARANTINE_NETWORK = "sentinel-quarantine"

# Number of consecutive over-threshold telemetry events required before a disruptive action
# (isolate / kill) fires. Debounces single-sample spikes.
ACT_STREAK = int(os.getenv("SENTINEL_ACT_STREAK", "2"))

# Sessions whose source IP is in this list count as a "known origin" for the Identity Consistency Score.
KNOWN_ORIGINS = [o for o in os.getenv("SENTINEL_KNOWN_ORIGINS", "").split(",") if o]

# --- Dual-Algorithm Configuration (HMR + SPECTRA-CRBAC v2) ---
SPECTRA_MODEL_PATH = MODELS_DIR / "spectra_model.joblib"
# Mode options: 'HYBRID' (both active, fused), 'HMR_ONLY', 'SPECTRA_ONLY'
DUAL_ALGORITHM_MODE = os.getenv("SENTINEL_ALGORITHM_MODE", "HYBRID").upper()
# Weight given to SPECTRA in HYBRID composite threat score (0.0 to 1.0)
# Threat Score = (1.0 - SPECTRA_WEIGHT) * TS_HMR + SPECTRA_WEIGHT * TS_SPECTRA
SPECTRA_WEIGHT = float(os.getenv("SENTINEL_SPECTRA_WEIGHT", "0.50"))
# Disruptive containment policy: 'consensus' (both must agree or one extreme streak) vs 'any'
DISRUPTIVE_ACTION_POLICY = os.getenv("SENTINEL_DISRUPTIVE_POLICY", "consensus").lower()
