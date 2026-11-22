"""NSL-KDD loading, feature sets and preprocessing (build prompt section 8.1).

Real data: data/raw/KDDTrain+.txt and data/raw/KDDTest+.txt (run scripts/download_nsl_kdd.sh).
Synthetic data exists ONLY so the pipeline can be smoke-tested without the dataset; it is never used
unless --allow-synthetic is passed, and every use logs a loud warning (ground rule 2).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import RAW_DIR
from utils.log import logger

COLUMNS: List[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL_ALL = ["protocol_type", "service", "flag"]
NUMERIC_ALL = [c for c in COLUMNS if c not in CATEGORICAL_ALL + ["label", "difficulty"]]

# The seven features named in the build prompt. These are the only ones the live sensors can approximate
# from a running container (sensors/flow_features.py), so the DEFAULT model uses exactly these (+ protocol).
CORE_FEATURES = ["duration", "src_bytes", "dst_bytes", "count", "srv_count", "same_srv_rate", "diff_srv_rate"]

FEATURE_SETS = {
    "core": {"numeric": CORE_FEATURES, "categorical": ["protocol_type"]},
    "full": {"numeric": NUMERIC_ALL, "categorical": CATEGORICAL_ALL},
}


def feature_columns(feature_set: str) -> List[str]:
    fs = FEATURE_SETS[feature_set]
    return fs["numeric"] + fs["categorical"]


def load_nsl_kdd(path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["attack_type"] = df["label"].astype(str).str.strip()
    df["y"] = (df["attack_type"] != "normal").astype(int)  # 0 = normal, 1 = attack
    return df


def real_data_available() -> bool:
    return (RAW_DIR / "KDDTrain+.txt").exists() and (RAW_DIR / "KDDTest+.txt").exists()


def make_synthetic_nsl_kdd(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic stand-in for smoke tests. NOT NSL-KDD. Distributions are invented."""
    logger.warning("SYNTHETIC DEMO DATA — not NSL-KDD. Run scripts/download_nsl_kdd.sh for real evaluation.")
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    df = pd.DataFrame({c: np.zeros(n) for c in NUMERIC_ALL})
    normal, attack = y == 0, y == 1
    df["duration"] = np.where(normal, rng.exponential(30, n), rng.exponential(2, n))
    df["src_bytes"] = np.where(normal, rng.lognormal(6, 1.5, n), rng.lognormal(3, 1.0, n))
    df["dst_bytes"] = np.where(normal, rng.lognormal(7, 1.5, n), rng.lognormal(2, 1.0, n))
    df["count"] = np.where(normal, rng.poisson(6, n), rng.poisson(150, n))
    df["srv_count"] = np.where(normal, rng.poisson(5, n), rng.poisson(140, n))
    same = np.where(normal, rng.beta(5, 2, n), rng.beta(9, 1, n))
    df["same_srv_rate"] = same
    df["diff_srv_rate"] = 1 - same
    flip = rng.random(n) < 0.12  # label noise so the problem is not trivially separable
    y = np.where(flip, 1 - y, y)
    df["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n, p=[0.8, 0.15, 0.05])
    df["service"] = rng.choice(["http", "ssh", "ftp", "other"], n)
    df["flag"] = rng.choice(["SF", "S0", "REJ"], n)
    df["attack_type"] = np.where(y == 0, "normal", "synthetic_attack")
    df["label"], df["difficulty"], df["y"] = df["attack_type"], 0, y
    return df[COLUMNS + ["attack_type", "y"]]


def load_train_test(allow_synthetic: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Returns (train, test, provenance) where provenance is 'real' or 'synthetic'."""
    if real_data_available():
        return load_nsl_kdd(RAW_DIR / "KDDTrain+.txt"), load_nsl_kdd(RAW_DIR / "KDDTest+.txt"), "real"
    if not allow_synthetic:
        raise FileNotFoundError(
            "NSL-KDD not found in data/raw/. Run scripts/download_nsl_kdd.sh first "
            "(or pass --allow-synthetic for a clearly-labelled smoke test)."
        )
    return make_synthetic_nsl_kdd(8000, seed=1), make_synthetic_nsl_kdd(3000, seed=2), "synthetic"


def split_train_val(train: pd.DataFrame, val_size: float = 0.2, seed: int = 42):
    """Deterministic stratified split of the TRAIN partition: fit on `fit`, tune tau on `val`."""
    fit, val = train_test_split(train, test_size=val_size, stratify=train["y"], random_state=seed)
    return fit.reset_index(drop=True), val.reset_index(drop=True)


def _log1p_clip(x):
    return np.log1p(np.clip(x, 0, None))  # module-level so joblib can pickle it


def build_preprocessor(feature_set: str = "core") -> ColumnTransformer:
    fs = FEATURE_SETS[feature_set]
    numeric = Pipeline([
        ("log", FunctionTransformer(_log1p_clip, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [("num", numeric, fs["numeric"]),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), fs["categorical"])],
        sparse_threshold=0.0,
    )


def class_balance(df: pd.DataFrame) -> dict:
    n_attack = int(df["y"].sum())
    return {"n": int(len(df)), "normal": int(len(df) - n_attack), "attack": n_attack,
            "attack_fraction": round(n_attack / max(len(df), 1), 4)}
