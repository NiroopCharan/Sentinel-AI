"""Feature mapping & extraction for SPECTRA-CRBAC v2 (103-feature window schema).

Extracts or maps telemetry features into the exact schema expected by the
SPECTRA model bundle, converting Falco/syscall events or direct feature dicts.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Typical syscall mapping for Falco rule / kernel events
_FALCO_SYSCALL_MAP = {
    "open": "syscall_open",
    "openat": "syscall_openat",
    "execve": "syscall_execve",
    "execveat": "syscall_execveat",
    "connect": "syscall_connect",
    "accept": "syscall_accept",
    "read": "syscall_read",
    "write": "syscall_write",
    "close": "syscall_close",
    "clone": "syscall_clone",
    "fork": "syscall_fork",
    "vfork": "syscall_vfork",
    "socket": "syscall_socket",
    "bind": "syscall_bind",
    "listen": "syscall_listen",
    "recvfrom": "network_recvfrom",
    "sendto": "network_sendto",
    "getsockname": "network_getsockname",
    "getpeername": "network_getpeername",
    "access": "syscall_access",
    "stat": "syscall_stat",
    "fstat": "syscall_fstat",
    "lstat": "syscall_lstat",
    "unlink": "syscall_unlink",
    "unlinkat": "syscall_unlinkat",
    "mprotect": "syscall_mprotect",
    "brk": "syscall_brk",
    "ioctl": "syscall_ioctl",
}


def build_feature_vector(
    event: Dict[str, Any],
    feature_names: Sequence[str],
) -> Optional[np.ndarray]:
    """Builds a 1D or 2D feature array matching `feature_names`.
    
    Returns None if the event contains no SPECTRA/syscall telemetry.
    """
    # 1. Direct numpy array or list under "spectra_vector"
    if "spectra_vector" in event and event["spectra_vector"] is not None:
        arr = np.asarray(event["spectra_vector"], dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == len(feature_names):
            return arr

    # 2. Direct feature dictionary under "spectra", "spectra_features", or "syscall_features"
    feat_dict = (
        event.get("spectra_features")
        or event.get("spectra")
        or event.get("syscall_features")
    )
    if isinstance(feat_dict, dict):
        vec = np.zeros(len(feature_names), dtype=float)
        for i, name in enumerate(feature_names):
            vec[i] = float(feat_dict.get(name, 0.0))
        return vec.reshape(1, -1)

    # 3. Derive from Falco events / syscall stream if explicit syscall telemetry is present
    falco_events = event.get("falco")
    if isinstance(falco_events, list) and len(falco_events) > 0:
        counts: Dict[str, float] = {}
        has_syscall = False
        completed = 0
        failed = 0
        for fe in falco_events:
            if not isinstance(fe, dict):
                continue
            syscall = str(fe.get("syscall", "")).lower()
            rule = str(fe.get("rule", "")).lower()
            prio = str(fe.get("priority", "")).lower()

            matched_col = None
            if syscall and syscall in _FALCO_SYSCALL_MAP:
                matched_col = _FALCO_SYSCALL_MAP[syscall]
                has_syscall = True
            elif "syscall" in fe and fe["syscall"]:
                has_syscall = True

            if matched_col:
                counts[matched_col] = counts.get(matched_col, 0.0) + 1.0

            if prio in ("error", "critical", "alert", "emergency"):
                failed += 1
            completed += 1

        if not has_syscall:
            return None

        counts["completed_syscalls"] = float(completed)
        counts["failed_syscalls"] = float(failed)
        counts["mean_syscall_latency_ms"] = float(event.get("resources", {}).get("latency_ms", 1.0))

        vec = np.zeros(len(feature_names), dtype=float)
        for i, name in enumerate(feature_names):
            vec[i] = float(counts.get(name, 0.0))
        return vec.reshape(1, -1)

    return None
