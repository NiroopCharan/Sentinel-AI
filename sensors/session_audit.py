"""Parse the in-container SSH session audit log written by victim/sentinel-session (see victim/Dockerfile).

Line format (pipe separated; the command may itself contain pipes, so we split at most 6 times):
  <epoch>|START|<session_id>|<user>|<client_ip>|<parent_comm>
  <epoch>|CMD|<session_id>|<user>|<client_ip>|<lineage>|<command>
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

AUDIT_LOG_PATH = "/var/log/sentinel_audit.log"


def parse_audit_log(text: str) -> Dict[str, dict]:
    sessions: Dict[str, dict] = {}
    for line in text.splitlines():
        parts = line.split("|", 6)
        if len(parts) < 6:
            continue
        try:
            ts = float(parts[0])
        except ValueError:
            continue
        kind, sid, user, ip = parts[1], parts[2], parts[3], parts[4]
        s = sessions.setdefault(sid, {"session_id": sid, "user": user, "source_ip": ip, "start_ts": ts,
                                      "parent": "", "cmd_ts": [], "lineage": "", "commands": []})
        if kind == "START":
            s["start_ts"], s["parent"] = ts, parts[5]
        elif kind == "CMD" and len(parts) == 7:
            s["cmd_ts"].append(ts)
            s["commands"].append(parts[6])
            s["lineage"] = s["lineage"] or parts[5]
        s["last_ts"] = max(s.get("last_ts", ts), ts)
    for s in sessions.values():
        s["intervals"] = [round(b - a, 3) for a, b in zip(s["cmd_ts"], s["cmd_ts"][1:])]
        s["start_hour"] = _hour(s["start_ts"])
        s["n_commands"] = len(s["cmd_ts"])
        if not s["lineage"] and s["parent"]:
            s["lineage"] = f"{s['parent']}>bash"
    return sessions


def _hour(ts: float) -> float:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)   # container clock is UTC
    return d.hour + d.minute / 60.0


def active_session(sessions: Dict[str, dict], now: Optional[float] = None, idle_s: float = 90.0) -> Optional[dict]:
    """Most recently active session with at least 2 commands (needed for a timing signal), else None."""
    now = time.time() if now is None else now
    live = [s for s in sessions.values() if now - s.get("last_ts", 0) <= idle_s and s["n_commands"] >= 2]
    return max(live, key=lambda s: s["last_ts"]) if live else None
