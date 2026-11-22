"""Forensic snapshots + MITRE ATT&CK tagging (build prompt section 7 / step 7).

snapshot_container() captures, BEFORE the container is killed (and again on the healed replica):
  * per-process cmdline / status / /proc/<pid>/maps  (collected with `docker exec`, so it works with Docker Desktop
    where container PIDs are not visible in the WSL distro; if a host PID is readable it is dumped too)
  * /proc/net/tcp(6), filesystem diff vs. the image (docker diff), container logs, SSH session audit log, inspect JSON
The artifacts are packed into a tar.gz and encrypted with Fernet (key in data/forensics/forensics.key, mode 600).
A small PLAINTEXT summary.json (counts, process names, ports, MITRE tags, sha256 of the artifact) is written next to it
for the dashboard's before/after viewer. Command lines in the summary are truncated to 120 chars.

Decrypt:  python -m forensics.forensic_snapshot --decrypt data/forensics/INC-x_before.tar.gz.enc --out /tmp/x.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FORENSICS_DIR  # noqa: E402
from sensors.flow_features import parse_proc_net_tcp  # noqa: E402
from sensors.session_audit import AUDIT_LOG_PATH  # noqa: E402

# Falco rule-name substring -> (technique id, name, tactic). Matching is case-insensitive substring.
MITRE_RULE_MAP: Dict[str, tuple] = {
    "read sensitive file": ("T1003.008", "OS Credential Dumping: /etc/passwd and /etc/shadow", "Credential Access"),
    "terminal shell in container": ("T1059.004", "Command and Scripting Interpreter: Unix Shell", "Execution"),
    "run shell untrusted": ("T1059.004", "Command and Scripting Interpreter: Unix Shell", "Execution"),
    "suspicious network tool": ("T1046", "Network Service Discovery", "Discovery"),
    "crypto miner": ("T1496", "Resource Hijacking", "Impact"),
    "stratum": ("T1496", "Resource Hijacking", "Impact"),
    "drop and execute new binary": ("T1105", "Ingress Tool Transfer", "Command and Control"),
    "netcat remote code execution": ("T1059", "Command and Scripting Interpreter", "Execution"),
}


def _t(tid, name, tactic, source):
    return {"technique": tid, "name": name, "tactic": tactic, "source": source}


def tag_event(falco_events: List[dict], scores: Dict[str, Optional[float]], flow: Optional[dict] = None,
              resources: Optional[dict] = None) -> List[dict]:
    """MITRE tags: authoritative ones come from Falco rule names; the rest are explicit HEURISTIC hints from
    vector values and are labelled source='vector-heuristic' so nobody mistakes them for sensor-confirmed facts."""
    tags: Dict[str, dict] = {}
    for e in falco_events or []:
        rule = e.get("rule", "").lower()
        for needle, (tid, name, tactic) in MITRE_RULE_MAP.items():
            if needle in rule:
                tags[tid] = _t(tid, name, tactic, "falco:" + e.get("rule", ""))
    if flow:
        if flow.get("count", 0) >= 20 and flow.get("same_srv_rate", 0) >= 0.9:
            tags.setdefault("T1498.001", _t("T1498.001", "Network Denial of Service: Direct Network Flood", "Impact", "vector-heuristic"))
        if flow.get("count", 0) >= 8 and flow.get("diff_srv_rate", 0) >= 0.5:
            tags.setdefault("T1046", _t("T1046", "Network Service Discovery", "Discovery", "vector-heuristic"))
    if resources and (scores.get("rds") or 0) >= 0.3 and resources.get("cpu_util", 0) >= 80:
        tags.setdefault("T1496", _t("T1496", "Resource Hijacking", "Impact", "vector-heuristic"))
    if (scores.get("ics") or 0) >= 0.6:
        tags.setdefault("T1078", _t("T1078", "Valid Accounts", "Defense Evasion / Initial Access", "vector-heuristic"))
    return list(tags.values())


# ---------------------------------------------------------------- crypto

def _key(forensics_dir: Path) -> bytes:
    path = forensics_dir / "forensics.key"
    if not path.exists():
        forensics_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(Fernet.generate_key())
    return path.read_bytes()


def decrypt_snapshot(enc_path, forensics_dir: Path = FORENSICS_DIR) -> bytes:
    return Fernet(_key(Path(forensics_dir))).decrypt(Path(enc_path).read_bytes())


# ---------------------------------------------------------------- collection

_PROC_SCRIPT = (
    'for p in /proc/[0-9]*; do pid=${p#/proc/}; echo "=== PID $pid"; echo "--- cmdline"; '
    "tr '\\0' ' ' < $p/cmdline 2>/dev/null; echo; echo '--- status'; head -n 9 $p/status 2>/dev/null; "
    "echo '--- maps'; cat $p/maps 2>/dev/null; done"
)


def _exec(container, cmd) -> str:
    try:
        r = container.exec_run(cmd)
        return r.output.decode("utf-8", "replace") if r.exit_code == 0 else ""
    except Exception:
        return ""


def parse_processes(raw: str) -> List[dict]:
    procs, cur, section = [], None, None
    for line in raw.splitlines():
        if line.startswith("=== PID "):
            cur = {"pid": int(line.split()[2]), "name": "", "cmdline": "", "map_count": 0}
            procs.append(cur); section = None
        elif line.startswith("--- ") and cur is not None:
            section = line[4:].strip()
        elif cur is not None:
            if section == "cmdline" and line.strip():
                cur["cmdline"] = line.strip()[:120]
            elif section == "status" and line.startswith("Name:"):
                cur["name"] = line.split(":", 1)[1].strip()
            elif section == "maps" and line.strip():
                cur["map_count"] += 1
    return procs


def snapshot_container(container, incident_id: str, label: str, host_pid: Optional[int] = None,
                       mitre: Optional[List[dict]] = None, forensics_dir: Path = FORENSICS_DIR) -> dict:
    """Capture + encrypt a snapshot. `label` is 'before' (compromised) or 'after' (healed replica)."""
    forensics_dir = Path(forensics_dir)
    forensics_dir.mkdir(parents=True, exist_ok=True)
    raw_proc = _exec(container, ["sh", "-c", _PROC_SCRIPT])
    net = _exec(container, ["cat", "/proc/net/tcp"]) + "\n" + _exec(container, ["cat", "/proc/net/tcp6"])
    audit = _exec(container, ["tail", "-n", "500", AUDIT_LOG_PATH])
    try:
        diff = container.diff() or []
    except Exception:
        diff = []
    try:
        logs = container.logs(tail=200).decode("utf-8", "replace")
    except Exception:
        logs = ""
    host_maps = ""
    if host_pid:
        try:
            host_maps = Path(f"/proc/{host_pid}/maps").read_text()
        except OSError:
            host_maps = ""   # e.g. Docker Desktop: container PIDs live in another VM/distro

    procs = parse_processes(raw_proc)
    conns = parse_proc_net_tcp(net)
    kinds = {0: "changed", 1: "added", 2: "deleted"}
    fs = {"changed": 0, "added": 0, "deleted": 0}
    for d in diff:
        fs[kinds.get(d.get("Kind"), "changed")] += 1

    files = {"processes.txt": raw_proc, "net_tcp.txt": net, "session_audit.log": audit,
             "fs_diff.json": json.dumps(diff, indent=2), "container_logs.txt": logs,
             "inspect.json": json.dumps(getattr(container, "attrs", {}), indent=2, default=str)}
    if host_maps:
        files["host_maps.txt"] = host_maps
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), int(time.time())
            tar.addfile(info, io.BytesIO(data))
    token = Fernet(_key(forensics_dir)).encrypt(buf.getvalue())
    enc_path = forensics_dir / f"{incident_id}_{label}.tar.gz.enc"
    enc_path.write_bytes(token)

    summary = {
        "incident_id": incident_id, "label": label, "captured_at": time.time(),
        "container": getattr(container, "name", ""), "image": ((container.attrs or {}).get("Config") or {}).get("Image", ""),
        "process_count": len(procs),
        "processes": [{"pid": p["pid"], "name": p["name"], "cmdline": p["cmdline"], "map_count": p["map_count"]} for p in procs],
        "listening_ports": sorted({c["local_port"] for c in conns if c["state"] == "LISTEN"}),
        "established": sum(c["state"] == "ESTABLISHED" for c in conns),
        "syn_recv": sum(c["state"] == "SYN_RECV" for c in conns),
        "remote_ips": sorted({c["remote_ip"] for c in conns if c["state"] == "ESTABLISHED"}),
        "fs_changes": {**fs, "sample": [d.get("Path") for d in diff[:15]]},
        "session_commands": audit.count("|CMD|"),
        "mitre": mitre or [], "artifact": str(enc_path), "artifact_sha256": hashlib.sha256(token).hexdigest(),
    }
    summary_path = forensics_dir / f"{incident_id}_{label}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


def load_summaries(incident_id: str, forensics_dir: Path = FORENSICS_DIR) -> dict:
    out = {}
    for label in ("before", "after"):
        p = Path(forensics_dir) / f"{incident_id}_{label}.summary.json"
        if p.exists():
            out[label] = json.loads(p.read_text())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decrypt a Sentinel-AI forensic snapshot")
    ap.add_argument("--decrypt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).write_bytes(decrypt_snapshot(a.decrypt))
    print(f"wrote {a.out} (tar.gz)")
