"""Record owner / intruder SSH sessions into data/identity_sessions.csv (build prompt section 8.3).

Reads the in-container audit log (victim/sentinel-session) and appends one row per session.

  python -m utils.session_logger --container sentinel-victim-pod --label owner
  python -m utils.session_logger --container sentinel-victim-pod --label intruder --scenario hydra+ssh

--simulate-hour H   overrides the start hour and marks hour_simulated=1 (use when you cannot run at the real hour)
--source-ip IP      overrides the recorded origin and marks origin_simulated=1 (single-machine collection)
Overrides are flagged in the CSV so nothing simulated is ever mistaken for observed.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR  # noqa: E402
from engine.identity_model import CSV_COLUMNS  # noqa: E402
from sensors.session_audit import AUDIT_LOG_PATH, parse_audit_log  # noqa: E402
from utils.log import logger  # noqa: E402

CSV_PATH = DATA_DIR / "identity_sessions.csv"


def _existing_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as fh:
        return {(r["container"], r["session_id"]) for r in csv.DictReader(fh)}


def log_sessions(container, label: str, scenario: str = "", simulate_hour=None, source_ip=None,
                 csv_path: Path = CSV_PATH, identity: str = "demoUser") -> int:
    res = container.exec_run(["cat", AUDIT_LOG_PATH])
    sessions = parse_audit_log(res.output.decode("utf-8", "replace"))
    seen = _existing_keys(csv_path)
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            w.writeheader()
        for s in sessions.values():
            key = (container.name, s["session_id"])
            if key in seen or s["n_commands"] < 2:
                continue
            w.writerow({
                "session_id": s["session_id"], "container": container.name, "label": label, "identity": identity,
                "source_ip": source_ip or s["source_ip"], "origin_known": "", "origin_simulated": int(bool(source_ip)),
                "start_hour": simulate_hour if simulate_hour is not None else round(s["start_hour"], 3),
                "hour_simulated": int(simulate_hour is not None), "lineage": s["lineage"],
                "cmd_intervals": ";".join(str(x) for x in s["intervals"]), "n_commands": s["n_commands"],
                "scenario": scenario, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            added += 1
    logger.info(f"identity_sessions.csv: +{added} {label} session(s) from {container.name}")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--label", choices=["owner", "intruder"], required=True)
    ap.add_argument("--scenario", default="")
    ap.add_argument("--simulate-hour", type=float)
    ap.add_argument("--source-ip")
    a = ap.parse_args()
    import docker
    log_sessions(docker.from_env().containers.get(a.container), a.label, a.scenario, a.simulate_hour, a.source_ip)


if __name__ == "__main__":
    main()
