"""Spawn a fresh victim container and run one real scenario against it, logging docker-stats rows for
`collect_resource_baseline.py` and (for identity scenarios) driving real SSH sessions for `session_logger.py`.

Requires the tools named in each scenario's docstring to actually be installed where this runs.

    python -m utils.live_attack_demo --scenario benign
    python -m utils.live_attack_demo --scenario cryptominer
    python -m utils.live_attack_demo --scenario flood --target-host <lan-ip or 127.0.0.1>
    python -m utils.live_attack_demo --scenario credential_theft
    python -m utils.live_attack_demo --scenario session_hijack
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, VICTIM_IMAGE  # noqa: E402
from sensors.resource_monitor import ResourceMonitor  # noqa: E402
from utils.log import logger  # noqa: E402

SCENARIOS = ("benign", "cryptominer", "flood", "credential_theft", "session_hijack")
BASELINE_CSV = DATA_DIR / "resource_baseline.csv"
BASELINE_COLS = ["run_id", "scenario", "label", "t", "cpu_util", "ram_util", "net_io_rate", "container"]


def spawn_victim(client, image: str, host_port: int):
    name = f"sentinel-demo-{uuid.uuid4().hex[:8]}"
    logger.info(f"spawning {name} from {image} (host port {host_port} -> 22)")
    return client.containers.run(image, name=name, detach=True, ports={"22/tcp": host_port}, remove=False)


def _ssh(host, port, user, password, command, timeout=15) -> subprocess.CompletedProcess:
    if not shutil.which("sshpass"):
        raise RuntimeError("sshpass not found (apt install sshpass) - needed to script the owner/intruder SSH sessions")
    cmd = ["sshpass", "-p", password, "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-p", str(port), f"{user}@{host}", command]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def run_benign(container, duration=30):
    time.sleep(duration)


def run_cryptominer(container, duration=30):
    """Requires stress-ng inside the image (victim/Dockerfile installs it) as a CPU-hog proxy for a miner."""
    container.exec_run(["sh", "-c", f"stress-ng --cpu 0 --timeout {duration}s &"], detach=True)
    time.sleep(duration)


def run_flood(target_host, port, duration=20):
    """Requires hping3 on THIS machine, run against a reachable target (not the container's internal IP)."""
    if not shutil.which("hping3"):
        raise RuntimeError("hping3 not found - install it, or run this scenario from the attacker machine per README §Demo")
    logger.warning(f"flooding {target_host}:{port} for {duration}s - get sign-off before running on shared networks")
    subprocess.run(["hping3", "-S", "-p", str(port), "--flood", target_host], timeout=duration + 2,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def run_credential_theft(host, port, user="demoUser", password="demoPass123"):
    _ssh(host, port, user, password, "cat /etc/shadow > /dev/null; id; whoami")


def run_session_hijack(host, port, user="demoUser", password="demoPass123"):
    """Fast, mechanical command bursts - the signature a stolen/scripted session leaves for the ICS to catch."""
    for cmd in ("whoami", "id", "cat /etc/shadow > /dev/null", "ls -la /", "ps aux", "cat /etc/passwd > /dev/null"):
        _ssh(host, port, user, password, cmd, timeout=10)


def collect(client, container, run_id: str, scenario: str, label: int, duration: float, interval=1.0):
    mon, rows, start = ResourceMonitor(client), [], time.time()
    while time.time() - start < duration:
        s = mon.sample(container)
        if s:
            rows.append({"run_id": run_id, "scenario": scenario, "label": label, "t": round(s["ts"] - start, 2),
                        "cpu_util": s["cpu_util"], "ram_util": s["ram_util"], "net_io_rate": s["net_io_rate"],
                        "container": container.name})
        time.sleep(interval)
    return rows


def append_baseline(rows, path: Path = BASELINE_CSV):
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BASELINE_COLS)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    logger.info(f"appended {len(rows)} rows to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=SCENARIOS, required=True)
    ap.add_argument("--image", default=VICTIM_IMAGE)
    ap.add_argument("--host-port", type=int, default=2222)
    ap.add_argument("--target-host", default="127.0.0.1", help="for --scenario flood: a reachable IP, not the container's internal one")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--keep", action="store_true", help="don't remove the victim container afterwards")
    a = ap.parse_args()
    import docker
    client = docker.from_env()
    run_id, c = uuid.uuid4().hex[:10], None
    try:
        c = spawn_victim(client, a.image, a.host_port)
        time.sleep(3)  # let sshd come up
        label = 0 if a.scenario == "benign" else 1
        if a.scenario == "benign":
            rows = collect(client, c, run_id, a.scenario, label, a.duration)
            run_benign(c, 0)
        elif a.scenario == "cryptominer":
            rows = collect(client, c, run_id, a.scenario, label, a.duration)
            run_cryptominer(c, a.duration)
        elif a.scenario == "flood":
            rows = collect(client, c, run_id, a.scenario, label, a.duration)
            run_flood(a.target_host, a.host_port, a.duration)
        elif a.scenario == "credential_theft":
            rows = collect(client, c, run_id, a.scenario, label, max(a.duration, 10))
            run_credential_theft("127.0.0.1", a.host_port)
        else:
            rows = collect(client, c, run_id, a.scenario, label, max(a.duration, 15))
            run_session_hijack("127.0.0.1", a.host_port)
        append_baseline(rows)
    finally:
        if c and not a.keep:
            c.remove(force=True)
            logger.info(f"removed {c.name}")


if __name__ == "__main__":
    main()
