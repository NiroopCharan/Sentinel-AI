"""Stage 1 - Sense. Combines Falco (eBPF) events with live docker stats, /proc/net flow state and the SSH
session audit log into one telemetry event per container per poll, and POSTs it to the engine.

    python -m sensors.telemetry_ingestor --container sentinel-victim-pod --engine-url http://127.0.0.1:5000

We ingest kernel events through Falco's eBPF driver; this project does not ship custom eBPF probes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ENGINE_TOKEN, ENGINE_URL, FALCO_LOG, VICTIM_NAME  # noqa: E402
from sensors.flow_features import FlowFeatureExtractor, parse_proc_net_tcp  # noqa: E402
from sensors.resource_monitor import ResourceMonitor  # noqa: E402
from sensors.session_audit import AUDIT_LOG_PATH, active_session, parse_audit_log  # noqa: E402
from utils.log import logger  # noqa: E402


class FalcoTail:
    """Follows Falco's JSON-lines output file; survives rotation/truncation."""

    def __init__(self, path: str):
        self.path, self._fh, self._ino = path, None, None

    def _open(self):
        try:
            self._fh = open(self.path, "r")
            self._fh.seek(0, os.SEEK_END)
            self._ino = os.fstat(self._fh.fileno()).st_ino
        except OSError:
            self._fh = None

    def drain(self) -> List[dict]:
        if self._fh is None:
            self._open()
            return []
        try:
            if os.stat(self.path).st_ino != self._ino:   # rotated
                self._fh.close(); self._open()
                return []
        except OSError:
            return []
        events = []
        for line in self._fh.readlines():
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            f = raw.get("output_fields", {})
            events.append({"rule": raw.get("rule", ""), "priority": raw.get("priority", ""), "output": raw.get("output", ""),
                           "time": raw.get("time", ""), "tags": raw.get("tags", []),
                           "container_id": str(f.get("container.id", "")), "container_name": str(f.get("container.name", "")),
                           "proc_name": f.get("proc.name", ""), "proc_cmdline": f.get("proc.cmdline", ""),
                           "proc_pname": f.get("proc.pname", ""), "user": f.get("user.name", ""),
                           "syscall": str(f.get("evt.type") or f.get("syscall") or raw.get("syscall") or "")})
        return events


def _exec(container, cmd: List[str]) -> str:
    try:
        res = container.exec_run(cmd, demux=False)
        return res.output.decode("utf-8", "replace") if res.exit_code == 0 else ""
    except Exception:
        return ""


class TelemetryIngestor:
    def __init__(self, docker_client, names: List[str], engine_url: str = ENGINE_URL, interval: float = 1.0,
                 falco_log: str = FALCO_LOG):
        self.client, self.names, self.url, self.interval = docker_client, names, engine_url.rstrip("/"), interval
        self.monitor = ResourceMonitor(docker_client)
        self.flows: Dict[str, FlowFeatureExtractor] = {}
        self.falco = FalcoTail(falco_log)
        self._sent_sessions: Dict[str, float] = {}
        self.headers = {"X-Sentinel-Token": ENGINE_TOKEN} if ENGINE_TOKEN else {}

    def collect(self, name: str, falco_events: List[dict]) -> dict | None:
        try:
            c = self.client.containers.get(name)
        except Exception:
            return None
        if c.status != "running":
            return None
        res = self.monitor.sample(c)
        if res is None:
            return None
        conns = parse_proc_net_tcp(_exec(c, ["cat", "/proc/net/tcp"]) + "\n" + _exec(c, ["cat", "/proc/net/tcp6"]))
        flow = self.flows.setdefault(name, FlowFeatureExtractor()).update(conns, res["rx_rate"] * self.interval,
                                                                           res["tx_rate"] * self.interval)
        sess = active_session(parse_audit_log(_exec(c, ["tail", "-n", "500", AUDIT_LOG_PATH])))
        identity = None
        if sess:
            identity = {k: sess[k] for k in ("session_id", "user", "source_ip", "start_hour", "lineage", "intervals", "n_commands")}
        mine = [e for e in falco_events if e["container_name"] == name or (e["container_id"] and c.id.startswith(e["container_id"]))]
        return {"container_id": c.id, "container_name": name, "ts": time.time(), "flow": flow,
                "resources": {k: res[k] for k in ("cpu_util", "ram_util", "net_io_rate")},
                "identity": identity, "falco": mine}

    def post(self, event: dict) -> None:
        try:
            requests.post(f"{self.url}/api/ingest_telemetry", json=event, headers=self.headers, timeout=5)
        except requests.RequestException as exc:
            logger.warning(f"engine unreachable ({exc}); event dropped")

    def run(self, once: bool = False):
        logger.info(f"Ingesting telemetry for {self.names} -> {self.url} every {self.interval}s (Falco: {self.falco.path})")
        while True:
            started = time.time()
            falco_events = self.falco.drain()
            for name in self.names:
                ev = self.collect(name, falco_events)
                if ev:
                    self.post(ev)
            if once:
                return
            time.sleep(max(self.interval - (time.time() - started), 0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", action="append", help="container name to watch (repeatable)")
    ap.add_argument("--engine-url", default=ENGINE_URL)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--falco-log", default=FALCO_LOG)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    import docker
    TelemetryIngestor(docker.from_env(), a.container or [VICTIM_NAME], a.engine_url, a.interval, a.falco_log).run(a.once)


if __name__ == "__main__":
    main()
