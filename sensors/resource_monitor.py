"""Live resource telemetry: docker stats API (per container) with a psutil fallback for the host.

cpu_util / ram_util are percentages; net_io_rate is bytes/second (rx+tx) computed from consecutive samples.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import psutil


def compute_cpu_pct(stats: dict) -> float:
    """Same formula as `docker stats` (cpu delta / system delta * online cpus * 100)."""
    try:
        cpu, pre = stats["cpu_stats"], stats["precpu_stats"]
        cpu_delta = cpu["cpu_usage"]["total_usage"] - pre["cpu_usage"]["total_usage"]
        sys_delta = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
        ncpu = cpu.get("online_cpus") or len(cpu["cpu_usage"].get("percpu_usage") or [1])
        if sys_delta > 0 and cpu_delta >= 0:
            return cpu_delta / sys_delta * ncpu * 100.0
    except (KeyError, TypeError):
        pass
    return 0.0


def compute_ram_pct(stats: dict) -> float:
    try:
        mem = stats["memory_stats"]
        used = mem["usage"] - (mem.get("stats", {}).get("inactive_file") or mem.get("stats", {}).get("cache") or 0)
        limit = mem["limit"]
        return max(used, 0) / limit * 100.0 if limit else 0.0
    except (KeyError, TypeError):
        return 0.0


def net_bytes(stats: dict) -> tuple:
    """(rx_bytes, tx_bytes) summed over the container's interfaces."""
    nets = stats.get("networks") or {}
    return (sum(n.get("rx_bytes", 0) for n in nets.values()), sum(n.get("tx_bytes", 0) for n in nets.values()))


class ResourceMonitor:
    def __init__(self, docker_client):
        self.client = docker_client
        self._last: Dict[str, tuple] = {}   # container id -> (t, rx, tx)

    def sample(self, container) -> Optional[Dict[str, float]]:
        try:
            stats = self.client.api.stats(container.id, stream=False)
        except Exception:
            return None
        now = time.time()
        rx, tx = net_bytes(stats)
        prev = self._last.get(container.id)
        self._last[container.id] = (now, rx, tx)
        rx_rate = tx_rate = 0.0
        if prev and now > prev[0]:
            rx_rate, tx_rate = max(rx - prev[1], 0) / (now - prev[0]), max(tx - prev[2], 0) / (now - prev[0])
        return {"cpu_util": compute_cpu_pct(stats), "ram_util": compute_ram_pct(stats),
                "net_io_rate": rx_rate + tx_rate, "rx_rate": rx_rate, "tx_rate": tx_rate, "ts": now}


def sample_host() -> Dict[str, float]:
    """Host-level fallback via psutil (used when no Docker daemon is reachable)."""
    io = psutil.net_io_counters()
    time.sleep(0.5)
    io2 = psutil.net_io_counters()
    return {"cpu_util": psutil.cpu_percent(interval=None), "ram_util": psutil.virtual_memory().percent,
            "net_io_rate": ((io2.bytes_recv + io2.bytes_sent) - (io.bytes_recv + io.bytes_sent)) / 0.5,
            "ts": time.time()}
