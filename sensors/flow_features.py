"""Approximate NSL-KDD-style flow features for a live container (honest about the gap).

NSL-KDD features were produced by 1999-era tooling over whole connections. Here we approximate the seven
core features from what is observable on a running container: /proc/net/tcp(6) snapshots (read through
`docker exec`) plus rx/tx byte deltas from docker stats. This is an APPROXIMATION and a documented
domain-gap limitation: live feature distributions are not guaranteed to match NSL-KDD's.

  count          inbound connections first seen within the last WINDOW seconds
  srv_count      of those, connections to the same local port as the newest one
  same_srv_rate  srv_count / count         diff_srv_rate  1 - same_srv_rate
  duration       age (s) of the oldest still-open inbound connection (0 if none)
  src_bytes      bytes received by the container in the poll interval (client -> server)
  dst_bytes      bytes sent by the container in the poll interval (server -> client)
"""
from __future__ import annotations

import socket
import struct
import time
from typing import Dict, List, Optional

STATES = {"01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV", "04": "FIN_WAIT1", "05": "FIN_WAIT2",
          "06": "TIME_WAIT", "07": "CLOSE", "08": "CLOSE_WAIT", "09": "LAST_ACK", "0A": "LISTEN", "0B": "CLOSING"}


def _hex_ip(h: str) -> str:
    if len(h) == 8:  # IPv4, little-endian
        return socket.inet_ntoa(struct.pack("<I", int(h, 16)))
    raw = b"".join(struct.pack("<I", int(h[i:i + 8], 16)) for i in range(0, 32, 4))  # IPv6, 4 LE words
    return socket.inet_ntop(socket.AF_INET6, raw)


def parse_proc_net_tcp(text: str) -> List[dict]:
    conns = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or ":" not in parts[1] or parts[0] == "sl":
            continue
        try:
            lip, lport = parts[1].split(":")
            rip, rport = parts[2].split(":")
            conns.append({"local_ip": _hex_ip(lip), "local_port": int(lport, 16), "remote_ip": _hex_ip(rip),
                          "remote_port": int(rport, 16), "state": STATES.get(parts[3], parts[3])})
        except (ValueError, OSError):
            continue
    return conns


class FlowFeatureExtractor:
    def __init__(self, window: float = 2.0):
        self.window = window
        self._first_seen: Dict[tuple, float] = {}

    def update(self, conns: List[dict], rx_delta: float, tx_delta: float, now: Optional[float] = None) -> Optional[Dict]:
        """Returns the feature dict, or None when there is no inbound connection activity or traffic at all
        (an idle container has no flow to classify - RF/AS are then simply absent from the fusion)."""
        now = time.time() if now is None else now
        listening = {c["local_port"] for c in conns if c["state"] == "LISTEN"}
        inbound = [c for c in conns if c["state"] not in ("LISTEN", "TIME_WAIT") and c["local_port"] in listening]
        live_keys = set()
        for c in inbound:
            k = (c["local_port"], c["remote_ip"], c["remote_port"])
            live_keys.add(k)
            self._first_seen.setdefault(k, now)
        for k in [k for k in self._first_seen if k not in live_keys]:
            del self._first_seen[k]          # forget closed connections

        recent = [(k, t) for k, t in self._first_seen.items() if now - t <= self.window]
        if not inbound and rx_delta <= 0 and tx_delta <= 0:
            return None
        count = len(recent)
        if count:
            newest_port = max(recent, key=lambda kt: kt[1])[0][0]
            srv_count = sum(1 for k, _ in recent if k[0] == newest_port)
            same = srv_count / count
        else:
            srv_count, same = 0, 0.0
        established = [now - t for k, t in self._first_seen.items()]
        return {"duration": max(established) if established else 0.0, "protocol_type": "tcp",
                "src_bytes": float(rx_delta), "dst_bytes": float(tx_delta), "count": float(count),
                "srv_count": float(srv_count), "same_srv_rate": float(same),
                "diff_srv_rate": float(1 - same if count else 0.0)}
