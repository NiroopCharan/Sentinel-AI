"""Streaming reader and window feature extraction for LID-DS 2021 Sysdig recordings.

The outer dataset ZIP is never extracted. Each nested recording ZIP is opened
one at a time; the large .sc trace is read line-by-line.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import csv
import io
import json
import math
from pathlib import PurePosixPath
import re
import zipfile
from typing import Iterator

_LIFECYCLE = {"clone", "clone3", "fork", "vfork", "execve", "execveat",
              "exit", "exit_group"}
_NETWORK = {"accept", "accept4", "connect", "bind", "listen", "sendto",
            "recvfrom", "sendmsg", "recvmsg", "sendmmsg", "recvmmsg",
            "shutdown", "getsockname", "getpeername"}
# Sysdig's text trace layout: timestamp cpu pid comm tid syscall direction args...
# Split with maxsplit so the argument/return-value tail remains intact.
_TRACE_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+([<>])(?:\s+(.*))?$")


@dataclass(frozen=True)
class TraceEvent:
    timestamp: float
    timestamp_ns: int
    cpu: int
    pid: int
    comm: str
    tid: int
    syscall: str
    direction: str
    args: str = ""


@dataclass
class Window:
    start: float
    end: float
    features: dict[str, float]
    label: int | None
    label_source: str


def parse_sysdig_line(line: str) -> TraceEvent | None:
    """Parse one LID-DS Sysdig line; return None for headers/malformed lines."""
    m = _TRACE_RE.match(line)
    if not m:
        return None
    ns, cpu, pid, comm, tid, syscall, direction, args = m.groups()
    ns_i = int(ns)
    return TraceEvent(ns_i / 1e9, ns_i, int(cpu), int(pid), comm, int(tid),
                      syscall, direction, args or "")


def _is_recording_zip_member(name: str) -> bool:
    """Accept real dataset recording ZIPs, excluding macOS metadata sidecars."""
    path = PurePosixPath(name)
    parts = path.parts
    # The distributed archive contains __MACOSX/._* AppleDouble sidecars
    # alongside each real recording. They are not ZIP files.
    if "__MACOSX" in parts or path.name.startswith("._"):
        return False
    if path.suffix.lower() != ".zip":
        return False
    # Only process files under the actual scenario root, not metadata trees.
    try:
        root_index = parts.index("CVE-2018-3760")
    except ValueError:
        return False
    relative = parts[root_index + 1:]
    return len(relative) >= 2 and relative[0] in {
        "training", "validation", "test"
    }


def _recording_parts(path: str) -> tuple[str, str, str]:
    p = PurePosixPath(path)
    parts = p.parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected LID-DS recording path: {path}")
    split = parts[-2]
    if split not in {"training", "validation", "normal", "normal_and_attack"}:
        raise ValueError(f"Unknown LID-DS split folder in {path}")
    # test has two subfolders; training/validation are direct.
    if split in {"normal", "normal_and_attack"}:
        dataset_split = "test"
        category = split
    else:
        dataset_split = split
        category = split
    return dataset_split, category, p.stem


def iter_recording_paths(archive: str) -> Iterator[tuple[str, str, str, str]]:
    """Yield (outer_zip_member, split, category, recording_id), no extraction."""
    with zipfile.ZipFile(archive) as outer:
        for info in outer.infolist():
            name = info.filename
            if not _is_recording_zip_member(name):
                continue
            try:
                split, category, rec_id = _recording_parts(name)
            except ValueError:
                continue
            yield name, split, category, rec_id


def _find_member(names: list[str], suffix: str) -> str:
    matches = [n for n in names if n.lower().endswith(suffix)]
    if not matches:
        raise ValueError(f"Nested recording is missing {suffix}")
    return matches[0]


def _metadata_time(meta: dict, key: str) -> float | None:
    try:
        value = meta["time"][key]
        if isinstance(value, list):
            value = value[0]
        if isinstance(value, dict):
            value = value.get("absolute")
        result = float(value)
        return result if math.isfinite(result) else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _iter_trace(stream):
    for raw in stream:
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", errors="replace")
        else:
            line = raw
        event = parse_sysdig_line(line)
        if event is not None:
            yield event


def _extract_windows(sc_stream, *, warmup_end: float | None,
                     attack_time: float | None, exploit: bool,
                     window_seconds: float) -> list[Window]:
    if window_seconds <= 0 or not math.isfinite(window_seconds):
        raise ValueError("window_seconds must be finite and > 0")

    # Counts are based on completed syscall exits ('<') to avoid double-counting
    # each operation's entry ('>') and exit. Unmatched exits remain observable.
    buckets = defaultdict(Counter)
    pending = defaultdict(deque)
    first_ts = last_ts = None
    for event in _iter_trace(sc_stream):
        if first_ts is None:
            first_ts = event.timestamp
        last_ts = event.timestamp
        if warmup_end is not None and event.timestamp < warmup_end:
            continue
        index = int((event.timestamp - (warmup_end if warmup_end is not None else first_ts))
                    // window_seconds)
        c = buckets[index]
        c["trace_lines"] += 1
        c[f"syscall_{event.syscall}"] += 1 if event.direction == "<" else 0
        if event.direction == ">":
            pending[(event.pid, event.tid, event.syscall)].append(event.timestamp_ns)
            continue

        c["completed_syscalls"] += 1
        c["unique_pid_marker_" + str(event.pid)] = 1
        c["unique_tid_marker_" + str(event.tid)] = 1
        c["unique_process_marker_" + event.comm] = 1
        if event.syscall in _LIFECYCLE:
            c[f"lifecycle_{event.syscall}"] += 1
        if event.syscall in _NETWORK:
            c[f"network_{event.syscall}"] += 1
        if re.search(r"\bres\s*=\s*-\d+", event.args):
            c["failed_syscalls"] += 1
        key = (event.pid, event.tid, event.syscall)
        if pending[key]:
            start_ns = pending[key].popleft()
            c["syscall_latency_ms_sum"] += max(0, event.timestamp_ns-start_ns)/1e6
            c["paired_syscalls"] += 1
        else:
            c["unmatched_exit_syscalls"] += 1

    if first_ts is None:
        return []
    origin = warmup_end if warmup_end is not None else first_ts
    out = []
    for idx in sorted(buckets):
        c = buckets[idx]
        start = origin + idx * window_seconds
        end = start + window_seconds
        # Exclude the window containing the exploit timestamp: its contents
        # straddle the label boundary and cannot be assigned cleanly.
        if exploit and attack_time is not None and start < attack_time < end:
            continue
        label = None
        source = "unlabelled"
        if exploit and attack_time is not None:
            label = int(start >= attack_time)
            source = "metadata_exploit_timestamp"
        elif not exploit:
            label = 0
            source = "metadata_exploit_false"
        features = {}
        for key, value in c.items():
            if key.startswith(("unique_pid_marker_", "unique_tid_marker_",
                               "unique_process_marker_")):
                continue
            features[key] = float(value)
        features["unique_pids"] = float(sum(k.startswith("unique_pid_marker_") for k in c))
        features["unique_tids"] = float(sum(k.startswith("unique_tid_marker_") for k in c))
        features["unique_process_names"] = float(sum(k.startswith("unique_process_marker_") for k in c))
        features["window_seconds"] = float(window_seconds)
        features["syscalls_per_second"] = features.get("completed_syscalls", 0.0)/window_seconds
        paired = features.get("paired_syscalls", 0.0)
        features["mean_syscall_latency_ms"] = (
            features.get("syscall_latency_ms_sum", 0.0)/paired if paired else 0.0
        )
        out.append(Window(start, end, features, label, source))
    return out


def read_recording_from_bytes(inner_zip_bytes: bytes, *, split: str,
                              category: str, recording_id: str,
                              window_seconds: float = 1.0) -> tuple[dict, list[Window]]:
    """Read one nested recording ZIP from bytes; does not extract files to disk."""
    with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as nested:
        names = nested.namelist()
        meta_name = _find_member(names, ".json")
        sc_name = _find_member(names, ".sc")
        meta = json.loads(nested.read(meta_name).decode("utf-8"))
        exploit = bool(meta.get("exploit", False))
        warmup = _metadata_time(meta, "warmup_end")
        attack = _metadata_time(meta, "exploit")
        with nested.open(sc_name, "r") as sc:
            windows = _extract_windows(sc, warmup_end=warmup, attack_time=attack,
                                       exploit=exploit, window_seconds=window_seconds)
        info = {
            "recording_id": recording_id, "split": split, "category": category,
            "exploit": exploit, "exploit_time": attack, "warmup_end": warmup,
            "recording_time": meta.get("recording_time"),
            "scenario": meta.get("exploit_name"),
        }
        return info, windows


def iter_lidds_windows(archive: str, *, window_seconds: float = 1.0,
                       splits: set[str] | None = None):
    """Yield (recording metadata, Window) from selected official dataset splits."""
    with zipfile.ZipFile(archive) as outer:
        members = []
        for name in outer.namelist():
            if not _is_recording_zip_member(name):
                continue
            try:
                split, category, rec_id = _recording_parts(name)
            except ValueError:
                continue
            if splits is None or split in splits:
                members.append((name, split, category, rec_id))
        for name, split, category, rec_id in members:
            inner = outer.read(name)  # only one nested recording in memory at a time
            info, windows = read_recording_from_bytes(
                inner, split=split, category=category, recording_id=rec_id,
                window_seconds=window_seconds
            )
            for window in windows:
                yield info, window



def export_windows_csv(archive: str, output_csv: str, *,
                       window_seconds: float = 1.0,
                       splits: set[str] | None = None) -> int:
    """Export stable CSV rows; sparse syscall feature maps are JSON-encoded."""
    import json as _json
    rows = iter_lidds_windows(archive, window_seconds=window_seconds, splits=splits)
    count = 0
    fieldnames = [
        "recording_id", "split", "category", "exploit", "window_start",
        "window_end", "label", "label_source", "exploit_time", "features_json"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for info, window in rows:
            writer.writerow({
                "recording_id": info["recording_id"], "split": info["split"],
                "category": info["category"], "exploit": info["exploit"],
                "window_start": window.start, "window_end": window.end,
                "label": window.label, "label_source": window.label_source,
                "exploit_time": info["exploit_time"],
                "features_json": _json.dumps(window.features, sort_keys=True),
            })
            count += 1
    return count
