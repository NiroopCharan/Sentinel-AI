"""Time-aware process/resource provenance with lossless edge histories."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
import networkx as nx

from .events import SyscallEvent
from .process_tracker import ProcessTracker

_READ_OPS = {"read", "open_read", "recv", "recvfrom", "pread", "readv"}
_WRITE_OPS = {"write", "open_write", "send", "sendto", "pwrite", "writev"}


def _add_observation(graph: nx.DiGraph, source: str, target: str, **attrs) -> None:
    """Append an event to an edge without overwriting its earlier history.

    The ordinary edge attributes mirror the latest observation for compatibility
    with existing consumers. ``observations`` is the authoritative event history.
    """
    observation = dict(attrs)
    if graph.has_edge(source, target):
        edge = graph[source][target]
        history = list(edge.get("observations", []))
        if not history:  # Upgrade an edge created by an older implementation.
            history.append({k: v for k, v in edge.items() if k != "observations"})
        history.append(observation)
        edge["observations"] = history
        edge.update(observation)
    else:
        graph.add_edge(source, target, **observation, observations=[observation])


@dataclass
class ProvenanceGraph:
    flow: nx.DiGraph = field(default_factory=nx.DiGraph)
    access: nx.DiGraph = field(default_factory=nx.DiGraph)
    process_tree: nx.DiGraph = field(default_factory=nx.DiGraph)
    tracker: ProcessTracker = field(default_factory=ProcessTracker)
    last_timestamp: float = -1.0

    def ingest(self, event: SyscallEvent) -> None:
        if event.timestamp < self.last_timestamp:
            raise ValueError("Events must be ingested in non-decreasing timestamp order")
        self.last_timestamp = event.timestamp
        observation = self.tracker.observe(event)
        proc = f"proc:{event.process_instance}"
        process_attrs = dict(kind="process", pid=event.pid, host_id=event.host_id,
                             boot_id=event.boot_id, observed_at=event.timestamp)
        # Preserve first-seen time; NetworkX updates attributes on an existing node.
        for graph in (self.flow, self.access, self.process_tree):
            if not graph.has_node(proc):
                graph.add_node(proc, **process_attrs)
        if observation.parent_instance:
            parent = f"proc:{observation.parent_instance}"
            if not self.process_tree.has_node(parent):
                self.process_tree.add_node(parent, kind="process",
                                           observed_at=event.timestamp)
            _add_observation(self.process_tree, parent, proc,
                             timestamp=event.timestamp, relation="parent_child")

        if not event.resource:
            return
        resource = f"resource:{event.resource}"
        if not self.flow.has_node(resource):
            self.flow.add_node(resource, kind="resource", observed_at=event.timestamp)
        if not self.access.has_node(resource):
            self.access.add_node(resource, kind="resource", observed_at=event.timestamp)
        operation = (event.operation or event.syscall).lower()
        if operation in _READ_OPS:
            _add_observation(self.flow, resource, proc,
                             timestamp=event.timestamp, operation=operation)
            _add_observation(self.access, proc, resource,
                             timestamp=event.timestamp, operation=operation, mode="read")
        elif operation in _WRITE_OPS:
            _add_observation(self.flow, proc, resource,
                             timestamp=event.timestamp, operation=operation)
            _add_observation(self.access, proc, resource,
                             timestamp=event.timestamp, operation=operation, mode="write")
        else:
            _add_observation(self.access, proc, resource,
                             timestamp=event.timestamp, operation=operation, mode="other")

    @staticmethod
    def _snapshot(graph: nx.DiGraph, at_time: float) -> nx.DiGraph:
        result = nx.DiGraph()
        visible = {n for n, d in graph.nodes(data=True)
                   if d.get("observed_at", 0.0) <= at_time}
        result.add_nodes_from((n, dict(graph.nodes[n])) for n in visible)

        for source, target, attrs in graph.edges(data=True):
            if source not in visible or target not in visible:
                continue
            history = attrs.get("observations")
            if history is None:
                # Backward compatibility for graphs built before history tracking.
                if attrs.get("timestamp", float("inf")) <= at_time:
                    result.add_edge(source, target, **dict(attrs))
                continue

            eligible = [dict(item) for item in history
                        if item.get("timestamp", float("inf")) <= at_time]
            if not eligible:
                continue
            # Preserve the familiar top-level attributes, but make them reflect
            # the latest event visible at this snapshot rather than future state.
            latest = dict(eligible[-1])
            latest["observations"] = eligible
            result.add_edge(source, target, **latest)
        return result

    def snapshot(self, at_time: float) -> tuple[nx.DiGraph, nx.DiGraph, nx.DiGraph]:
        if isinstance(at_time, bool) or not isinstance(at_time, (int, float)):
            raise TypeError("snapshot time must be a finite number")
        if not isfinite(float(at_time)) or at_time < 0:
            raise ValueError("snapshot time must be finite and non-negative")
        return (self._snapshot(self.flow, float(at_time)),
                self._snapshot(self.access, float(at_time)),
                self._snapshot(self.process_tree, float(at_time)))
