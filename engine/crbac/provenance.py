"""Online, time-respecting provenance graph with distinct access and flow edges."""
from dataclasses import dataclass
import networkx as nx
from .events import SyscallEvent

@dataclass
class ProvenanceState:
    flow: nx.DiGraph
    access: nx.DiGraph
    last_timestamp: float = -1.0

    @classmethod
    def create(cls):
        return cls(nx.DiGraph(), nx.DiGraph())

    def ingest(self, event: SyscallEvent) -> None:
        if event.timestamp < self.last_timestamp:
            raise ValueError("Events must be ingested in non-decreasing timestamp order")
        self.last_timestamp = event.timestamp
        proc = f"proc:{event.process_instance}"
        self.flow.add_node(proc, kind="process")
        self.access.add_node(proc, kind="process")
        if not event.resource:
            return
        resource = f"resource:{event.resource}"
        self.flow.add_node(resource, kind="resource")
        self.access.add_node(resource, kind="resource")
        op = (event.operation or event.syscall).lower()
        # Information flow: resource -> process on read; process -> resource on write.
        if op in {"read", "open_read", "recv", "recvfrom", "pread", "readv"}:
            self.flow.add_edge(resource, proc, timestamp=event.timestamp, operation=op)
            self.access.add_edge(proc, resource, timestamp=event.timestamp, operation=op, mode="read")
        elif op in {"write", "open_write", "send", "sendto", "pwrite", "writev"}:
            self.flow.add_edge(proc, resource, timestamp=event.timestamp, operation=op)
            self.access.add_edge(proc, resource, timestamp=event.timestamp, operation=op, mode="write")
        else:
            self.access.add_edge(proc, resource, timestamp=event.timestamp, operation=op, mode="other")

    def snapshot(self, at_time: float) -> tuple[nx.DiGraph, nx.DiGraph]:
        """Return copies containing only observations at or before at_time."""
        def filtered(g):
            out = nx.DiGraph()
            for n, d in g.nodes(data=True): out.add_node(n, **d)
            for u, v, d in g.edges(data=True):
                if d.get("timestamp", float("inf")) <= at_time: out.add_edge(u, v, **d)
            return out
        return filtered(self.flow), filtered(self.access)
