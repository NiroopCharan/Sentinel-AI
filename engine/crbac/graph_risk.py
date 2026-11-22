"""Connect temporal provenance evidence to the CRBAC risk-budget engine."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import networkx as nx

from .events import SyscallEvent
from .provenance_graph import ProvenanceGraph
from .risk_budget import RiskBudgetEngine, RiskDecision


@dataclass
class GraphRiskIntegrator:
    graph: ProvenanceGraph = field(default_factory=ProvenanceGraph)
    engine: RiskBudgetEngine = field(default_factory=RiskBudgetEngine)
    persistence_window_seconds: float = 10.0
    persistence_saturation_events: int = 5
    event_history: dict[str, deque] = field(
        default_factory=lambda: defaultdict(deque)
    )

    def __post_init__(self):
        if self.persistence_window_seconds <= 0:
            raise ValueError("Persistence window must be positive")
        if self.persistence_saturation_events < 1:
            raise ValueError("Persistence saturation must be at least 1")

    def _causal_score(self, process_instance: str, timestamp: float) -> float:
        flow, access, tree = self.graph.snapshot(timestamp)
        process = f"proc:{process_instance}"

        if process not in access:
            return 0.0

        descendants = (
            len(nx.descendants(tree, process))
            if process in tree else 0
        )

        resources = [
            node for node in access.successors(process)
            if access.nodes[node].get("kind") == "resource"
        ]

        shared_processes = set()
        for resource in resources:
            for other in access.predecessors(resource):
                if other != process and access.nodes[other].get("kind") == "process":
                    shared_processes.add(other)

        # Transparent bounded graph-evidence score.
        score = (
            0.25 * min(descendants, 4)
            + 0.15 * min(len(resources), 4)
            + 0.10 * min(len(shared_processes), 5)
        )
        return min(1.0, score)

    def _persistence_score(self, process_instance: str, timestamp: float) -> float:
        history = self.event_history[process_instance]
        cutoff = timestamp - self.persistence_window_seconds

        while history and history[0] < cutoff:
            history.popleft()

        history.append(timestamp)

        return min(
            1.0,
            len(history) / self.persistence_saturation_events
        )

    def evaluate_event(
        self,
        event: SyscallEvent,
        anomaly_score: float,
    ) -> RiskDecision:
        if not isinstance(event, SyscallEvent):
            raise TypeError("event must be a SyscallEvent")

        self.graph.ingest(event)

        causal_score = self._causal_score(
            event.process_instance, event.timestamp
        )
        persistence_score = self._persistence_score(
            event.process_instance, event.timestamp
        )

        return self.engine.evaluate(
            timestamp=event.timestamp,
            process_instance=event.process_instance,
            anomaly_score=anomaly_score,
            causal_score=causal_score,
            persistence_score=persistence_score,
        )
