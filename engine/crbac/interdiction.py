"""Exact small-instance risk-constrained action selection by exhaustive search.

For larger instances replace with a verified MILP solver. Risk sums are union-bound
upper bounds only when each path probability is itself a valid upper bound and the
candidate path set is complete (or omitted-path mass is bounded).
"""
from dataclasses import dataclass
from itertools import combinations

@dataclass(frozen=True)
class AttackPath:
    path_id: str
    interceptable_actions: frozenset[str]
    risk_upper: float

@dataclass(frozen=True)
class Action:
    action_id: str
    cost: float

@dataclass
class InterdictionResult:
    selected_actions: tuple[str, ...]
    total_cost: float
    residual_risk_upper: float
    feasible: bool


def solve_exact(actions: list[Action], paths: list[AttackPath], delta: float) -> InterdictionResult:
    if not 0 <= delta <= 1: raise ValueError("delta must be in [0, 1]")
    if any(a.cost < 0 for a in actions): raise ValueError("action costs must be non-negative")
    if any(not 0 <= p.risk_upper <= 1 for p in paths): raise ValueError("path risks must be in [0, 1]")
    ids = [a.action_id for a in actions]
    if len(set(ids)) != len(ids): raise ValueError("action IDs must be unique")
    if any(not p.interceptable_actions.issubset(set(ids)) for p in paths):
        raise ValueError("path references an unknown action")
    best = None
    for r in range(len(actions) + 1):
        for subset in combinations(actions, r):
            chosen = {a.action_id for a in subset}
            residual = sum(p.risk_upper for p in paths if chosen.isdisjoint(p.interceptable_actions))
            cost = sum(a.cost for a in subset)
            if residual <= delta + 1e-12:
                key = (cost, len(chosen), tuple(sorted(chosen)))
                if best is None or key < best[0]: best = (key, chosen, residual)
    if best is None:
        return InterdictionResult((), float("inf"), sum(p.risk_upper for p in paths), False)
    return InterdictionResult(tuple(sorted(best[1])), best[0][0], best[2], True)
