"""Deterministic toy demonstration. Outputs are synthetic, not benchmark evidence."""
import json
from .events import SyscallEvent
from .provenance import ProvenanceState
from .interdiction import Action, AttackPath, solve_exact
from .sequential import ProductEProcess

def main():
    state = ProvenanceState.create()
    events = [
        SyscallEvent(1, "hostA", "bootA", 100, 0.5, "execA", 1, "open", "/tmp/input", "read"),
        SyscallEvent(2, "hostA", "bootA", 100, 0.5, "execA", 1, "connect", "socket:db", "send"),
    ]
    for e in events: state.ingest(e)
    flow, access = state.snapshot(1.5)
    result = solve_exact(
        [Action("freeze_worker", 1.0), Action("block_socket", 0.4), Action("kill_container", 5.0)],
        [AttackPath("p1", frozenset({"freeze_worker", "block_socket", "kill_container"}), 0.08)],
        delta=0.01,
    )
    detector = ProductEProcess(kappa=0.5, alpha=0.05)
    decisions = [detector.update(p) for p in [0.8, 0.9, 0.7]]
    print(json.dumps({"warning": "SYNTHETIC TOY OUTPUT; NOT REAL DATA PERFORMANCE", "flow_edges_at_time_1_5": list(flow.edges()), "access_edges_at_time_1_5": list(access.edges()), "optimizer": result.__dict__, "sequential_log_e": decisions[-1][0], "alert": decisions[-1][1]}, indent=2, default=list))

if __name__ == "__main__": main()
