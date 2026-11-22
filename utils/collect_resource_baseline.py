"""Drives utils.live_attack_demo across all resource scenarios N times each (build prompt section 8.2),
building data/resource_baseline.csv for engine.resource_model's W_rds fit.

    python -m utils.collect_resource_baseline --runs 10 --duration 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.live_attack_demo import append_baseline, collect, spawn_victim  # noqa: E402
from utils.log import logger  # noqa: E402
from config import VICTIM_IMAGE  # noqa: E402

RESOURCE_SCENARIOS = {"benign": 0, "cryptominer": 1, "flood": 1}  # flood here = `stress-ng --cpu --io` proxy, not a real hping3 flood


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="per scenario (build prompt suggests 30-50 TOTAL)")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--image", default=VICTIM_IMAGE)
    a = ap.parse_args()
    import docker
    import uuid
    client = docker.from_env()
    for scenario, label in RESOURCE_SCENARIOS.items():
        for i in range(a.runs):
            run_id = uuid.uuid4().hex[:10]
            logger.info(f"[{scenario} {i + 1}/{a.runs}] run_id={run_id}")
            c = spawn_victim(client, a.image, host_port=0)
            try:
                import time
                time.sleep(3)
                if scenario == "cryptominer":
                    c.exec_run(["sh", "-c", f"stress-ng --cpu 0 --timeout {int(a.duration)}s &"], detach=True)
                elif scenario == "flood":
                    c.exec_run(["sh", "-c", f"stress-ng --cpu 0 --io 2 --timeout {int(a.duration)}s &"], detach=True)
                rows = collect(client, c, run_id, scenario, label, a.duration)
                append_baseline(rows)
            finally:
                c.remove(force=True)
    logger.info("Done. Run `python -m engine.train_models` (or `python -m engine.resource_model --train`) to fit W_rds.")


if __name__ == "__main__":
    main()
