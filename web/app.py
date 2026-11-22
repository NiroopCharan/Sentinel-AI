"""Flask + Socket.IO backend (build prompt section 11).

    python -m web.app                 # tries a real Docker client; falls back to dry-run (scoring only) if none
    SENTINEL_DRY_RUN=1 python -m web.app

REST:
  POST /api/ingest_telemetry   <- sensors.telemetry_ingestor  (one event per container per poll)
  GET  /api/state              full current state (history, incidents, weights) - what the dashboard loads on open
  GET  /api/incidents/<id>/snapshot/<before|after>   decrypted forensic snapshot SUMMARY (not the raw tarball)
  POST /api/isolation/<name>/release
  POST /api/demo/inject        synthetic telemetry event, for rehearsing the dashboard without live attack tools
Socket.IO events pushed to clients: 'telemetry' (every scored event), 'incident' (on open/escalate/close/heal).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from flask import Flask, abort, jsonify, request
from flask_socketio import SocketIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ENGINE_TOKEN, MODELS_DIR  # noqa: E402
from engine.hmr_engine import HMREngine  # noqa: E402
from forensics.forensic_snapshot import load_summaries  # noqa: E402
from utils.log import logger  # noqa: E402

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "sentinel-ai-dev"  # dashboard has no auth beyond the optional ingest token; demo scope only
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")


def _make_docker_client():
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return client
    except Exception as exc:
        logger.warning(f"No usable Docker daemon ({exc}) - starting in dry-run (scoring only, no heal actions).")
        return None


def _emit(event, payload):
    socketio.emit(event, payload)


docker_client = _make_docker_client()
engine = HMREngine(models_dir=MODELS_DIR, docker_client=docker_client, emit=_emit, dry_run=docker_client is None)


def _check_token():
    if ENGINE_TOKEN and request.headers.get("X-Sentinel-Token") != ENGINE_TOKEN:
        abort(401)


@app.post("/api/ingest_telemetry")
def ingest_telemetry():
    _check_token()
    event = request.get_json(force=True, silent=False)
    for field in ("container_id", "container_name"):
        if field not in event:
            return jsonify({"error": f"missing field: {field}"}), 400
    result = engine.process_event(event)
    return jsonify(result)


@app.get("/api/state")
def state():
    return jsonify(engine.state())


@app.get("/api/incidents/<incident_id>/snapshot/<label>")
def incident_snapshot(incident_id, label):
    if label not in ("before", "after"):
        abort(404)
    summaries = load_summaries(incident_id, engine.forensics_dir)
    if label not in summaries:
        abort(404)
    return jsonify(summaries[label])


@app.post("/api/isolation/<name>/release")
def release(name):
    return jsonify({"released": engine.release_isolation(name)})


@app.post("/api/demo/inject")
def demo_inject():
    """Synthetic telemetry for rehearsing the dashboard UI without live attack tooling. Every event this
    produces is tagged demo=true in the payload so it is never confused with a real detection."""
    body = request.get_json(force=True, silent=True) or {}
    kind = body.get("kind", "attack")
    name = body.get("container", "sentinel-victim-pod")
    flows = {
        "benign": {"duration": 40.0, "protocol_type": "tcp", "src_bytes": 900.0, "dst_bytes": 1500.0,
                   "count": 4.0, "srv_count": 4.0, "same_srv_rate": 0.8, "diff_srv_rate": 0.2},
        "attack": {"duration": 0.0, "protocol_type": "tcp", "src_bytes": 20.0, "dst_bytes": 0.0, "count": 400.0,
                  "srv_count": 400.0, "same_srv_rate": 1.0, "diff_srv_rate": 0.0},
    }
    if kind == "attack":
        falco = [
            {"rule": "Read sensitive file untrusted", "priority": "Warning", "syscall": "openat"},
            {"rule": "Directory traversal probe", "priority": "Error", "syscall": "getdents"},
        ]
        spectra_features = {
            "completed_syscalls": 480.0, "failed_syscalls": 14.0, "network_recvfrom": 95.0,
            "syscall_openat": 50.0, "syscall_getdents": 22.0, "syscall_execve": 4.0
        }
    else:
        falco = []
        spectra_features = {
            "completed_syscalls": 6.0, "failed_syscalls": 0.0, "network_recvfrom": 2.0,
            "syscall_openat": 1.0, "syscall_close": 1.0
        }
    event = {"container_id": name, "container_name": name, "ts": time.time(), "flow": flows.get(kind, flows["attack"]),
             "resources": {"cpu_util": 85.0 if kind == "attack" else 2.0, "ram_util": 10.0, "net_io_rate": 5000.0},
             "identity": None, "falco": falco, "spectra_features": spectra_features, "demo": True}
    result = engine.process_event(event)
    result["demo"] = True
    return jsonify(result)


@app.get("/")
def dashboard():
    from flask import render_template
    return render_template("dashboard.html")


if __name__ == "__main__":
    logger.info("Sentinel-AI dashboard on http://0.0.0.0:5000  (dry_run=%s)" % (docker_client is None))
    socketio.run(app, host="0.0.0.0", port=5000)
