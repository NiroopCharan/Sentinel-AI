"""HMR engine: Reason -> Decide -> Heal -> Verify, plus the feedback loop (build prompt sections 4-7).

    process_event(event)  ->  four vector scores -> fused Threat Score (adaptive weights) -> tier -> action

Stage 4/5 live in _execute_isolate / _execute_kill_restart / _verify_replica. Disruptive actions need
`act_streak` consecutive over-threshold events and are skipped entirely when `dry_run` is set or no Docker
client is available (the engine then only scores and reports what it WOULD do).
"""
from __future__ import annotations

import collections
import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ACT_STREAK, DATA_DIR, DEFAULT_MARGIN, DEFAULT_TAU, FORENSICS_DIR, MODELS_DIR,  # noqa: E402
                    QUARANTINE_NETWORK, TIERS, VECTORS,
                    SPECTRA_MODEL_PATH, DUAL_ALGORITHM_MODE, SPECTRA_WEIGHT, DISRUPTIVE_ACTION_POLICY)
from engine.adaptive_weights import AdaptiveWeightTracker, default_tracker  # noqa: E402
from engine.data_loader import feature_columns  # noqa: E402
from engine.fusion import (fuse, normalize_anomaly, normalize_anomaly_mad,  # noqa: E402
                           tier_for, tier_rank, update_sequential_evidence)
from engine.identity_model import IdentityModel  # noqa: E402
from engine.resource_model import ResourceDriftModel  # noqa: E402
from engine.spectra_engine import SpectraEngine  # noqa: E402
from forensics import forensic_snapshot  # noqa: E402
from sensors.resource_monitor import compute_cpu_pct, compute_ram_pct  # noqa: E402
from utils.log import logger  # noqa: E402

SEVERE_PRIORITIES = {"warning", "error", "critical", "alert", "emergency"}
# Process names that are never part of the clean victim image; seeing one in a forensic snapshot counts as evidence.
SUSPICIOUS_PROCS = {"xmrig", "minerd", "cpuminer", "stress-ng", "ncat", "nc", "socat", "hydra", "hping3", "nmap", "masscan"}
_CAT_DEFAULTS = {"protocol_type": "tcp", "service": "other", "flag": "SF"}


class HMREngine:
    def __init__(self, models_dir: Path = MODELS_DIR, docker_client=None, emit: Optional[Callable] = None,
                 tracker: Optional[AdaptiveWeightTracker] = None, dry_run: bool = False,
                 forensics_dir: Path = FORENSICS_DIR, act_streak: int = ACT_STREAK, verify_window: float = 5.0,
                 close_after: int = 20, release_after: int = 60, cooldown: float = 5.0,
                 evidence_tau: float = 0.70,
                 spectra_model_path: Optional[Path] = None,
                 algorithm_mode: Optional[str] = None,
                 spectra_weight: Optional[float] = None,
                 disruptive_policy: Optional[str] = None):
        self.models_dir, self.docker, self.dry_run = Path(models_dir), docker_client, dry_run
        self.forensics_dir, self.act_streak, self.verify_window = Path(forensics_dir), act_streak, verify_window
        self.close_after, self.release_after, self.cooldown = close_after, release_after, cooldown
        self.evidence_tau = evidence_tau
        self._emit = emit or (lambda *_a, **_k: None)
        self.tracker = tracker or default_tracker()
        self._lock = threading.RLock()
        self.history: collections.deque = collections.deque(maxlen=600)
        self.incidents: Dict[str, dict] = {}
        self._open: Dict[str, str] = {}                 # container name -> open incident id
        self._streak: Dict[str, int] = collections.defaultdict(int)
        self._calm: Dict[str, int] = collections.defaultdict(int)
        self._evidence_s: Dict[str, float] = collections.defaultdict(float)
        self._isolated: Dict[str, dict] = {}
        self._healing: set = set()
        self._cooldown_until: Dict[str, float] = {}
        self._counter = 0

        # Dual-Algorithm Orchestration (HMR + SPECTRA-CRBAC v2)
        self.algorithm_mode = (algorithm_mode or DUAL_ALGORITHM_MODE).upper()
        self.spectra_weight = float(SPECTRA_WEIGHT if spectra_weight is None else spectra_weight)
        self.disruptive_policy = (disruptive_policy or DISRUPTIVE_ACTION_POLICY).lower()
        s_path = spectra_model_path or SPECTRA_MODEL_PATH
        self.spectra = SpectraEngine(model_path=s_path)

        self._load_models()

    # ------------------------------------------------------------------ setup
    def _load_models(self):
        meta_path = self.models_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} missing - run `python -m engine.train_models` first.")
        self.meta = json.loads(meta_path.read_text())
        if self.meta["trained_on"] != "real":
            logger.warning("SYNTHETIC DEMO DATA — models in use were trained on synthetic data, not NSL-KDD. "
                           "Run scripts/download_nsl_kdd.sh and engine.train_models for real evaluation.")
        self.pre = joblib.load(self.models_dir / "preprocessor.joblib")
        self.rf = joblib.load(self.models_dir / "rf.joblib")
        self.iso = joblib.load(self.models_dir / "iso.joblib")
        self.cols = feature_columns(self.meta["feature_set"])
        self.tau = float(self.meta.get("tau", DEFAULT_TAU))
        self.margin = float(self.meta.get("margin", DEFAULT_MARGIN))
        self.resource_model = ResourceDriftModel.load(self.models_dir / "rds_weights.json")
        self.identity_model = IdentityModel.load(self.models_dir / "identity_model.json")
        if self.identity_model is None:
            logger.warning("ICS: models/identity_model.json missing - Identity Consistency Score disabled "
                           "(weights renormalise over the remaining vectors).")
        # Verification CPU ceiling: spec default 0.5%, widened to the collected benign baseline if one exists.
        self.verify_cpu_max = 0.5
        gb = self.resource_model.global_baseline
        if gb:
            self.verify_cpu_max = max(0.5, gb["cpu_util"]["mean"] + 3 * gb["cpu_util"]["std"])

    # ------------------------------------------------------------------ scoring
    def _score_network(self, flow: dict):
        row = {c: flow.get(c, _CAT_DEFAULTS.get(c, 0.0)) for c in self.cols}
        X = self.pre.transform(pd.DataFrame([row])[self.cols])
        rf = float(self.rf.predict_proba(X)[0, 1])
        a = float(normalize_anomaly(self.iso.score_samples(X), self.meta["as_min"], self.meta["as_max"])[0])
        return rf, a

    def process_event(self, event: dict) -> dict:
        name, now = event["container_name"], event.get("ts") or time.time()
        with self._lock:
            if now < self._cooldown_until.get(name, 0) or name in self._healing:
                return {"skipped": "healing/cooldown", "container": name}
            flow, res, ident, falco = event.get("flow"), event.get("resources"), event.get("identity"), event.get("falco") or []
            scores: Dict[str, Optional[float]] = {v: None for v in VECTORS}
            if flow:
                scores["rf"], scores["as"] = self._score_network(flow)
            if res:
                scores["rds"] = self.resource_model.score(name, res)
            if ident and self.identity_model:
                scores["ics"] = self.identity_model.score(ident)
            weights = self.tracker.current_weights()
            ts_hmr, eff = fuse(scores, weights)

            # Evaluate SPECTRA-CRBAC v2 (Syscall / Kernel Telemetry Engine)
            spectra_eval = self.spectra.evaluate_event(name, event)
            spectra_risk = spectra_eval["spectra_risk"] if spectra_eval else None
            spectra_ano = spectra_eval.get("is_anomalous", False) if spectra_eval else False

            # Dual-Algorithm Threat Score Fusion
            if self.algorithm_mode == "HMR_ONLY" or not self.spectra.active or spectra_risk is None:
                ts = ts_hmr
            elif self.algorithm_mode == "SPECTRA_ONLY":
                ts = spectra_risk
            else:  # HYBRID mode: weighted meta-fusion of HMR and SPECTRA
                ts = float(np.clip(
                    (1.0 - self.spectra_weight) * ts_hmr + self.spectra_weight * spectra_risk,
                    0.0, 1.0,
                ))

            tier = tier_for(ts, self.tau, self.margin)
            rank = tier_rank(tier)

            # Dual-Guard Sensitivity: if SPECTRA flags an anomaly, escalate at least to LOG_WATCH
            if spectra_ano and rank == 0:
                tier = TIERS[1]
                rank = 1

            if tier == TIERS[0] and res:
                self.resource_model.observe_healthy(name, res)   # baseline only learns from calm moments

            # --- Sequential Accumulator & Streak Gate ---
            prev_s = self._evidence_s[name]
            new_s, ev_score = update_sequential_evidence(prev_s, ts, prior_p0=0.023, lambda_decay=0.88, beta_scale=0.25)
            self._evidence_s[name] = new_s

            is_over_threshold = rank >= tier_rank(TIERS[2])

            # Disruptive policy check: consensus requirement in HYBRID mode
            if is_over_threshold and self.disruptive_policy == "consensus" and self.algorithm_mode == "HYBRID" and spectra_risk is not None:
                corroborated = (ts_hmr >= self.tau / 2) and (spectra_risk >= self.tau / 2)
                if not corroborated and self._streak[name] < self.act_streak:
                    tier = TIERS[1]
                    rank = 1
                    is_over_threshold = False

            traditional_streak = self._streak[name] + 1 if is_over_threshold else 0
            self._streak[name] = traditional_streak

            self._calm[name] = self._calm[name] + 1 if tier == TIERS[0] else 0

            mitre = forensic_snapshot.tag_event(falco, scores, flow, res) if rank >= 1 else []
            inc = self._track_incident(name, event, scores, ts, tier, weights, mitre, falco)
            action = self._decide(name, event["container_id"], tier, inc)

            payload = {
                "t": now, "container": name, "scores": scores, "score": ts, "hmr_score": ts_hmr,
                "spectra": spectra_eval, "algorithm_mode": self.algorithm_mode, "evidence": round(ev_score, 4),
                "tier": tier, "weights": weights, "effective_weights": eff, "tau": self.tau, "margin": self.margin,
                "action": action, "incident_id": inc["id"] if inc else None,
                "resources": res, "flow": flow, "falco_rules": [e.get("rule") for e in falco]
            }
            self.history.append(payload)
            self._emit("telemetry", payload)
            return payload

    # ------------------------------------------------------------------ incidents
    def _track_incident(self, name, event, scores, ts, tier, weights, mitre, falco) -> Optional[dict]:
        inc = self.incidents.get(self._open.get(name, ""))
        if inc is None:
            if tier_rank(tier) < 1:
                return None
            self._counter += 1
            iid = f"INC-{time.strftime('%Y%m%d-%H%M%S')}-{self._counter:03d}"
            inc = {"id": iid, "container": name, "opened_at": time.time(), "status": "open", "tier": tier, "peak_score": ts,
                   "trigger_scores": dict(scores), "mitre": [], "falco": [], "actions": [], "verified": None,
                   "evidence": [], "weights_at_open": weights, "snapshots": {}}
            self.incidents[iid], self._open[name] = inc, iid
            self._log_action(inc, "opened", f"tier {tier}, TS {ts:.2f}")
        if inc["status"] == "open":
            if ts >= inc["peak_score"]:
                inc["peak_score"], inc["trigger_scores"] = ts, dict(scores)
                inc["trigger_flow"], inc["trigger_resources"] = event.get("flow"), event.get("resources")
            if tier_rank(tier) > tier_rank(inc["tier"]):
                inc["tier"] = tier
                self._log_action(inc, "escalated", f"{tier}, TS {ts:.2f}")
            known = {m["technique"] for m in inc["mitre"]}
            inc["mitre"] += [m for m in mitre if m["technique"] not in known]
            inc["falco"] += [{"rule": e.get("rule"), "priority": e.get("priority")} for e in falco][: max(0, 50 - len(inc["falco"]))]
            if self._calm[name] >= self.close_after and name not in self._isolated:
                inc["status"], inc["closed_at"] = "closed", time.time()
                self._open.pop(name, None)
                self._log_action(inc, "closed", "returned to Monitor tier without disruptive action")
            self._emit("incident", self.public_incident(inc))
        return inc

    @staticmethod
    def _log_action(inc: dict, action: str, detail: str = ""):
        inc["actions"].append({"t": time.time(), "action": action, "detail": detail})

    @staticmethod
    def public_incident(inc: dict) -> dict:
        return {k: v for k, v in inc.items() if k not in ("trigger_flow", "trigger_resources")}

    # ------------------------------------------------------------------ decide
    def _decide(self, name: str, cid: str, tier: str, inc: Optional[dict]) -> str:
        if inc is None or self._streak[name] < self.act_streak:
            return {"MONITOR": "monitor", "LOG_WATCH": "log_watch"}.get(tier, "pending_confirmation")
        if tier == TIERS[3]:
            if self.dry_run or self.docker is None:
                self._log_action(inc, "dry_run", "would forensic-snapshot, freeze, kill, respawn and verify")
                return "dry_run_kill_restart"
            self._healing.add(name)
            inc["status"] = "healing"
            threading.Thread(target=self._heal_thread, args=(name, cid, inc["id"]), daemon=True).start()
            return "kill_restart_started"
        if tier == TIERS[2] and name not in self._isolated:
            if self.dry_run or self.docker is None:
                self._log_action(inc, "dry_run", "would isolate (quarantine network, drop egress)")
                return "dry_run_isolate"
            try:
                self._execute_isolate(name, cid)
                self._log_action(inc, "isolated", f"moved to {QUARANTINE_NETWORK} (internal, no egress)")
                return "isolated"
            except Exception as exc:
                logger.error(f"isolate failed for {name}: {exc}")
                self._log_action(inc, "isolate_failed", str(exc))
                return "isolate_failed"
        return "isolated" if name in self._isolated else "monitor"

    # ------------------------------------------------------------------ heal: isolate
    def _execute_isolate(self, name: str, cid: str):
        c = self.docker.containers.get(cid)
        c.reload()
        original = [n for n in c.attrs["NetworkSettings"]["Networks"] if n != QUARANTINE_NETWORK]
        try:
            q = self.docker.networks.get(QUARANTINE_NETWORK)
        except Exception as exc:   # docker.errors.NotFound -> create the internal (no-egress) network
            if exc.__class__.__name__ not in ("NotFound", "KeyError"):
                raise
            q = self.docker.networks.create(QUARANTINE_NETWORK, driver="bridge", internal=True)
        q.connect(c)                                   # join the no-egress network first, then leave the others
        for n in original:
            self.docker.networks.get(n).disconnect(c, force=True)
        self._isolated[name] = {"since": time.time(), "networks": original}
        logger.warning(f"ISOLATED {name}: detached from {original}, attached to {QUARANTINE_NETWORK}")

    def release_isolation(self, name: str):
        with self._lock:
            info = self._isolated.pop(name, None)
            if not info or self.docker is None:
                return False
            c = self.docker.containers.get(name)
            for n in info["networks"]:
                self.docker.networks.get(n).connect(c)
            try:
                self.docker.networks.get(QUARANTINE_NETWORK).disconnect(c, force=True)
            except Exception:
                pass
            self.spectra.reset_session(name)
            logger.info(f"released isolation on {name}")
            return True

    # ------------------------------------------------------------------ heal: kill + restart
    def _heal_thread(self, name: str, cid: str, incident_id: str):
        try:
            result = self._execute_kill_restart(cid, incident_id)
        except Exception as exc:  # never let a heal failure kill the engine
            logger.exception(f"kill/restart failed for {name}")
            result = {"healed": False, "verified": False, "error": str(exc)}
        with self._lock:
            inc = self.incidents[incident_id]
            inc["status"] = "healed" if result.get("healed") else "heal_failed"
            inc["verified"], inc["closed_at"] = result.get("verified"), time.time()
            self._log_action(inc, "heal_result", json.dumps({k: v for k, v in result.items() if k != "details"}, default=str))
            self._open.pop(name, None)
            self._isolated.pop(name, None)
            self._streak[name] = 0
            self._evidence_s.pop(name, None)
            self.spectra.reset_session(name)
            self._healing.discard(name)
            self._cooldown_until[name] = time.time() + self.cooldown
            self._emit("incident", self.public_incident(inc))

    def _execute_kill_restart(self, container_id: str, incident_id: str = "manual") -> dict:
        inc = self.incidents.get(incident_id) or {"mitre": [], "trigger_scores": {}, "falco": [], "actions": []}
        c = self.docker.containers.get(container_id)
        c.reload()
        attrs = c.attrs
        pid = (attrs.get("State") or {}).get("Pid")
        # 1. forensic snapshot BEFORE anything destructive (exec needs a running, unpaused container)
        before = forensic_snapshot.snapshot_container(c, incident_id, "before", host_pid=pid, mitre=inc["mitre"],
                                                      forensics_dir=self.forensics_dir)
        self._log_action(inc, "snapshot_before", before["artifact"])
        # 2. freeze -> kill -> remove
        for step in (c.pause, c.kill):
            try:
                step()
            except Exception as exc:   # e.g. already stopped; removal below is what matters
                logger.debug(f"{step.__name__} ignored: {exc}")
        c.remove(force=True)
        self._log_action(inc, "killed", f"container {container_id[:12]} frozen, killed, removed")
        # 3. respawn from the clean image
        new = self._respawn(attrs)
        self._log_action(inc, "respawned", f"new container {new.id[:12]} from image {attrs['Config']['Image']}")
        # 4. verify the replica (never assume the heal worked)
        details = self._verify_replica_details(new, self.verify_window)
        verified = details["ok"]
        after = forensic_snapshot.snapshot_container(new, incident_id, "after", forensics_dir=self.forensics_dir)
        inc["snapshots"] = {"before": before["summary_path"], "after": after["summary_path"]}
        # 5. feedback loop: re-weight vectors from the VERIFIED outcome
        self._apply_verified_feedback(incident_id, inc, verified, before)
        return {"healed": True, "verified": verified, "snapshot": before["artifact"], "details": details}

    def _respawn(self, attrs: dict):
        cfg = attrs["Config"]
        name = attrs["Name"].lstrip("/")
        nets = (self._isolated.get(name) or {}).get("networks") or [
            n for n in (attrs.get("NetworkSettings") or {}).get("Networks", {}) if n != QUARANTINE_NETWORK]
        network = next((n for n in nets if n not in ("bridge", "default")), None)
        kwargs = dict(name=name, detach=True, ports=self._ports_from_attrs(attrs), labels=cfg.get("Labels") or None,
                      environment=cfg.get("Env") or None, network=network)
        rp = (attrs.get("HostConfig") or {}).get("RestartPolicy") or {}
        if rp.get("Name"):
            kwargs["restart_policy"] = {"Name": rp["Name"], "MaximumRetryCount": rp.get("MaximumRetryCount", 0)}
        return self.docker.containers.run(cfg["Image"], **{k: v for k, v in kwargs.items() if v is not None})

    @staticmethod
    def _ports_from_attrs(attrs: dict) -> dict:
        ports = {}
        for cport, binds in ((attrs.get("HostConfig") or {}).get("PortBindings") or {}).items():
            specs = [((b.get("HostIp") or "0.0.0.0"), int(b["HostPort"])) if b.get("HostPort") else None for b in binds or []]
            if specs:
                ports[cport] = specs[0] if len(specs) == 1 else specs
        return ports

    def _verify_replica(self, container, window_seconds: float = 5) -> bool:
        return self._verify_replica_details(container, window_seconds)["ok"]

    def _verify_replica_details(self, container, window_seconds: float = 5) -> dict:
        """Canary window: the replica must be running and idle (CPU under the calibrated healthy ceiling)."""
        time.sleep(window_seconds)
        container.reload()
        running = container.status == "running"
        stats = self.docker.api.stats(container.id, stream=False)
        cpu = self._compute_cpu_pct(stats)
        ok = bool(running and cpu < self.verify_cpu_max)
        return {"ok": ok, "running": running, "cpu_pct": round(cpu, 3), "ram_pct": round(compute_ram_pct(stats), 3),
                "cpu_ceiling": round(self.verify_cpu_max, 3), "window_s": window_seconds}

    @staticmethod
    def _compute_cpu_pct(stats: dict) -> float:
        return compute_cpu_pct(stats)

    # ------------------------------------------------------------------ feedback
    def _apply_verified_feedback(self, incident_id: str, inc: dict, verified: bool, before_summary: dict):
        """Re-weight vectors ONLY from verified heals.

        Why not the literal `for vec: update(vec, was_correct=verified)` of the build prompt: updating all four
        vectors with the same boolean keeps their accuracies equal, so the weights could never diverge. Here each
        vector is judged on whether its own verdict (score >= 0.5) agrees with whether the incident was confirmed by
        independent evidence (a severe Falco rule hit, or a known-bad process in the forensic snapshot).
        Unverified heals produce no update at all.
        """
        evidence = [f"falco:{f['rule']}" for f in inc.get("falco", []) if str(f.get("priority", "")).lower() in SEVERE_PRIORITIES]
        evidence += [f"proc:{p['name']}" for p in before_summary.get("processes", []) if p.get("name") in SUSPICIOUS_PROCS]
        inc["evidence"] = sorted(set(evidence))
        if not verified:
            self._log_action(inc, "weights_unchanged", "replica failed verification - no update from unverified outcome")
            return
        confirmed = bool(inc["evidence"])
        for vec, score in (inc.get("trigger_scores") or {}).items():
            if score is None:
                continue
            self.tracker.update(vec, was_correct=((score >= 0.5) == confirmed), incident_id=incident_id,
                                context={"verified": True, "attack_confirmed": confirmed, "vector_score": score,
                                         "evidence": inc["evidence"]})
        self._log_action(inc, "weights_updated", f"verified heal; attack_confirmed={confirmed}")

    # ------------------------------------------------------------------ views
    def state(self) -> dict:
        with self._lock:
            return {"tau": self.tau, "margin": self.margin, "weights": self.tracker.current_weights(),
                    "history": list(self.history)[-300:], "dry_run": self.dry_run or self.docker is None,
                    "trained_on": self.meta["trained_on"], "feature_set": self.meta["feature_set"],
                    "ics_enabled": self.identity_model is not None, "isolated": list(self._isolated),
                    "algorithm_mode": self.algorithm_mode,
                    "spectra_active": bool(self.spectra and self.spectra.active),
                    "spectra_weight": self.spectra_weight,
                    "spectra_threshold": getattr(self.spectra, "threshold", 0.7724) if self.spectra else None,
                    "incidents": [self.public_incident(i) for i in sorted(self.incidents.values(), key=lambda i: -i["opened_at"])[:50]]}