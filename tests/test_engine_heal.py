"""End-to-end test of Reason -> Decide -> Heal -> Verify -> re-weight using a FAKE docker daemon and
models trained on SYNTHETIC data (clearly not NSL-KDD). It validates control flow, not detection quality."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import train_models  # noqa: E402
from engine.adaptive_weights import AdaptiveWeightTracker  # noqa: E402
from engine.hmr_engine import HMREngine  # noqa: E402
from forensics.forensic_snapshot import decrypt_snapshot, load_summaries  # noqa: E402
from tests.fakes import FakeContainer, FakeDocker  # noqa: E402

ATTACK_FLOW = {"duration": 0.0, "protocol_type": "tcp", "src_bytes": 20.0, "dst_bytes": 0.0, "count": 400.0,
               "srv_count": 400.0, "same_srv_rate": 1.0, "diff_srv_rate": 0.0}
BENIGN_FLOW = {"duration": 40.0, "protocol_type": "tcp", "src_bytes": 900.0, "dst_bytes": 1500.0, "count": 4.0,
               "srv_count": 4.0, "same_srv_rate": 0.8, "diff_srv_rate": 0.2}
FALCO = [{"rule": "Read sensitive file untrusted", "priority": "Warning"}]


class EngineHealTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        train_models.main(["--allow-synthetic", "--models-dir", str(cls.tmp / "models")])

    def make(self, docker, **kw):
        return HMREngine(models_dir=self.tmp / "models", docker_client=docker, forensics_dir=self.tmp / "for",
                         tracker=AdaptiveWeightTracker(audit_path=self.tmp / "audit.jsonl"), verify_window=0,
                         cooldown=0, **kw)

    def pin_scores(self, eng, rf, a):
        """The synthetic-trained model is not meaningful, so control-flow tests pin the network vector scores."""
        eng._score_network = lambda flow: (rf, a)

    def test_real_model_path_returns_probabilities(self):
        eng = self.make(FakeDocker())
        rf, a = eng._score_network(BENIGN_FLOW)
        self.assertTrue(0.0 <= rf <= 1.0 and 0.0 <= a <= 1.0)

    def ev(self, c, flow, falco=None):
        return {"container_id": c.id, "container_name": c.name, "ts": time.time(), "flow": flow,
                "resources": {"cpu_util": 1.0, "ram_util": 2.0, "net_io_rate": 100.0}, "identity": None, "falco": falco or []}

    def test_benign_stays_monitor(self):
        d = FakeDocker(); c = d.add(FakeContainer("victim", "sentinel-victim", ports=True))
        eng = self.make(d); self.pin_scores(eng, 0.02, 0.10)
        out = eng.process_event(self.ev(c, BENIGN_FLOW))
        self.assertIn(out["tier"], ("MONITOR", "LOG_WATCH"))
        self.assertNotIn("kill", out["action"])
        self.assertEqual(c.calls, [])

    def test_attack_triggers_kill_restart_verify_and_reweight(self):
        d = FakeDocker(); c = d.add(FakeContainer("victim", "sentinel-victim", ports=True))
        eng = self.make(d); self.pin_scores(eng, 0.97, 0.95)
        w0 = eng.tracker.current_weights()
        outs = [eng.process_event(self.ev(c, ATTACK_FLOW, FALCO)) for _ in range(2)]
        self.assertEqual(outs[-1]["tier"], "KILL_RESTART", outs[-1])
        self.assertEqual(outs[-1]["action"], "kill_restart_started")
        for _ in range(100):                      # heal runs in a thread
            inc = next(iter(eng.incidents.values()))
            if inc["status"] in ("healed", "heal_failed"):
                break
            time.sleep(0.05)
        self.assertEqual(inc["status"], "healed", inc["actions"])
        self.assertTrue(inc["verified"])
        self.assertEqual(c.calls, ["pause", "kill", "remove"])           # frozen, killed, removed
        new = d.created[-1]
        self.assertEqual(new.name, "victim")                              # respawned under the same name
        self.assertEqual(new.run_kwargs["ports"], {"22/tcp": ("0.0.0.0", 2222)})
        self.assertNotEqual(eng.tracker.current_weights(), w0)            # verified outcome re-weighted the fusion
        self.assertTrue((self.tmp / "audit.jsonl").exists())
        self.assertIn("falco:Read sensitive file untrusted", inc["evidence"])
        snaps = load_summaries(inc["id"], self.tmp / "for")
        self.assertEqual(set(snaps), {"before", "after"})
        self.assertTrue(any(p["name"] == "stress-ng" for p in snaps["before"]["processes"]))
        enc = Path(snaps["before"]["artifact"])
        self.assertTrue(enc.read_bytes()[:4] == b"gAAA")                  # Fernet token, i.e. actually encrypted
        self.assertGreater(len(decrypt_snapshot(enc, self.tmp / "for")), 100)

    def test_failed_verification_does_not_touch_weights(self):
        d = FakeDocker(); c = d.add(FakeContainer("victim2", "sentinel-victim", ports=True))
        eng = self.make(d); self.pin_scores(eng, 0.97, 0.95)
        orig = d._run
        d.containers.run = lambda image, **kw: (lambda n: (setattr(n, "cpu", 50.0), n)[1])(orig(image, **kw))  # hot replica
        w0 = eng.tracker.current_weights()
        for _ in range(2):
            eng.process_event(self.ev(c, ATTACK_FLOW, FALCO))
        for _ in range(100):
            inc = next(iter(eng.incidents.values()))
            if inc["status"] in ("healed", "heal_failed"):
                break
            time.sleep(0.05)
        self.assertFalse(inc["verified"])
        self.assertEqual(eng.tracker.current_weights(), w0)

    def test_dry_run_never_touches_docker(self):
        d = FakeDocker(); c = d.add(FakeContainer("victim3", "sentinel-victim", ports=True))
        eng = self.make(d, dry_run=True); self.pin_scores(eng, 0.97, 0.95)
        for _ in range(3):
            out = eng.process_event(self.ev(c, ATTACK_FLOW, FALCO))
        self.assertEqual(out["action"], "dry_run_kill_restart")
        self.assertEqual(c.calls, [])

    def test_isolate_moves_to_internal_network_and_releases(self):
        d = FakeDocker(); c = d.add(FakeContainer("victim4", "sentinel-victim", ports=True))
        eng = self.make(d)
        eng._execute_isolate("victim4", c.id)
        self.assertEqual(set(c.attrs["NetworkSettings"]["Networks"]), {"sentinel-quarantine"})
        self.assertTrue(eng.release_isolation("victim4"))
        self.assertEqual(set(c.attrs["NetworkSettings"]["Networks"]), {"bridge"})


if __name__ == "__main__":
    unittest.main()
