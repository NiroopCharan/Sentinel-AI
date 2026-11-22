import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.adaptive_weights import AdaptiveWeightTracker  # noqa: E402
from engine.fusion import fuse, tier_for  # noqa: E402


class FusionTest(unittest.TestCase):
    def test_fuse_all_four_present_sums_to_weighted_average(self):
        w = {"rf": 0.4, "as": 0.25, "rds": 0.15, "ics": 0.2}
        ts, eff = fuse({"rf": 1.0, "as": 1.0, "rds": 1.0, "ics": 1.0}, w)
        self.assertAlmostEqual(ts, 1.0)
        self.assertAlmostEqual(sum(eff.values()), 1.0)

    def test_fuse_renormalises_when_vectors_missing(self):
        w = {"rf": 0.4, "as": 0.25, "rds": 0.15, "ics": 0.2}
        ts, eff = fuse({"rf": 0.8, "as": None, "rds": None, "ics": None}, w)
        self.assertAlmostEqual(ts, 0.8)
        self.assertEqual(set(eff), {"rf"})

    def test_fuse_no_vectors_is_zero(self):
        ts, eff = fuse({"rf": None, "as": None, "rds": None, "ics": None}, {"rf": 1})
        self.assertEqual(ts, 0.0)
        self.assertEqual(eff, {})

    def test_tiers_follow_build_prompt_thresholds(self):
        tau, margin = 0.60, 0.20
        self.assertEqual(tier_for(0.0, tau, margin), "MONITOR")
        self.assertEqual(tier_for(0.29, tau, margin), "MONITOR")
        self.assertEqual(tier_for(0.30, tau, margin), "LOG_WATCH")
        self.assertEqual(tier_for(0.59, tau, margin), "LOG_WATCH")
        self.assertEqual(tier_for(0.60, tau, margin), "ISOLATE")
        self.assertEqual(tier_for(0.79, tau, margin), "ISOLATE")
        self.assertEqual(tier_for(0.80, tau, margin), "KILL_RESTART")
        self.assertEqual(tier_for(1.0, tau, margin), "KILL_RESTART")


class AdaptiveWeightsTest(unittest.TestCase):
    def test_starts_at_prescribed_weights(self):
        t = AdaptiveWeightTracker()
        w = t.current_weights()
        for k, v in {"rf": 0.40, "as": 0.25, "rds": 0.15, "ics": 0.20}.items():
            self.assertAlmostEqual(w[k], v, places=6)

    def test_repeated_correct_outcomes_raise_a_vectors_weight(self):
        t = AdaptiveWeightTracker()
        w0 = t.current_weights()["ics"]
        for _ in range(40):
            t.update("ics", True)
        self.assertGreater(t.current_weights()["ics"], w0)

    def test_repeated_incorrect_outcomes_lower_a_vectors_weight(self):
        t = AdaptiveWeightTracker()
        w0 = t.current_weights()["rf"]
        for _ in range(40):
            t.update("rf", False)
        self.assertLess(t.current_weights()["rf"], w0)

    def test_state_persists_across_instances(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        t1 = AdaptiveWeightTracker(state_path=d / "s.json", audit_path=d / "a.jsonl")
        for _ in range(20):
            t1.update("as", False)
        t2 = AdaptiveWeightTracker(state_path=d / "s.json", audit_path=d / "a.jsonl")
        self.assertAlmostEqual(t2.current_weights()["as"], t1.current_weights()["as"], places=6)
        self.assertTrue((d / "a.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
