"""Unit tests for Dual-Algorithm Integration (HMR + SPECTRA-CRBAC v2)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR, SPECTRA_MODEL_PATH
from engine.hmr_engine import HMREngine
from engine.spectra_engine import SpectraEngine
from engine.spectra_features import build_feature_vector


class TestSpectraEngine(unittest.TestCase):
    def setUp(self):
        self.spectra = SpectraEngine(model_path=SPECTRA_MODEL_PATH)

    def test_spectra_initialization(self):
        self.assertTrue(self.spectra.active, "SPECTRA engine should be active with valid model file")
        self.assertEqual(len(self.spectra.features), 103, "Should have 103 LID-DS syscall features")
        self.assertGreater(self.spectra.threshold, 0.5, "Threshold should be calibrated")

    def test_spectra_evaluate_zero_vector(self):
        vec = np.zeros(103)
        res = self.spectra.evaluate_vector("test-session", vec)
        self.assertTrue(res["active"])
        self.assertIn("spectra_risk", res)
        self.assertIn("threat_prob", res)
        self.assertIn("anomaly_quantile", res)
        self.assertIn("evidence_score", res)
        self.assertIn("is_anomalous", res)
        self.assertFalse(res["is_anomalous"], "Zero vector should not be anomalous")

    def test_spectra_sequential_evidence_accumulation(self):
        # High simulated activity should increase cumulative evidence
        vec = np.ones(103) * 100.0
        r1 = self.spectra.evaluate_vector("sess-acc", vec)
        r2 = self.spectra.evaluate_vector("sess-acc", vec)
        self.assertGreaterEqual(r2["evidence_score"], r1["evidence_score"])

        # Reset session
        self.spectra.reset_session("sess-acc")
        self.assertNotIn("sess-acc", self.spectra.cusum_state)


class TestDualAlgorithmHMREngine(unittest.TestCase):
    def setUp(self):
        # Initialize engine in dry-run mode
        self.engine = HMREngine(models_dir=MODELS_DIR, dry_run=True, algorithm_mode="HYBRID", spectra_weight=0.5)

    def test_engine_state_reports_dual_algorithm(self):
        st = self.engine.state()
        self.assertEqual(st["algorithm_mode"], "HYBRID")
        self.assertTrue(st["spectra_active"])
        self.assertEqual(st["spectra_weight"], 0.5)
        self.assertIsNotNone(st["spectra_threshold"])

    def test_process_event_hybrid_mode(self):
        event = {
            "container_id": "test-c1",
            "container_name": "test-c1",
            "flow": {
                "duration": 40.0, "protocol_type": "tcp", "src_bytes": 900.0, "dst_bytes": 1500.0,
                "count": 4.0, "srv_count": 4.0, "same_srv_rate": 0.8, "diff_srv_rate": 0.2
            },
            "resources": {"cpu_util": 5.0, "ram_util": 12.0, "net_io_rate": 200.0},
            "spectra_features": {
                "completed_syscalls": 10.0, "failed_syscalls": 0.0, "network_recvfrom": 2.0
            }
        }
        res = self.engine.process_event(event)
        self.assertIn("score", res)
        self.assertIn("hmr_score", res)
        self.assertIn("spectra", res)
        self.assertEqual(res["algorithm_mode"], "HYBRID")
        self.assertIsNotNone(res["spectra"])
        self.assertIn("spectra_risk", res["spectra"])

        # Composite score should be the average of HMR and SPECTRA when weight is 0.5
        expected = 0.5 * res["hmr_score"] + 0.5 * res["spectra"]["spectra_risk"]
        self.assertAlmostEqual(res["score"], expected, places=4)

    def test_process_event_hmr_only_mode(self):
        engine_hmr = HMREngine(models_dir=MODELS_DIR, dry_run=True, algorithm_mode="HMR_ONLY")
        event = {
            "container_id": "test-c2",
            "container_name": "test-c2",
            "flow": {
                "duration": 40.0, "protocol_type": "tcp", "src_bytes": 900.0, "dst_bytes": 1500.0,
                "count": 4.0, "srv_count": 4.0, "same_srv_rate": 0.8, "diff_srv_rate": 0.2
            },
            "spectra_features": {"completed_syscalls": 10.0}
        }
        res = engine_hmr.process_event(event)
        self.assertEqual(res["score"], res["hmr_score"])
        self.assertEqual(res["algorithm_mode"], "HMR_ONLY")

    def test_process_event_spectra_only_mode(self):
        engine_sp = HMREngine(models_dir=MODELS_DIR, dry_run=True, algorithm_mode="SPECTRA_ONLY")
        event = {
            "container_id": "test-c3",
            "container_name": "test-c3",
            "flow": {
                "duration": 40.0, "protocol_type": "tcp", "src_bytes": 900.0, "dst_bytes": 1500.0,
                "count": 4.0, "srv_count": 4.0, "same_srv_rate": 0.8, "diff_srv_rate": 0.2
            },
            "spectra_features": {"completed_syscalls": 10.0}
        }
        res = engine_sp.process_event(event)
        self.assertEqual(res["score"], res["spectra"]["spectra_risk"])
        self.assertEqual(res["algorithm_mode"], "SPECTRA_ONLY")


if __name__ == "__main__":
    unittest.main()
