import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.identity_model import IdentityModel, circ_dist, timing_entropy  # noqa: E402
from engine.resource_model import ResourceDriftModel, fit_weights  # noqa: E402


def make_sessions(n_owner=20, n_intr=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_owner):
        rows.append(dict(session_id=f"o{i}", label="owner", source_ip="10.0.0.5",
                         start_hour=float(rng.normal(14, 1.5) % 24), lineage="sshd>bash>ls",
                         cmd_intervals=";".join(f"{x:.2f}" for x in rng.lognormal(1.5, 0.9, 8))))
    for i in range(n_intr):
        rows.append(dict(session_id=f"i{i}", label="intruder", source_ip="10.0.0.99",
                         start_hour=float(rng.normal(3, 1) % 24), lineage="sshd>bash>cat",
                         cmd_intervals=";".join(f"{x:.2f}" for x in rng.normal(0.2, 0.02, 8))))
    return pd.DataFrame(rows)


class IdentityModelTest(unittest.TestCase):
    def test_circ_dist_wraps_around_midnight(self):
        self.assertAlmostEqual(circ_dist(23, 1), 2)
        self.assertAlmostEqual(circ_dist(1, 23), 2)
        self.assertEqual(circ_dist(5, 5), 0)

    def test_scripted_bursts_have_lower_entropy_than_human_timing(self):
        human = timing_entropy([0.4, 3.1, 0.9, 7.2, 1.1, 0.3, 5.0])
        scripted = timing_entropy([0.2, 0.2, 0.21, 0.19, 0.2, 0.2])
        self.assertGreater(human, scripted)

    def test_owner_sessions_score_near_zero_intruder_near_one(self):
        df = make_sessions()
        m = IdentityModel(); m.fit(df)
        owner_scores = [m.score(r) for r in df[df.label == "owner"].to_dict("records")]
        intr_scores = [m.score(r) for r in df[df.label == "intruder"].to_dict("records")]
        self.assertLess(max(owner_scores), 0.3)
        self.assertGreater(min(intr_scores), 0.6)

    def test_roundtrip_through_dict(self):
        df = make_sessions()
        m = IdentityModel(); m.fit(df)
        m2 = IdentityModel.from_dict(m.to_dict())
        for r in df.sample(5, random_state=1).to_dict("records"):
            self.assertAlmostEqual(m.score(r), m2.score(r), places=9)

    def test_requires_minimum_owner_sessions(self):
        with self.assertRaises(ValueError):
            IdentityModel().fit(make_sessions(n_owner=2, n_intr=2))


class ResourceDriftModelTest(unittest.TestCase):
    def test_no_baseline_returns_none(self):
        m = ResourceDriftModel()
        self.assertIsNone(m.score("c1", {"cpu_util": 90, "ram_util": 10, "net_io_rate": 1}))

    def test_baseline_then_drift_detected(self):
        m = ResourceDriftModel(weights={"cpu_util": 0.6, "ram_util": 0.2, "net_io_rate": 0.2})
        for _ in range(15):
            m.observe_healthy("c1", {"cpu_util": 5.0, "ram_util": 10.0, "net_io_rate": 500.0})
        self.assertTrue(m.ready("c1"))
        calm = m.score("c1", {"cpu_util": 6.0, "ram_util": 10.0, "net_io_rate": 520.0})
        spike = m.score("c1", {"cpu_util": 95.0, "ram_util": 10.0, "net_io_rate": 500.0})
        self.assertLess(calm, 0.2)
        self.assertGreater(spike, calm)

    def test_fit_weights_from_correlated_csv(self):
        import tempfile
        rng = np.random.default_rng(0)
        n = 400
        label = (rng.random(n) < 0.5).astype(int)
        df = pd.DataFrame({
            "cpu_util": np.where(label == 1, rng.normal(90, 5, n), rng.normal(10, 5, n)),
            "ram_util": rng.normal(30, 10, n),  # uncorrelated with label
            "net_io_rate": np.where(label == 1, rng.normal(50, 20, n), rng.normal(500, 100, n)),
            "label": label,
        })
        d = Path(tempfile.mkdtemp()) / "rb.csv"
        df.to_csv(d, index=False)
        blob = fit_weights(d, d.parent / "w.json")
        self.assertGreater(blob["weights"]["cpu_util"], blob["weights"]["ram_util"])
        self.assertAlmostEqual(sum(blob["weights"].values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
