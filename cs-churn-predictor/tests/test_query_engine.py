"""
tests/test_query_engine.py
============================
Testes para ChurnQueryEngine, AuditTrail e DriftDetector.

Usa stubs para isolar dependências externas (modelos, pgvector),
portando o padrão _StubMemoryTools do claw-code.

Run:
    python -m unittest tests/test_query_engine.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _StubChurnPredictor:
    """Stub determinístico — não requer modelos em disco."""
    def predict(self, features: dict) -> dict:
        eng = features.get("engajamento_pct", 50)
        risk = 0.9 if eng < 20 else (0.5 if eng < 60 else 0.1)
        level = "HIGH" if risk > 0.7 else ("MEDIUM" if risk > 0.4 else "LOW")
        return {"churn_risk": risk, "retention_prob": 1 - risk, "risk_level": level}


class _StubExpansionPredictor:
    def predict(self, features: dict) -> dict:
        nps = features.get("nps_score", 5)
        prob = 0.6 if nps > 8 else 0.1
        return {"upsell_probability": prob, "upsell_signal": "HIGH" if prob > 0.5 else "LOW"}


class _StubMemoryStore:
    """Portado do _StubMemoryTools do claw-code."""
    def __init__(self):
        self.saved: list = []

    def recall(self, query="", account_id="default", limit=5):
        return []

    def remember_prediction(self, account_id, session_id, churn_risk, risk_level, actions, extra=None):
        self.saved.append({"account_id": account_id, "churn_risk": churn_risk})
        return len(self.saved)

    def recall_account(self, account_id, limit=5):
        return []


# ── ChurnQueryEngine Tests ────────────────────────────────────────────────────

class TestChurnQueryEngine(unittest.TestCase):

    def _make_engine(self, save=True):
        from decision_service.query_engine import ChurnQueryEngine, ChurnEngineConfig
        stub_mem = _StubMemoryStore()
        engine = ChurnQueryEngine(
            churn_predictor     = _StubChurnPredictor(),
            expansion_predictor = _StubExpansionPredictor(),
            memory_store        = stub_mem,
            config              = ChurnEngineConfig(save_predictions=save),
        )
        return engine, stub_mem

    def _high_risk_features(self) -> dict:
        return {
            "account_id": "ACC_001", "engajamento_pct": 10.0, "nps_score": 2.0,
            "tickets_abertos": 12, "dias_no_contrato": 200, "engagement_trend": -0.8,
            "tickets_trend": 1.2, "nps_trend": -1.5, "dias_sem_interacao": 25,
            "mrr": 5000, "max_users": 20, "segment": "MID_MARKET",
        }

    def _low_risk_features(self) -> dict:
        return {
            "account_id": "ACC_002", "engajamento_pct": 90.0, "nps_score": 9.0,
            "tickets_abertos": 0, "dias_no_contrato": 500, "engagement_trend": 0.1,
            "tickets_trend": -0.2, "nps_trend": 0.5, "dias_sem_interacao": 1,
            "mrr": 3000, "max_users": 50, "segment": "ENTERPRISE",
        }

    def test_analyze_returns_prediction_turn(self):
        from decision_service.query_engine import PredictionTurn
        engine, _ = self._make_engine()
        result = engine.analyze("ACC_001", self._high_risk_features())
        self.assertIsInstance(result, PredictionTurn)

    def test_analyze_high_risk_account(self):
        engine, _ = self._make_engine()
        result = engine.analyze("ACC_001", self._high_risk_features())
        self.assertEqual(result.risk_level, "HIGH")
        self.assertGreater(result.churn_risk, 0.7)

    def test_analyze_low_risk_account(self):
        engine, _ = self._make_engine()
        result = engine.analyze("ACC_002", self._low_risk_features())
        self.assertEqual(result.risk_level, "LOW")
        self.assertLess(result.churn_risk, 0.5)

    def test_analyze_includes_recommended_actions(self):
        engine, _ = self._make_engine()
        result = engine.analyze("ACC_001", self._high_risk_features())
        self.assertGreater(len(result.recommended_actions), 0)
        for action in result.recommended_actions:
            self.assertIn("code",  action)
            self.assertIn("label", action)

    def test_analyze_saves_to_memory(self):
        engine, stub_mem = self._make_engine(save=True)
        engine.analyze("ACC_001", self._high_risk_features())
        self.assertEqual(len(stub_mem.saved), 1)
        self.assertEqual(stub_mem.saved[0]["account_id"], "ACC_001")

    def test_analyze_session_id_assigned(self):
        engine, _ = self._make_engine()
        result = engine.analyze("ACC_001", self._high_risk_features())
        self.assertIsNotNone(result.session_id)
        self.assertGreater(len(result.session_id), 0)

    def test_analyze_mrr_at_risk_calculated(self):
        engine, _ = self._make_engine()
        result = engine.analyze("ACC_001", self._high_risk_features())
        expected = round(result.mrr * result.churn_risk, 2)
        self.assertAlmostEqual(result.mrr_at_risk, expected, places=1)

    def test_analyze_batch_returns_list(self):
        engine, _ = self._make_engine()
        accounts = [self._high_risk_features(), self._low_risk_features()]
        results  = engine.analyze_batch(accounts)
        self.assertEqual(len(results), 2)

    def test_risk_summary(self):
        engine, _ = self._make_engine()
        accounts = [self._high_risk_features(), self._low_risk_features()]
        turns    = engine.analyze_batch(accounts)
        summary  = engine.risk_summary(turns)
        self.assertEqual(summary["total"], 2)
        self.assertIn("avg_churn_risk",  summary)
        self.assertIn("mrr_at_risk",     summary)


# ── AuditTrail Tests ──────────────────────────────────────────────────────────

class TestChurnAuditTrail(unittest.TestCase):

    def test_log_prediction(self):
        from analytics.audit_trail import ChurnAuditTrail
        trail = ChurnAuditTrail()
        trail.log_prediction("ACC_001", "sess_1", churn_risk=0.85, risk_level="HIGH", mrr=5000)
        self.assertEqual(len(trail.entries), 1)
        self.assertEqual(trail.entries[0].event_type, "prediction")

    def test_log_recommendation(self):
        from analytics.audit_trail import ChurnAuditTrail
        trail = ChurnAuditTrail()
        trail.log_recommendation("ACC_001", "sess_1", ["ESCALATE", "SCHEDULE_CALL"])
        self.assertEqual(trail.entries[0].details["actions"], ["ESCALATE", "SCHEDULE_CALL"])

    def test_compact(self):
        from analytics.audit_trail import ChurnAuditTrail
        trail = ChurnAuditTrail()
        for i in range(50):
            trail.log_prediction(f"ACC_{i:03d}", "sess", 0.5, "MEDIUM", 1000)
        removed = trail.compact(keep_last=20)
        self.assertEqual(removed, 30)
        self.assertEqual(len(trail.entries), 20)

    def test_to_csv_headers(self):
        from analytics.audit_trail import ChurnAuditTrail
        trail = ChurnAuditTrail()
        trail.log_prediction("ACC_001", "sess_1", 0.9, "HIGH", 5000)
        csv_out = trail.to_csv()
        self.assertIn("timestamp",   csv_out)
        self.assertIn("account_id",  csv_out)
        self.assertIn("event_type",  csv_out)

    def test_for_account_filters(self):
        from analytics.audit_trail import ChurnAuditTrail
        trail = ChurnAuditTrail()
        trail.log_prediction("ACC_001", "sess_1", 0.9, "HIGH",   5000)
        trail.log_prediction("ACC_002", "sess_1", 0.2, "LOW",    1000)
        trail.log_prediction("ACC_001", "sess_2", 0.8, "HIGH",   5000)
        entries = trail.for_account("ACC_001")
        self.assertEqual(len(entries), 2)

    def test_summary(self):
        from analytics.audit_trail import ChurnAuditTrail
        trail = ChurnAuditTrail()
        trail.log_prediction("ACC_001", "s1", 0.9, "HIGH",   5000, mrr_at_risk=4500)
        trail.log_prediction("ACC_002", "s1", 0.1, "LOW",    1000, mrr_at_risk=100)
        s = trail.summary()
        self.assertEqual(s["predictions"], 2)
        self.assertIn("avg_churn_risk", s)


# ── DriftDetector Tests ───────────────────────────────────────────────────────

class TestDriftDetector(unittest.TestCase):

    def _make_baseline(self, n: int = 200, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "engajamento_pct":  rng.normal(60, 15, n).clip(0, 100),
            "nps_score":        rng.normal(7.0, 1.5, n).clip(1, 10),
            "tickets_abertos":  rng.poisson(3, n).astype(float),
            "dias_sem_interacao": rng.exponential(5, n),
            "mrr":              rng.uniform(500, 5000, n),
        })

    def test_no_drift_on_same_distribution(self):
        from analytics.drift_monitor import DriftDetector
        baseline = self._make_baseline(300)
        current  = self._make_baseline(100, seed=99)  # mesma distribuição, semente diferente
        detector = DriftDetector(baseline)
        report   = detector.check_drift(current)
        # Pode haver drift em alguns, mas não em todos os críticos
        self.assertIsNotNone(report)
        self.assertFalse(report.should_retrain)

    def test_severe_drift_detected(self):
        from analytics.drift_monitor import DriftDetector
        baseline = self._make_baseline(300)
        # Distribuição muito diferente (engajamento caiu de 60 → 15)
        rng     = np.random.default_rng(0)
        current = pd.DataFrame({
            "engajamento_pct":  rng.normal(15, 5, 100).clip(0, 100),
            "nps_score":        rng.normal(2.0, 0.5, 100).clip(1, 10),
            "tickets_abertos":  rng.poisson(15, 100).astype(float),
            "dias_sem_interacao": rng.exponential(25, 100),
            "mrr":              rng.uniform(500, 5000, 100),
        })
        detector = DriftDetector(baseline)
        report   = detector.check_drift(current)
        drifted  = [d.feature_name for d in report.drifts if d.drift_detected]
        self.assertGreater(len(drifted), 0, "Deve detectar drift em distribuição severamente diferente")

    def test_report_fields_present(self):
        from analytics.drift_monitor import DriftDetector
        baseline = self._make_baseline(200)
        current  = self._make_baseline(50)
        report   = DriftDetector(baseline).check_drift(current)
        self.assertIsNotNone(report.timestamp)
        self.assertEqual(report.n_baseline, 200)
        self.assertEqual(report.n_current, 50)
        self.assertIsInstance(report.drifts, list)

    def test_summary_line(self):
        from analytics.drift_monitor import DriftDetector
        baseline = self._make_baseline(200)
        current  = self._make_baseline(50)
        report   = DriftDetector(baseline).check_drift(current)
        line     = report.summary_line()
        self.assertIn("DriftReport", line)
        self.assertIn("Retreinar",   line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
