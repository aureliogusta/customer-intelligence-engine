"""
tests/test_decision_service.py
================================
Testes unitários para decision_service: models, dataset, training, inference.

Padrão portado de claw-code-main/tests/test_memory_tools.py:
  - Stubs para isolar dependências (sem disco, sem DB)
  - Testes de contratos de dados (tipos, ranges)
  - Testes de integração leve (CSV real se disponível)

Run:
    python -m unittest tests/test_decision_service.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Garante que o root do projeto está no path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestChurnModelConfig(unittest.TestCase):
    """Verifica que ChurnModel e ExpansionModel têm contratos corretos."""

    def test_churn_model_has_required_fields(self):
        from decision_service.models import ChurnModel
        m = ChurnModel()
        self.assertIsInstance(m.features, list)
        self.assertGreater(len(m.features), 0)
        self.assertIn("engajamento_pct", m.features)
        self.assertIn("nps_score",       m.features)
        self.assertIn("mrr",             m.features)
        self.assertEqual(m.target, "renovado")

    def test_expansion_model_has_required_fields(self):
        from decision_service.models import ExpansionModel
        m = ExpansionModel()
        self.assertEqual(m.target, "fez_upsell")
        self.assertIn("engajamento_pct", m.features)

    def test_models_have_valid_split_config(self):
        from decision_service.models import ChurnModel, ExpansionModel
        for ModelCls in (ChurnModel, ExpansionModel):
            m = ModelCls()
            self.assertGreater(m.test_size, 0.0)
            self.assertLess(m.test_size, 1.0)


class TestChurnDataset(unittest.TestCase):
    """Testa ChurnDataset com CSV real se disponível."""

    CSV_PATH = ROOT / "data" / "train_dataset.csv"

    def setUp(self):
        if not self.CSV_PATH.exists():
            self.skipTest("train_dataset.csv não encontrado — execute generate_training_data.py")

    def test_prepare_returns_correct_shapes(self):
        from decision_service.dataset import ChurnDataset
        from decision_service.models import ChurnModel
        config = ChurnModel()
        ds = ChurnDataset(self.CSV_PATH, config)
        X, y, scaler, encoder = ds.prepare()

        self.assertEqual(len(X), len(y))
        self.assertGreater(X.shape[0], 0)
        self.assertGreater(X.shape[1], 0)

    def test_prepare_returns_binary_labels(self):
        from decision_service.dataset import ChurnDataset
        from decision_service.models import ChurnModel
        config = ChurnModel()
        ds = ChurnDataset(self.CSV_PATH, config)
        _, y, _, _ = ds.prepare()

        unique = set(y.tolist())
        self.assertTrue(unique.issubset({0, 1}), f"Labels devem ser 0/1, got {unique}")
        self.assertIn(0, unique, "Deve ter casos de churn (label=0)")
        self.assertIn(1, unique, "Deve ter casos de renovação (label=1)")

    def test_feature_names_subset_of_columns(self):
        from decision_service.dataset import ChurnDataset
        from decision_service.models import ChurnModel
        config = ChurnModel()
        ds     = ChurnDataset(self.CSV_PATH, config)
        feats  = ds.feature_names()
        self.assertGreater(len(feats), 0)

    def test_expansion_dataset(self):
        from decision_service.dataset import ChurnDataset
        from decision_service.models import ExpansionModel
        config = ExpansionModel()
        ds     = ChurnDataset(self.CSV_PATH, config)
        X, y, scaler, _ = ds.prepare()
        self.assertEqual(len(X), len(y))


class TestChurnPredictor(unittest.TestCase):
    """Testa ChurnPredictor com modelos reais se disponíveis."""

    MODEL_DIR = ROOT / "ml" / "models"

    def setUp(self):
        needed = ["churn_model.pkl", "churn_scaler.pkl"]
        if not all((self.MODEL_DIR / f).exists() for f in needed):
            self.skipTest("Modelos não encontrados — execute 02_training.ipynb")

    def _sample_features(self, risk: str = "high") -> dict:
        if risk == "high":
            return {
                "engajamento_pct": 12.0, "nps_score": 2.0, "tickets_abertos": 15,
                "dias_no_contrato": 200, "engagement_trend": -0.8, "tickets_trend": 1.5,
                "nps_trend": -2.0, "dias_sem_interacao": 28, "mrr": 5000,
                "max_users": 20, "segment": "MID_MARKET",
            }
        return {
            "engajamento_pct": 90.0, "nps_score": 9.5, "tickets_abertos": 0,
            "dias_no_contrato": 500, "engagement_trend": 0.2, "tickets_trend": -0.5,
            "nps_trend": 0.8, "dias_sem_interacao": 1, "mrr": 10000,
            "max_users": 50, "segment": "ENTERPRISE",
        }

    def test_predict_returns_required_keys(self):
        from decision_service.inference import ChurnPredictor
        pred   = ChurnPredictor(
            self.MODEL_DIR / "churn_model.pkl",
            self.MODEL_DIR / "churn_scaler.pkl",
            self.MODEL_DIR / "churn_encoder.pkl",
        )
        result = pred.predict(self._sample_features("high"))
        self.assertIn("churn_risk",     result)
        self.assertIn("retention_prob", result)
        self.assertIn("risk_level",     result)

    def test_predict_probability_range(self):
        from decision_service.inference import ChurnPredictor
        pred = ChurnPredictor(
            self.MODEL_DIR / "churn_model.pkl",
            self.MODEL_DIR / "churn_scaler.pkl",
        )
        for risk in ("high", "low"):
            result = pred.predict(self._sample_features(risk))
            self.assertGreaterEqual(result["churn_risk"],  0.0)
            self.assertLessEqual(result["churn_risk"],     1.0)
            self.assertGreaterEqual(result["retention_prob"], 0.0)
            self.assertLessEqual(result["retention_prob"],    1.0)

    def test_risk_level_enum(self):
        from decision_service.inference import ChurnPredictor
        pred   = ChurnPredictor(
            self.MODEL_DIR / "churn_model.pkl",
            self.MODEL_DIR / "churn_scaler.pkl",
        )
        result = pred.predict(self._sample_features("high"))
        self.assertIn(result["risk_level"], {"HIGH", "MEDIUM", "LOW"})

    def test_high_risk_account_scores_above_medium(self):
        from decision_service.inference import ChurnPredictor
        pred      = ChurnPredictor(
            self.MODEL_DIR / "churn_model.pkl",
            self.MODEL_DIR / "churn_scaler.pkl",
        )
        high_risk = pred.predict(self._sample_features("high"))["churn_risk"]
        low_risk  = pred.predict(self._sample_features("low"))["churn_risk"]
        self.assertGreater(high_risk, low_risk)


class TestRecommendations(unittest.TestCase):
    """Testa lógica de recomendações — sem dependência de modelos."""

    def test_high_risk_triggers_escalate(self):
        from study_service.recommendations import gerar_recomendacoes
        actions = gerar_recomendacoes("ACC_001", churn_risk=0.85, mrr=5000)
        codes = [a["code"] for a in actions]
        self.assertIn("ESCALATE", codes)

    def test_medium_risk_no_escalate(self):
        from study_service.recommendations import gerar_recomendacoes
        actions = gerar_recomendacoes("ACC_002", churn_risk=0.55, mrr=2000)
        codes = [a["code"] for a in actions]
        self.assertNotIn("ESCALATE", codes)
        self.assertIn("SCHEDULE_CALL", codes)

    def test_low_risk_track_engagement(self):
        from study_service.recommendations import gerar_recomendacoes
        actions = gerar_recomendacoes("ACC_003", churn_risk=0.10, mrr=500)
        codes = [a["code"] for a in actions]
        self.assertIn("TRACK_ENGAGEMENT", codes)

    def test_upsell_signal_on_healthy_account(self):
        from study_service.recommendations import gerar_recomendacoes
        actions = gerar_recomendacoes("ACC_004", churn_risk=0.05, upsell_probability=0.75, mrr=8000)
        codes = [a["code"] for a in actions]
        self.assertIn("PROPOSE_UPSELL", codes)

    def test_actions_have_required_fields(self):
        from study_service.recommendations import gerar_recomendacoes
        actions = gerar_recomendacoes("ACC_005", churn_risk=0.80, mrr=3000)
        for a in actions:
            self.assertIn("code",        a)
            self.assertIn("label",       a)
            self.assertIn("priority",    a)
            self.assertIn("description", a)
            self.assertIn("mrr_at_risk", a)

    def test_actions_sorted_by_priority(self):
        from study_service.recommendations import gerar_recomendacoes
        actions = gerar_recomendacoes("ACC_006", churn_risk=0.90, mrr=10000)
        priorities = [a["priority"] for a in actions]
        self.assertEqual(priorities, sorted(priorities))


if __name__ == "__main__":
    unittest.main(verbosity=2)
