"""
revenue_automation/tests/test_revenue_automation.py
=====================================================
Suite de testes para o subsistema de automação de intervenção.

Cobertura:
  - PolicyEngine: regras embutidas, avaliação de condições, stop flag, dedup
  - ActionDispatcher: dry-run, fallback CONSOLE, tratamento de falhas
  - FeedbackStore: record, for_account, for_correlation, outcome_counts, total
  - ReportBuilder: campos computados, breakdown de segmentos
  - InterventionEngine: batch com stubs, tratamento de erros por conta

Todos os testes usam stubs/mocks ou SQLite in-memory para não depender
de modelos treinados, arquivos externos, ou canais de rede.
"""

from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path bootstrap — garante que o root do projeto está em sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Helpers para construir objetos de teste sem dependências externas
# ---------------------------------------------------------------------------

def _make_context(
    account_id: str = "TEST_001",
    churn_risk: float = 0.85,
    mrr: float = 8000.0,
    segment: str = "ENTERPRISE",
    upsell_probability: float = 0.1,
    dias_sem_interacao: int = 5,
    nps_score: float = 3.0,
    engagement_trend: float = -0.4,
    tickets_abertos: int = 3,
    dias_no_contrato: int = 365,
    nps_trend: float = -0.2,
    tickets_trend: float = 0.2,
):
    from revenue_automation.schemas.models import InterventionContext
    return InterventionContext(
        account_id         = account_id,
        session_id         = "sess_test",
        churn_risk         = churn_risk,
        retention_prob     = 1.0 - churn_risk,
        risk_level         = "HIGH" if churn_risk >= 0.60 else ("MEDIUM" if churn_risk >= 0.35 else "LOW"),
        upsell_probability = upsell_probability,
        mrr_at_risk        = mrr * churn_risk,
        mrr                = mrr,
        segment            = segment,
        dias_sem_interacao = dias_sem_interacao,
        tickets_abertos    = tickets_abertos,
        engajamento_pct    = 30.0,
        nps_score          = nps_score,
        dias_no_contrato   = dias_no_contrato,
        engagement_trend   = engagement_trend,
        nps_trend          = nps_trend,
        tickets_trend      = tickets_trend,
    )


# ===========================================================================
# 1. PolicyEngine
# ===========================================================================

class TestPolicyEngineBuiltinRules(unittest.TestCase):
    """Testa o PolicyEngine com regras embutidas (_builtin_rules)."""

    def _engine(self):
        from revenue_automation.policy.engine import PolicyEngine, _builtin_rules
        engine = PolicyEngine.__new__(PolicyEngine)
        engine.rules_path = Path("/nonexistent/rules.yaml")
        engine.rules = _builtin_rules()
        return engine

    def test_critical_high_mrr_triggers_escalation(self):
        """churn > 0.80 + mrr > 5000 deve disparar ESCALATE_TO_MANAGER."""
        engine = self._engine()
        ctx = _make_context(churn_risk=0.90, mrr=10000.0)
        decision = engine.evaluate(ctx)
        codes = decision.action_codes()
        self.assertIn("ESCALATE_TO_MANAGER", codes)

    def test_critical_high_mrr_stop_flag(self):
        """Regra com stop=True: apenas essa regra deve disparar (sem continuar)."""
        engine = self._engine()
        ctx = _make_context(churn_risk=0.90, mrr=10000.0)
        decision = engine.evaluate(ctx)
        # Como stop=True, apenas as ações da regra crítica devem aparecer
        self.assertIn("critical_high_mrr", decision.triggered_rules)
        # Regras de prioridade menor NÃO devem aparecer
        self.assertNotIn("healthy_monitor", decision.triggered_rules)

    def test_high_risk_no_high_mrr(self):
        """churn > 0.60 mas mrr <= 5000 → high_risk_action, não critical_high_mrr."""
        engine = self._engine()
        ctx = _make_context(churn_risk=0.65, mrr=2000.0)
        decision = engine.evaluate(ctx)
        self.assertNotIn("ESCALATE_TO_MANAGER", decision.action_codes())
        self.assertTrue(
            "CREATE_CRM_TASK" in decision.action_codes()
            or "SCHEDULE_CALL" in decision.action_codes()
        )

    def test_upsell_opportunity(self):
        """upsell > 0.50 + churn < 0.30 → PROPOSE_UPSELL."""
        engine = self._engine()
        ctx = _make_context(churn_risk=0.10, mrr=1000.0, upsell_probability=0.70)
        decision = engine.evaluate(ctx)
        self.assertIn("PROPOSE_UPSELL", decision.action_codes())

    def test_healthy_account_monitor_only(self):
        """churn < 0.40 sem upsell → MONITOR_ONLY."""
        engine = self._engine()
        ctx = _make_context(churn_risk=0.20, mrr=1000.0, upsell_probability=0.10)
        decision = engine.evaluate(ctx)
        self.assertIn("MONITOR_ONLY", decision.action_codes())
        self.assertTrue(decision.monitor_only)

    def test_dedup_actions(self):
        """Ações duplicadas de várias regras devem ser deduplicadas por code."""
        engine = self._engine()
        ctx = _make_context(churn_risk=0.65, mrr=3000.0)
        decision = engine.evaluate(ctx)
        codes = decision.action_codes()
        self.assertEqual(len(codes), len(set(codes)), "Ações duplicadas encontradas")

    def test_decision_has_required_fields(self):
        """PolicyDecision deve ter account_id, session_id, timestamp e triggered_rules."""
        engine = self._engine()
        ctx = _make_context()
        decision = engine.evaluate(ctx)
        self.assertEqual(decision.account_id, "TEST_001")
        self.assertEqual(decision.session_id, "sess_test")
        self.assertIsInstance(decision.timestamp, str)
        self.assertIsInstance(decision.triggered_rules, list)

    def test_top_action_returns_highest_priority(self):
        """top_action() deve retornar a ação de prioridade mais alta."""
        engine = self._engine()
        ctx = _make_context(churn_risk=0.90, mrr=10000.0)
        decision = engine.evaluate(ctx)
        top = decision.top_action()
        self.assertIsNotNone(top)
        if len(decision.actions) > 1:
            self.assertLessEqual(top.priority, decision.actions[1].priority)


class TestConditionEvaluator(unittest.TestCase):
    """Testa _eval_condition com todos os operadores suportados."""

    def setUp(self):
        from revenue_automation.policy.engine import _eval_condition
        self.eval = _eval_condition

    def test_gt(self):
        self.assertTrue(self.eval(0.9, {"gt": 0.8}))
        self.assertFalse(self.eval(0.7, {"gt": 0.8}))

    def test_lt(self):
        self.assertTrue(self.eval(0.3, {"lt": 0.4}))
        self.assertFalse(self.eval(0.5, {"lt": 0.4}))

    def test_gte(self):
        self.assertTrue(self.eval(0.8, {"gte": 0.8}))
        self.assertFalse(self.eval(0.79, {"gte": 0.8}))

    def test_lte(self):
        self.assertTrue(self.eval(0.4, {"lte": 0.4}))
        self.assertFalse(self.eval(0.41, {"lte": 0.4}))

    def test_eq(self):
        self.assertTrue(self.eval("ENTERPRISE", {"eq": "enterprise"}))
        self.assertFalse(self.eval("SMB", {"eq": "ENTERPRISE"}))

    def test_ne(self):
        self.assertTrue(self.eval("SMB", {"ne": "ENTERPRISE"}))
        self.assertFalse(self.eval("ENTERPRISE", {"ne": "ENTERPRISE"}))

    def test_in(self):
        self.assertTrue(self.eval("ENTERPRISE", {"in": ["ENTERPRISE", "MID_MARKET"]}))
        self.assertFalse(self.eval("SMB", {"in": ["ENTERPRISE", "MID_MARKET"]}))

    def test_not_in(self):
        self.assertTrue(self.eval("SMB", {"not_in": ["ENTERPRISE", "MID_MARKET"]}))
        self.assertFalse(self.eval("ENTERPRISE", {"not_in": ["ENTERPRISE", "MID_MARKET"]}))

    def test_multiple_operators_and(self):
        """Múltiplos operadores para o mesmo campo = AND."""
        self.assertTrue(self.eval(0.75, {"gt": 0.60, "lt": 0.90}))
        self.assertFalse(self.eval(0.95, {"gt": 0.60, "lt": 0.90}))

    def test_unknown_operator_skipped(self):
        """Operador desconhecido deve ser ignorado (retorna True)."""
        self.assertTrue(self.eval(0.5, {"unknown_op": 1.0}))

    def test_type_error_returns_false(self):
        """Erro de tipo na comparação → False (sem crash)."""
        self.assertFalse(self.eval("text", {"gt": 0.5}))


# ===========================================================================
# 2. ActionDispatcher
# ===========================================================================

class TestActionDispatcher(unittest.TestCase):
    """Testa o ActionDispatcher com dry_run e canais mockados."""

    def _make_decision(self, churn_risk: float = 0.85, mrr: float = 8000.0):
        from revenue_automation.policy.engine import PolicyEngine, _builtin_rules
        from revenue_automation.policy.engine import PolicyEngine
        engine = PolicyEngine.__new__(PolicyEngine)
        engine.rules_path = Path("/nonexistent")
        engine.rules = _builtin_rules()
        ctx = _make_context(churn_risk=churn_risk, mrr=mrr)
        decision = engine.evaluate(ctx)
        return decision, ctx

    def test_dry_run_always_succeeds(self):
        """Em dry_run=True, todas as dispatches devem retornar success=True."""
        from revenue_automation.dispatch.dispatcher import ActionDispatcher
        dispatcher = ActionDispatcher(dry_run=True)
        decision, ctx = self._make_decision()
        record = dispatcher.dispatch(decision, ctx)
        # Todos os resultados devem ter success=True
        for result in record.dispatch_results:
            self.assertTrue(result["success"], f"Resultado falhou: {result}")

    def test_record_has_correlation_id(self):
        """InterventionRecord deve ter correlation_id não vazio."""
        from revenue_automation.dispatch.dispatcher import ActionDispatcher
        dispatcher = ActionDispatcher(dry_run=True)
        decision, ctx = self._make_decision()
        record = dispatcher.dispatch(decision, ctx)
        self.assertTrue(len(record.correlation_id) > 0)

    def test_record_actions_taken_not_empty(self):
        """actions_taken deve conter pelo menos uma ação."""
        from revenue_automation.dispatch.dispatcher import ActionDispatcher
        dispatcher = ActionDispatcher(dry_run=True)
        decision, ctx = self._make_decision()
        record = dispatcher.dispatch(decision, ctx)
        self.assertGreater(len(record.actions_taken), 0)

    def test_console_channel_always_present(self):
        """Canal CONSOLE deve aparecer entre os canais usados."""
        from revenue_automation.dispatch.dispatcher import ActionDispatcher
        dispatcher = ActionDispatcher(dry_run=True)
        decision, ctx = self._make_decision()
        record = dispatcher.dispatch(decision, ctx)
        used = [r["channel"] for r in record.dispatch_results]
        self.assertIn("console", used)

    def test_channel_failure_does_not_abort(self):
        """Falha em um canal não deve impedir outros canais de serem tentados."""
        from revenue_automation.dispatch.dispatcher import ActionDispatcher
        from revenue_automation.dispatch.channels import slack as slack_ch

        dispatcher = ActionDispatcher(dry_run=False)
        decision, ctx = self._make_decision(churn_risk=0.65, mrr=2000.0)

        # Força falha em todos os canais exceto console
        with patch.object(slack_ch, "send", side_effect=RuntimeError("Slack down")):
            record = dispatcher.dispatch(decision, ctx)

        # Deve ainda ter resultados (console garante pelo menos um)
        self.assertGreater(len(record.dispatch_results), 0)

    def test_record_risk_level(self):
        """risk_level deve refletir o contexto da conta."""
        from revenue_automation.dispatch.dispatcher import ActionDispatcher
        dispatcher = ActionDispatcher(dry_run=True)
        decision, ctx = self._make_decision(churn_risk=0.85)
        record = dispatcher.dispatch(decision, ctx)
        self.assertEqual(record.risk_level, "HIGH")

    def test_reasons_deduped(self):
        """reasons deve ser lista sem duplicatas."""
        from revenue_automation.dispatch.dispatcher import ActionDispatcher
        dispatcher = ActionDispatcher(dry_run=True)
        decision, ctx = self._make_decision()
        record = dispatcher.dispatch(decision, ctx)
        self.assertEqual(len(record.reasons), len(set(record.reasons)))


# ===========================================================================
# 3. FeedbackStore
# ===========================================================================

class TestFeedbackStore(unittest.TestCase):
    """Testa FeedbackStore com banco SQLite em memória (arquivo temporário)."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.tmp.close()

    def tearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()

    def _store(self):
        from revenue_automation.feedback.store import FeedbackStore
        return FeedbackStore(db_path=self.db_path)

    def test_record_returns_entry(self):
        """record() deve retornar FeedbackEntry com campos corretos."""
        from revenue_automation.schemas.models import FeedbackOutcome
        store = self._store()
        entry = store.record(
            correlation_id = "corr_001",
            account_id     = "ACC_001",
            outcome        = FeedbackOutcome.RENEWAL_HAPPENED,
            notes          = "Cliente renovou",
        )
        self.assertEqual(entry.correlation_id, "corr_001")
        self.assertEqual(entry.account_id, "ACC_001")
        self.assertEqual(entry.outcome, "renewal_happened")
        self.assertTrue(len(entry.feedback_id) > 0)

    def test_total_increments(self):
        """total() deve aumentar a cada record()."""
        from revenue_automation.schemas.models import FeedbackOutcome
        store = self._store()
        self.assertEqual(store.total(), 0)
        store.record("corr_001", "ACC_001", FeedbackOutcome.MEETING_BOOKED)
        self.assertEqual(store.total(), 1)
        store.record("corr_002", "ACC_002", FeedbackOutcome.NO_RESPONSE)
        self.assertEqual(store.total(), 2)

    def test_for_account_filters_correctly(self):
        """for_account() deve retornar apenas entradas da conta solicitada."""
        from revenue_automation.schemas.models import FeedbackOutcome
        store = self._store()
        store.record("c1", "ACC_001", FeedbackOutcome.RENEWAL_HAPPENED)
        store.record("c2", "ACC_001", FeedbackOutcome.RISK_DECREASED)
        store.record("c3", "ACC_002", FeedbackOutcome.NO_RESPONSE)
        results = store.for_account("ACC_001")
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r.account_id, "ACC_001")

    def test_for_correlation(self):
        """for_correlation() deve retornar entradas pelo correlation_id."""
        from revenue_automation.schemas.models import FeedbackOutcome
        store = self._store()
        store.record("corr_abc", "ACC_001", FeedbackOutcome.MEETING_BOOKED)
        store.record("corr_def", "ACC_001", FeedbackOutcome.RENEWAL_HAPPENED)
        results = store.for_correlation("corr_abc")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].outcome, "meeting_booked")

    def test_outcome_counts(self):
        """outcome_counts() deve retornar contagem correta por outcome."""
        from revenue_automation.schemas.models import FeedbackOutcome
        store = self._store()
        store.record("c1", "ACC_001", FeedbackOutcome.RENEWAL_HAPPENED)
        store.record("c2", "ACC_001", FeedbackOutcome.RENEWAL_HAPPENED)
        store.record("c3", "ACC_002", FeedbackOutcome.NO_RESPONSE)
        counts = store.outcome_counts(since_days=30)
        self.assertEqual(counts.get("renewal_happened", 0), 2)
        self.assertEqual(counts.get("no_response", 0), 1)

    def test_record_with_risk_scores(self):
        """record() com churn_risk_before/after deve persistir corretamente."""
        from revenue_automation.schemas.models import FeedbackOutcome
        store = self._store()
        entry = store.record(
            correlation_id    = "c1",
            account_id        = "ACC_001",
            outcome           = FeedbackOutcome.RISK_DECREASED,
            churn_risk_before = 0.85,
            churn_risk_after  = 0.20,
        )
        self.assertAlmostEqual(entry.churn_risk_before, 0.85, places=4)
        self.assertAlmostEqual(entry.churn_risk_after,  0.20, places=4)

    def test_recent_limit(self):
        """recent(limit=N) deve retornar no máximo N entradas."""
        from revenue_automation.schemas.models import FeedbackOutcome
        store = self._store()
        for i in range(10):
            store.record(f"c{i}", f"ACC_{i:03d}", FeedbackOutcome.NO_RESPONSE)
        results = store.recent(limit=5)
        self.assertLessEqual(len(results), 5)

    def test_feedback_entry_to_jsonl(self):
        """FeedbackEntry.to_jsonl() deve retornar JSON válido."""
        import json
        from revenue_automation.schemas.models import FeedbackOutcome
        store = self._store()
        entry = store.record("c1", "ACC_001", FeedbackOutcome.RENEWAL_HAPPENED)
        raw = entry.to_jsonl()
        parsed = json.loads(raw)
        self.assertEqual(parsed["account_id"], "ACC_001")
        self.assertEqual(parsed["outcome"], "renewal_happened")


# ===========================================================================
# 4. ReportBuilder
# ===========================================================================

class TestReportBuilder(unittest.TestCase):
    """Testa ReportBuilder com dados sintéticos."""

    def _make_records(self, n: int = 5):
        from revenue_automation.schemas.models import InterventionRecord
        records = []
        for i in range(n):
            risk = 0.9 if i < 2 else (0.5 if i < 4 else 0.2)
            records.append(InterventionRecord(
                correlation_id  = f"corr_{i:03d}",
                account_id      = f"ACC_{i:03d}",
                session_id      = f"sess_{i:03d}",
                timestamp       = datetime.now(timezone.utc).isoformat(),
                previous_score  = None,
                current_score   = None,
                churn_risk      = risk,
                risk_level      = "HIGH" if risk >= 0.60 else ("MEDIUM" if risk >= 0.35 else "LOW"),
                actions_taken   = ["SEND_EMAIL", "CREATE_CRM_TASK"] if risk > 0.5 else ["MONITOR_ONLY"],
                channels_used   = ["console", "file"],
                reasons         = ["Test reason"],
                dispatch_results= [],
                payload_summary = {},
            ))
        return records

    def _make_contexts(self, n: int = 5):
        ctxs = []
        for i in range(n):
            risk = 0.9 if i < 2 else (0.5 if i < 4 else 0.2)
            ctxs.append(_make_context(
                account_id  = f"ACC_{i:03d}",
                churn_risk  = risk,
                mrr         = 1000.0 * (i + 1),
                segment     = "ENTERPRISE" if i < 2 else "MID_MARKET",
            ))
        return ctxs

    def _build(self):
        from revenue_automation.reports.builder import ReportBuilder
        from revenue_automation.schemas.models import ReportPeriod

        records = self._make_records(5)
        ctxs    = self._make_contexts(5)
        builder = ReportBuilder(
            period          = ReportPeriod.WEEKLY,
            period_start    = "2025-01-01",
            period_end      = "2025-01-07",
            contexts        = ctxs,
            records         = records,
            feedback_counts = {"renewal_happened": 2, "no_response": 1},
        )
        return builder.build()

    def test_total_accounts(self):
        """total_accounts deve refletir o número de records."""
        report = self._build()
        self.assertEqual(report.total_accounts, 5)

    def test_accounts_at_risk(self):
        """accounts_at_risk deve contar apenas HIGH risk."""
        report = self._build()
        self.assertEqual(report.accounts_at_risk, 2)

    def test_recoverable_accounts(self):
        """recoverable_accounts deve contar MEDIUM risk."""
        report = self._build()
        self.assertEqual(report.recoverable_accounts, 2)

    def test_safe_accounts(self):
        """safe_accounts deve contar LOW risk."""
        report = self._build()
        self.assertEqual(report.safe_accounts, 1)

    def test_total_mrr_at_risk_positive(self):
        """total_mrr_at_risk deve ser positivo quando há contas em risco."""
        report = self._build()
        self.assertGreater(report.total_mrr_at_risk, 0.0)

    def test_feedback_outcomes_present(self):
        """feedback_outcomes deve refletir os dados passados."""
        report = self._build()
        self.assertEqual(report.feedback_outcomes.get("renewal_happened"), 2)
        self.assertEqual(report.feedback_outcomes.get("no_response"), 1)

    def test_executive_summary_not_empty(self):
        """executive_summary deve ser string não vazia."""
        report = self._build()
        self.assertIsInstance(report.executive_summary, str)
        self.assertGreater(len(report.executive_summary), 0)

    def test_segment_breakdown_keys(self):
        """segment_breakdown deve conter as chaves de segmentos usados."""
        report = self._build()
        self.assertIn("ENTERPRISE", report.segment_breakdown)
        self.assertIn("MID_MARKET", report.segment_breakdown)

    def test_report_as_dict(self):
        """as_dict() deve retornar um dicionário com os campos esperados."""
        report = self._build()
        d = report.as_dict()
        self.assertIn("total_accounts", d)
        self.assertIn("accounts_at_risk", d)
        self.assertIn("executive_summary", d)


# ===========================================================================
# 5. Renderer
# ===========================================================================

class TestReportRenderer(unittest.TestCase):
    """Testa renderização de relatórios em Markdown, JSON e HTML."""

    def _report(self):
        from revenue_automation.reports.builder import ReportBuilder
        from revenue_automation.schemas.models import InterventionRecord, ReportPeriod

        records = [
            InterventionRecord(
                correlation_id  = "c001",
                account_id      = "ACC_001",
                session_id      = "s001",
                timestamp       = datetime.now(timezone.utc).isoformat(),
                previous_score  = None,
                current_score   = None,
                churn_risk      = 0.85,
                risk_level      = "HIGH",
                actions_taken   = ["ESCALATE_TO_MANAGER"],
                channels_used   = ["console"],
                reasons         = ["Test"],
                dispatch_results= [],
                payload_summary = {},
            )
        ]
        ctxs = [_make_context(churn_risk=0.85)]
        builder = ReportBuilder(
            period       = ReportPeriod.WEEKLY,
            period_start = "2025-01-01",
            period_end   = "2025-01-07",
            contexts     = ctxs,
            records      = records,
        )
        return builder.build()

    def test_render_markdown(self):
        """render_markdown() deve retornar string com cabeçalho markdown."""
        from revenue_automation.reports.renderer import render_markdown
        report = self._report()
        md = render_markdown(report)
        self.assertIn("#", md)
        self.assertIn("ENTERPRISE", md)

    def test_render_json_valid(self):
        """render_json() deve retornar JSON válido."""
        import json
        from revenue_automation.reports.renderer import render_json
        report = self._report()
        raw = render_json(report)
        parsed = json.loads(raw)
        self.assertIn("total_accounts", parsed)

    def test_render_html(self):
        """render_html() deve retornar HTML com tags básicas."""
        from revenue_automation.reports.renderer import render_html
        report = self._report()
        html = render_html(report)
        self.assertIn("<html", html.lower())
        self.assertIn("</html>", html.lower())


# ===========================================================================
# 6. InterventionEngine (com stubs)
# ===========================================================================

class _StubTurn:
    """PredictionTurn stub — sem dependência de modelos ML."""
    def __init__(self, account_id: str, churn_risk: float = 0.75):
        self.account_id        = account_id
        self.session_id        = f"stub_sess_{account_id}"
        self.churn_risk        = churn_risk
        self.retention_prob    = 1.0 - churn_risk
        self.risk_level        = "HIGH" if churn_risk >= 0.60 else "LOW"
        self.upsell_probability= 0.10
        self.upsell_signal     = False
        self.recommended_actions = []
        self.memory_hits       = 0
        self.memory_saved      = False
        self.mrr               = 2000.0
        self.mrr_at_risk       = 2000.0 * churn_risk


class _StubQueryEngine:
    """ChurnQueryEngine stub — retorna _StubTurn sem tocar em arquivos/modelos."""
    def analyze(self, account_id: str, features: dict, session_id=None) -> _StubTurn:
        return _StubTurn(account_id, churn_risk=float(features.get("churn_risk", 0.75)))


class _StubAuditTrail:
    def __init__(self):
        self.events = []

    def _add(self, event_type, account_id, session_id, details):
        self.events.append({
            "event_type": event_type,
            "account_id": account_id,
        })


class TestInterventionEngine(unittest.TestCase):
    """Testa InterventionEngine com componentes stub."""

    def _engine(self):
        from revenue_automation.engine import InterventionEngine
        from revenue_automation.policy.engine import PolicyEngine, _builtin_rules
        from revenue_automation.dispatch.dispatcher import ActionDispatcher

        pe = PolicyEngine.__new__(PolicyEngine)
        pe.rules_path = Path("/nonexistent")
        pe.rules = _builtin_rules()

        import pandas as pd
        return InterventionEngine(
            query_engine  = _StubQueryEngine(),
            policy_engine = pe,
            dispatcher    = ActionDispatcher(dry_run=True),
            audit_trail   = _StubAuditTrail(),
            df_accounts   = pd.DataFrame(),
        )

    def test_intervene_one_returns_record(self):
        """intervene_one() deve retornar InterventionRecord válido."""
        from revenue_automation.schemas.models import InterventionRecord
        engine = self._engine()
        features = {"churn_risk": 0.85, "mrr": 8000.0, "segment": "ENTERPRISE"}
        record = engine.intervene_one("ACC_001", features)
        self.assertIsInstance(record, InterventionRecord)
        self.assertEqual(record.account_id, "ACC_001")

    def test_intervene_one_audit_logged(self):
        """intervene_one() deve registrar evento no audit trail."""
        engine = self._engine()
        features = {"churn_risk": 0.85, "mrr": 8000.0}
        engine.intervene_one("ACC_001", features)
        self.assertEqual(len(engine.audit_trail.events), 1)
        self.assertEqual(engine.audit_trail.events[0]["event_type"], "intervention")

    def test_intervene_batch_empty_df(self):
        """intervene_batch() com DataFrame vazio deve retornar lista vazia."""
        import pandas as pd
        engine = self._engine()
        records = engine.intervene_batch(df=pd.DataFrame())
        self.assertEqual(records, [])

    def test_intervene_batch_processes_all_rows(self):
        """intervene_batch() deve processar todas as linhas do DataFrame."""
        import pandas as pd
        engine = self._engine()
        df = pd.DataFrame([
            {"account_id": "ACC_001", "churn_risk": 0.85, "mrr": 8000.0},
            {"account_id": "ACC_002", "churn_risk": 0.30, "mrr": 1000.0},
            {"account_id": "ACC_003", "churn_risk": 0.65, "mrr": 3000.0},
        ])
        records = engine.intervene_batch(df=df)
        self.assertEqual(len(records), 3)

    def test_intervene_batch_skips_errors(self):
        """Erros em contas individuais não devem abortar o batch."""
        import pandas as pd
        engine = self._engine()

        # Substitui query_engine para falhar em uma conta específica
        class _PartialFailEngine:
            def analyze(self, account_id, features, session_id=None):
                if account_id == "ACC_FAIL":
                    raise ValueError("Erro simulado")
                return _StubTurn(account_id, churn_risk=0.5)

        engine.query_engine = _PartialFailEngine()
        df = pd.DataFrame([
            {"account_id": "ACC_001", "churn_risk": 0.5},
            {"account_id": "ACC_FAIL", "churn_risk": 0.9},
            {"account_id": "ACC_003", "churn_risk": 0.3},
        ])
        records = engine.intervene_batch(df=df)
        self.assertEqual(len(records), 2)  # apenas 2 sucessos

    def test_batch_summary_risk_distribution(self):
        """batch_summary() deve retornar risk_distribution com HIGH/MEDIUM/LOW."""
        import pandas as pd
        engine = self._engine()
        df = pd.DataFrame([
            {"account_id": "ACC_001", "churn_risk": 0.85},
            {"account_id": "ACC_002", "churn_risk": 0.50},
            {"account_id": "ACC_003", "churn_risk": 0.10},
        ])
        records = engine.intervene_batch(df=df)
        summary = engine.batch_summary(records)
        self.assertIn("risk_distribution", summary)
        self.assertIn("total_interventions", summary)
        self.assertEqual(summary["total_interventions"], 3)


# ===========================================================================
# 7. Schemas
# ===========================================================================

class TestSchemas(unittest.TestCase):
    """Testa campos e métodos dos tipos centrais."""

    def test_intervention_context_from_prediction_turn(self):
        """from_prediction_turn() deve construir InterventionContext corretamente."""
        from revenue_automation.schemas.models import InterventionContext
        turn = _StubTurn("ACC_001", churn_risk=0.70)
        meta = {"segment": "MID_MARKET", "dias_sem_interacao": 15}
        ctx = InterventionContext.from_prediction_turn(turn, df_row=meta)
        self.assertEqual(ctx.account_id, "ACC_001")
        self.assertAlmostEqual(ctx.churn_risk, 0.70, places=4)
        self.assertEqual(ctx.segment, "MID_MARKET")
        self.assertEqual(ctx.dias_sem_interacao, 15)

    def test_intervention_record_to_jsonl(self):
        """InterventionRecord.to_jsonl() deve retornar JSON válido."""
        import json
        from revenue_automation.schemas.models import InterventionRecord
        record = InterventionRecord(
            correlation_id  = "ctest",
            account_id      = "ACC_001",
            session_id      = "sess",
            timestamp       = datetime.now(timezone.utc).isoformat(),
            previous_score  = None,
            current_score   = None,
            churn_risk      = 0.85,
            risk_level      = "HIGH",
            actions_taken   = ["SEND_EMAIL"],
            channels_used   = ["console"],
            reasons         = ["Test"],
            dispatch_results= [],
            payload_summary = {},
        )
        raw = record.to_jsonl()
        parsed = json.loads(raw)
        self.assertEqual(parsed["account_id"], "ACC_001")
        self.assertEqual(parsed["risk_level"], "HIGH")

    def test_feedback_outcome_enum_values(self):
        """FeedbackOutcome deve ter os valores esperados."""
        from revenue_automation.schemas.models import FeedbackOutcome
        expected = {"renewal_happened", "upsell_happened", "meeting_booked",
                    "risk_decreased", "customer_replied", "ticket_created",
                    "no_response", "unknown"}
        actual = {o.value for o in FeedbackOutcome}
        self.assertTrue(expected.issubset(actual))

    def test_action_code_enum_values(self):
        """ActionCode deve ter todos os 11 valores definidos."""
        from revenue_automation.schemas.models import ActionCode
        self.assertGreaterEqual(len(list(ActionCode)), 11)

    def test_dispatch_channel_enum_values(self):
        """DispatchChannel deve incluir console, slack, email, file, api_hook."""
        from revenue_automation.schemas.models import DispatchChannel
        values = {c.value for c in DispatchChannel}
        for expected in ["console", "slack", "email", "file", "api_hook"]:
            self.assertIn(expected, values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
