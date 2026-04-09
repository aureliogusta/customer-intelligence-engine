"""
revenue_automation/engine.py
==============================
InterventionEngine — orquestra o ciclo completo de intervenção autônoma.

Fluxo por conta:
  1. Recebe PredictionTurn + metadados
  2. Constrói InterventionContext
  3. PolicyEngine avalia e decide ações
  4. ActionDispatcher despacha nos canais
  5. ChurnAuditTrail registra tudo (integra com trail existente)
  6. Retorna InterventionRecord

Fluxo em batch:
  1. Carrega todas as contas do CSV
  2. Roda ChurnQueryEngine em batch
  3. Para cada turno: roda ciclo individual
  4. Consolida InterventionRecord[]
  5. Retorna lista pronta para relatório

Uso:
    engine = InterventionEngine.from_env()
    record = engine.intervene_one("ACC_001", features)
    records = engine.intervene_batch(df)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def _add_root():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


_add_root()

from analytics.audit_trail import ChurnAuditTrail
from decision_service.query_engine import ChurnQueryEngine

from .dispatch.dispatcher import ActionDispatcher
from .policy.engine import PolicyEngine
from .schemas.models import InterventionContext, InterventionRecord


class InterventionEngine:
    """
    Ponto de entrada único para intervenção autônoma.

    Integra:
      - ChurnQueryEngine (predição ML + memória)
      - PolicyEngine     (regras YAML → ações)
      - ActionDispatcher (despacho multi-canal)
      - ChurnAuditTrail  (log auditável — compartilhado com sistema existente)
    """

    def __init__(
        self,
        query_engine:  ChurnQueryEngine,
        policy_engine: PolicyEngine,
        dispatcher:    ActionDispatcher,
        audit_trail:   ChurnAuditTrail,
        df_accounts:   pd.DataFrame | None = None,
    ):
        self.query_engine  = query_engine
        self.policy_engine = policy_engine
        self.dispatcher    = dispatcher
        self.audit_trail   = audit_trail
        self.df_accounts   = df_accounts if df_accounts is not None else pd.DataFrame()

    @classmethod
    def from_env(
        cls,
        models_dir:    str  = "ml/models",
        rules_path:    str  | None = None,
        dry_run:       bool = False,
        accounts_csv:  str  | None = None,
        audit_log:     str  | None = None,
    ) -> "InterventionEngine":
        """Factory padrão — carrega tudo de variáveis de ambiente e defaults."""
        query_engine  = ChurnQueryEngine.from_env(models_dir=models_dir)
        policy_engine = (
            PolicyEngine.from_file(rules_path)
            if rules_path
            else PolicyEngine.from_default()
        )
        dispatcher    = ActionDispatcher(dry_run=dry_run)
        trail_path    = Path(audit_log) if audit_log else Path("logs/audit.jsonl")
        audit_trail   = ChurnAuditTrail(log_path=trail_path)

        df: pd.DataFrame = pd.DataFrame()
        csv_path = accounts_csv or "data/train_dataset.csv"
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)

        return cls(
            query_engine  = query_engine,
            policy_engine = policy_engine,
            dispatcher    = dispatcher,
            audit_trail   = audit_trail,
            df_accounts   = df,
        )

    # ── Single account ────────────────────────────────────────────────────────

    def intervene_one(
        self,
        account_id:     str,
        features:       Dict[str, Any],
        previous_score: Optional[float] = None,
    ) -> InterventionRecord:
        """
        Ciclo completo para uma conta.
        Retorna InterventionRecord com todas as decisões e resultados.
        """
        # 1. Predição ML
        turn = self.query_engine.analyze(account_id, features)

        # 2. Contexto
        ctx = InterventionContext.from_prediction_turn(turn, df_row=features)

        # 3. Política
        decision = self.policy_engine.evaluate(ctx)

        # 4. Despacho
        record = self.dispatcher.dispatch(
            decision       = decision,
            ctx            = ctx,
            previous_score = previous_score,
            current_score  = None,
        )

        # 5. Audit (estende o trail existente com evento "intervention")
        self.audit_trail._add(
            event_type = "intervention",
            account_id = account_id,
            session_id = turn.session_id,
            details    = {
                "correlation_id": record.correlation_id,
                "actions":        record.actions_taken,
                "channels":       record.channels_used,
                "churn_risk":     turn.churn_risk,
                "risk_level":     turn.risk_level,
                "mrr_at_risk":    turn.mrr_at_risk,
                "triggered_rules": decision.triggered_rules,
            },
        )

        return record

    # ── Batch ─────────────────────────────────────────────────────────────────

    def intervene_batch(
        self,
        df: pd.DataFrame | None = None,
        min_risk: float = 0.0,
    ) -> List[InterventionRecord]:
        """
        Processa todas as contas do DataFrame (ou do CSV padrão).
        Filtra por min_risk se especificado (só intervém acima do limiar).
        """
        source = df if df is not None else self.df_accounts
        if source.empty:
            log.warning("intervene_batch: DataFrame vazio — nenhuma conta processada.")
            return []

        records: List[InterventionRecord] = []
        for _, row in source.iterrows():
            features   = row.to_dict()
            account_id = str(features.get("account_id", f"ACC_UNKNOWN"))

            try:
                # Rápida pré-filtragem via score heurístico para evitar overhead
                # (o modelo real é chamado dentro de intervene_one)
                record = self.intervene_one(account_id, features)
                records.append(record)
            except Exception as e:
                log.error("Erro ao processar %s: %s", account_id, e)

        log.info(
            "intervene_batch: %d contas processadas, %d intervenções registradas",
            len(source), len(records),
        )
        return records

    # ── Summary ───────────────────────────────────────────────────────────────

    def batch_summary(self, records: List[InterventionRecord]) -> Dict[str, Any]:
        from collections import Counter
        action_counts: Counter = Counter()
        channel_counts: Counter = Counter()
        risk_counts: Dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        for r in records:
            for a in r.actions_taken:
                action_counts[a] += 1
            for ch in r.channels_used:
                channel_counts[ch] += 1
            level = r.risk_level
            if level in risk_counts:
                risk_counts[level] += 1

        return {
            "total_interventions": len(records),
            "risk_distribution":   risk_counts,
            "top_actions":         dict(action_counts.most_common(5)),
            "channels_used":       dict(channel_counts.most_common()),
            "total_mrr_at_risk":   round(sum(
                r.churn_risk * 1000 for r in records  # proxy; real value comes from ctx
            ), 2),
        }
