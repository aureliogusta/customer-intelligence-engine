"""
decision_service/query_engine.py
==================================
ChurnQueryEngine — camada de orquestração central.

Inspirado em claw-code-main/src/query_engine.py (QueryEnginePort):
  - Session ID por análise
  - Recupera memória histórica antes de recomendar
  - Auto-salva predições no VectorStore
  - Transcript de eventos
  - TurnResult estruturado

Uso:
    engine = ChurnQueryEngine.from_env()
    result = engine.analyze("ACC_000042", features_dict)

    print(result.churn_risk)
    print(result.recommended_actions)
    print(result.memory_hits)     # contexto histórico recuperado
    print(result.session_id)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from .inference import ChurnPredictor, ExpansionPredictor
from .memory import AccountMemoryStore, MemoryMatch, _NoopMemoryStore, get_memory_store


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChurnEngineConfig:
    memory_limit:      int   = 5      # quantas memórias recuperar por consulta
    high_risk_threshold: float = 0.70
    medium_risk_threshold: float = 0.40
    save_predictions:  bool  = True   # salvar predições no VectorStore
    max_actions:       int   = 5      # limitar recomendações


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PredictionTurn:
    session_id:          str
    account_id:          str
    timestamp:           str

    churn_risk:          float
    retention_prob:      float
    risk_level:          str

    upsell_probability:  float | None
    upsell_signal:       str   | None

    recommended_actions: tuple[dict, ...]
    memory_hits:         tuple[MemoryMatch, ...]   # histórico recuperado
    memory_saved:        bool

    mrr:                 float
    mrr_at_risk:         float

    stop_reason:         str   = "completed"


# ── Engine ────────────────────────────────────────────────────────────────────

@dataclass
class ChurnQueryEngine:
    """
    Orquestra predição de churn + expansão + recomendações + memória semântica.

    Equivalente ao QueryEnginePort do claw-code: cada chamada a analyze()
    é um "turno" com sessão própria, recuperação de contexto e auto-save.
    """

    churn_predictor:     ChurnPredictor
    expansion_predictor: ExpansionPredictor | None = None
    memory_store:        Any = field(default_factory=get_memory_store)
    config:              ChurnEngineConfig = field(default_factory=ChurnEngineConfig)

    @classmethod
    def from_env(cls, models_dir: str = "ml/models") -> "ChurnQueryEngine":
        """
        Factory padrão: carrega modelos do disco e conecta ao pgvector se disponível.
        Cai graciosamente se modelos ou DB não existirem.
        """
        from pathlib import Path

        mdir = Path(models_dir)

        churn_pred = ChurnPredictor(
            model_path   = mdir / "churn_model.pkl",
            scaler_path  = mdir / "churn_scaler.pkl",
            encoder_path = mdir / "churn_encoder.pkl",
        )

        exp_pred: ExpansionPredictor | None = None
        if (mdir / "expansion_model.pkl").exists():
            exp_pred = ExpansionPredictor(
                model_path   = mdir / "expansion_model.pkl",
                scaler_path  = mdir / "expansion_scaler.pkl",
                encoder_path = mdir / "churn_encoder.pkl",
            )

        return cls(
            churn_predictor     = churn_pred,
            expansion_predictor = exp_pred,
            memory_store        = get_memory_store(),
        )

    # ── Core turn ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        account_id:   str,
        features:     Dict[str, Any],
        session_id:   str | None = None,
    ) -> PredictionTurn:
        """
        Executa um turno completo de análise:
        1. Recupera memória histórica da conta (context window semântico)
        2. Prediz churn + expansão
        3. Gera recomendações
        4. Salva resultado no VectorStore
        5. Retorna PredictionTurn estruturado
        """
        session_id = session_id or uuid4().hex
        timestamp  = datetime.now(timezone.utc).isoformat()

        # ── 1. Recuperar histórico (equivale ao retrieve_memory do QueryEnginePort) ──
        query       = f"conta {account_id} churn risco engajamento"
        memory_hits = self.memory_store.recall(query=query, account_id=account_id, limit=self.config.memory_limit)

        # ── 2. Predições ML ───────────────────────────────────────────────────
        churn_result = self.churn_predictor.predict(features)
        churn_risk   = churn_result["churn_risk"]
        risk_level   = churn_result["risk_level"]

        upsell_prob:   float | None = None
        upsell_signal: str   | None = None
        if self.expansion_predictor:
            exp_result    = self.expansion_predictor.predict(features)
            upsell_prob   = exp_result["upsell_probability"]
            upsell_signal = exp_result["upsell_signal"]

        # ── 3. Recomendações ──────────────────────────────────────────────────
        from study_service.recommendations import gerar_recomendacoes
        mrr     = float(features.get("mrr", 0))
        actions = gerar_recomendacoes(
            account_id         = account_id,
            churn_risk         = churn_risk,
            upsell_probability = upsell_prob or 0.0,
            mrr                = mrr,
        )[: self.config.max_actions]

        # ── 4. Auto-save (equivale ao auto_save_from_prompt do MemoryTools) ───
        saved = False
        if self.config.save_predictions:
            try:
                self.memory_store.remember_prediction(
                    account_id = account_id,
                    session_id = session_id,
                    churn_risk = churn_risk,
                    risk_level = risk_level,
                    actions    = [a["code"] for a in actions],
                    extra      = {
                        "mrr":             mrr,
                        "engajamento_pct": features.get("engajamento_pct"),
                        "nps_score":       features.get("nps_score"),
                    },
                )
                saved = True
            except Exception:
                pass  # Memory é opcional — não quebra o fluxo

        return PredictionTurn(
            session_id          = session_id,
            account_id          = account_id,
            timestamp           = timestamp,
            churn_risk          = churn_risk,
            retention_prob      = churn_result["retention_prob"],
            risk_level          = risk_level,
            upsell_probability  = upsell_prob,
            upsell_signal       = upsell_signal,
            recommended_actions = tuple(actions),
            memory_hits         = tuple(memory_hits),
            memory_saved        = saved,
            mrr                 = mrr,
            mrr_at_risk         = round(mrr * churn_risk, 2),
            stop_reason         = "completed",
        )

    def analyze_batch(
        self,
        accounts: List[Dict[str, Any]],
        id_column: str = "account_id",
    ) -> List[PredictionTurn]:
        """Analisa múltiplas contas, compartilhando a mesma sessão."""
        session_id = uuid4().hex
        return [
            self.analyze(
                account_id = str(row.get(id_column, f"UNKNOWN_{i}")),
                features   = row,
                session_id = session_id,
            )
            for i, row in enumerate(accounts)
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def risk_summary(self, turns: List[PredictionTurn]) -> Dict[str, Any]:
        """Agrega estatísticas de um batch de predições."""
        if not turns:
            return {}
        import statistics
        return {
            "total":          len(turns),
            "high_risk":      sum(1 for t in turns if t.risk_level == "HIGH"),
            "medium_risk":    sum(1 for t in turns if t.risk_level == "MEDIUM"),
            "low_risk":       sum(1 for t in turns if t.risk_level == "LOW"),
            "avg_churn_risk": round(statistics.mean(t.churn_risk for t in turns), 4),
            "mrr_at_risk":    round(sum(t.mrr_at_risk for t in turns), 2),
            "memory_hits":    sum(len(t.memory_hits) for t in turns),
        }
