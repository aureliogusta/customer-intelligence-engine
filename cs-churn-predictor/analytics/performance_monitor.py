"""
analytics/performance_monitor.py
==================================
Rastreia performance do modelo em produção ao longo do tempo.

Inspirado no CostTracker do claw-code (custo de tokens por turno),
mas adaptado para métricas de ML e negócio:
  - Precision/Recall se labels reais disponíveis
  - MRR em risco por ciclo de predição
  - Contagem de alertas por nível de risco
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class BatchMetric:
    timestamp:          str
    session_id:         str
    account_count:      int
    avg_churn_risk:     float
    high_risk_count:    int
    medium_risk_count:  int
    low_risk_count:     int
    total_mrr_at_risk:  float
    # Preenchidos se labels reais estiverem disponíveis
    accuracy:           float | None = None
    precision:          float | None = None
    recall:             float | None = None
    f1:                 float | None = None


@dataclass
class PerformanceMonitor:
    """
    Acumula métricas de cada rodada de predição.

    Uso:
        monitor = PerformanceMonitor(log_path="data/perf_log.jsonl")
        metric  = monitor.record_batch(turns)
        print(monitor.trend_summary())
    """

    metrics:  List[BatchMetric] = field(default_factory=list)
    log_path: Path | None       = None

    def __post_init__(self) -> None:
        if self.log_path:
            self.log_path = Path(self.log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Record ────────────────────────────────────────────────────────────────

    def record_batch(
        self,
        turns:         List[Any],        # List[PredictionTurn]
        session_id:    str = "",
        actual_labels: pd.Series | None = None,   # pd.Series de bool (True=churnou)
    ) -> BatchMetric:
        """
        Registra métricas de um batch de PredictionTurn.
        Se actual_labels fornecido, calcula precision/recall reais.
        """
        risks       = [t.churn_risk for t in turns]
        levels      = [t.risk_level for t in turns]
        mrr_at_risk = [t.mrr_at_risk for t in turns]

        metric = BatchMetric(
            timestamp         = datetime.now(timezone.utc).isoformat(),
            session_id        = session_id,
            account_count     = len(turns),
            avg_churn_risk    = round(float(np.mean(risks)), 4) if risks else 0.0,
            high_risk_count   = levels.count("HIGH"),
            medium_risk_count = levels.count("MEDIUM"),
            low_risk_count    = levels.count("LOW"),
            total_mrr_at_risk = round(sum(mrr_at_risk), 2),
        )

        if actual_labels is not None and len(actual_labels) == len(turns):
            metric = self._add_classification_metrics(metric, turns, actual_labels)

        self.metrics.append(metric)

        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(metric), ensure_ascii=False) + "\n")

        return metric

    def record_single(self, turn: Any) -> None:
        """Registra uma predição individual (acumula em batch implícito)."""
        self.record_batch([turn], session_id=turn.session_id)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _add_classification_metrics(
        self,
        metric:        BatchMetric,
        turns:         List[Any],
        actual_labels: pd.Series,
    ) -> BatchMetric:
        try:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            y_pred = [t.risk_level in ("HIGH", "MEDIUM") for t in turns]
            y_true = actual_labels.astype(bool).tolist()
            metric.accuracy  = round(float(accuracy_score(y_true, y_pred)), 4)
            metric.precision = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
            metric.recall    = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
            metric.f1        = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        except Exception:
            pass
        return metric

    # ── Summaries ─────────────────────────────────────────────────────────────

    def trend_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Resumo das últimas N rodadas."""
        subset = self.metrics[-last_n:]
        if not subset:
            return {}

        return {
            "rodadas":            len(subset),
            "total_contas":       sum(m.account_count for m in subset),
            "avg_churn_risk":     round(float(np.mean([m.avg_churn_risk for m in subset])), 4),
            "high_risk_trend":    [m.high_risk_count for m in subset],
            "mrr_at_risk_trend":  [m.total_mrr_at_risk for m in subset],
            "avg_precision":      self._avg_or_none([m.precision for m in subset]),
            "avg_recall":         self._avg_or_none([m.recall for m in subset]),
        }

    def latest(self) -> BatchMetric | None:
        return self.metrics[-1] if self.metrics else None

    def should_alert(self, mrr_threshold: float = 50_000.0) -> bool:
        """Dispara alerta se MRR em risco do último batch excede threshold."""
        m = self.latest()
        return m is not None and m.total_mrr_at_risk >= mrr_threshold

    def _avg_or_none(self, values: List[float | None]) -> float | None:
        clean = [v for v in values if v is not None]
        return round(float(np.mean(clean)), 4) if clean else None
