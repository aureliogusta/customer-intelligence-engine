"""
analytics/audit_trail.py
=========================
Registro auditável de todas as predições e ações recomendadas.

Inspirado em claw-code-main:
  - history.py  → HistoryLog (eventos com title + detail)
  - transcript.py → TranscriptStore (log compactável de mensagens)

ChurnAuditTrail persiste em JSON linha a linha (JSONL) para fácil análise.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


# ── Entry types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuditEntry:
    timestamp:   str
    session_id:  str
    account_id:  str
    event_type:  str          # "prediction" | "recommendation" | "batch" | "drift"
    details:     Dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ── Audit Trail ───────────────────────────────────────────────────────────────

@dataclass
class ChurnAuditTrail:
    """
    Log estruturado de predições e recomendações.

    Suporta:
      - Append em memória (rápido)
      - Flush para JSONL em disco (persistência)
      - Export para CSV (análise)
      - Compactação (descarta entradas antigas além de keep_last)
    """

    entries:    List[AuditEntry] = field(default_factory=list)
    log_path:   Path | None      = None
    _flushed:   bool             = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.log_path:
            self.log_path = Path(self.log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Append ────────────────────────────────────────────────────────────────

    def _add(self, event_type: str, account_id: str, session_id: str, details: Dict[str, Any]) -> None:
        entry = AuditEntry(
            timestamp  = datetime.now(timezone.utc).isoformat(),
            session_id = session_id,
            account_id = account_id,
            event_type = event_type,
            details    = details,
        )
        self.entries.append(entry)
        self._flushed = False

        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(entry.to_jsonl() + "\n")

    def log_prediction(
        self,
        account_id:   str,
        session_id:   str,
        churn_risk:   float,
        risk_level:   str,
        mrr:          float = 0.0,
        mrr_at_risk:  float = 0.0,
        memory_hits:  int   = 0,
    ) -> None:
        self._add("prediction", account_id, session_id, {
            "churn_risk":  churn_risk,
            "risk_level":  risk_level,
            "mrr":         mrr,
            "mrr_at_risk": mrr_at_risk,
            "memory_hits": memory_hits,
        })

    def log_recommendation(
        self,
        account_id: str,
        session_id: str,
        actions:    List[str],
    ) -> None:
        self._add("recommendation", account_id, session_id, {"actions": actions})

    def log_from_turn(self, turn: Any) -> None:
        """Loga diretamente de um PredictionTurn (integração com ChurnQueryEngine)."""
        self.log_prediction(
            account_id  = turn.account_id,
            session_id  = turn.session_id,
            churn_risk  = turn.churn_risk,
            risk_level  = turn.risk_level,
            mrr         = turn.mrr,
            mrr_at_risk = turn.mrr_at_risk,
            memory_hits = len(turn.memory_hits),
        )
        self.log_recommendation(
            account_id = turn.account_id,
            session_id = turn.session_id,
            actions    = [a["code"] for a in turn.recommended_actions],
        )

    def log_drift_event(
        self,
        session_id:       str,
        drifted_features: List[str],
        should_retrain:   bool,
    ) -> None:
        self._add("drift", "SYSTEM", session_id, {
            "drifted_features": drifted_features,
            "should_retrain":   should_retrain,
        })

    # ── Read ──────────────────────────────────────────────────────────────────

    def for_account(self, account_id: str) -> List[AuditEntry]:
        return [e for e in self.entries if e.account_id == account_id]

    def last_n(self, n: int) -> List[AuditEntry]:
        return self.entries[-n:]

    def predictions_only(self) -> List[AuditEntry]:
        return [e for e in self.entries if e.event_type == "prediction"]

    # ── Compact ───────────────────────────────────────────────────────────────

    def compact(self, keep_last: int = 2000) -> int:
        """Remove entradas antigas. Retorna quantas foram removidas."""
        if len(self.entries) <= keep_last:
            return 0
        removed = len(self.entries) - keep_last
        self.entries = self.entries[-keep_last:]
        return removed

    # ── Export ────────────────────────────────────────────────────────────────

    def to_csv(self) -> str:
        """Exporta para CSV (útil para análise em Excel/notebooks)."""
        buf = io.StringIO()
        w   = csv.DictWriter(buf, fieldnames=["timestamp", "session_id", "account_id", "event_type", "details"])
        w.writeheader()
        for e in self.entries:
            w.writerow({
                "timestamp":  e.timestamp,
                "session_id": e.session_id,
                "account_id": e.account_id,
                "event_type": e.event_type,
                "details":    json.dumps(e.details, ensure_ascii=False),
            })
        return buf.getvalue()

    def summary(self) -> Dict[str, Any]:
        preds = self.predictions_only()
        if not preds:
            return {"total_entries": len(self.entries), "predictions": 0}

        import statistics
        risks = [e.details.get("churn_risk", 0) for e in preds]
        return {
            "total_entries":    len(self.entries),
            "predictions":      len(preds),
            "avg_churn_risk":   round(statistics.mean(risks), 4),
            "high_risk_count":  sum(1 for e in preds if e.details.get("risk_level") == "HIGH"),
            "total_mrr_at_risk": sum(e.details.get("mrr_at_risk", 0) for e in preds),
        }
