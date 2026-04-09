"""
revenue_automation/feedback/store.py
======================================
FeedbackStore — persiste outcomes de intervenções em SQLite.

Propósito:
  - Registrar o resultado real de cada intervenção (renovação, churn, upsell, etc.)
  - Alimentar análise de performance e futuramente retreino de modelos
  - Zero novas dependências: SQLite já é padrão Python

Schema:
  feedback(
    feedback_id TEXT PK,
    correlation_id TEXT,
    account_id TEXT,
    outcome TEXT,
    notes TEXT,
    recorded_at TEXT,
    recorded_by TEXT,
    churn_risk_before REAL,
    churn_risk_after REAL
  )
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from ..schemas.models import FeedbackEntry, FeedbackOutcome

DEFAULT_DB = Path(os.getenv("FEEDBACK_DB_PATH", "logs/feedback.db"))

_lock = threading.Lock()

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id       TEXT PRIMARY KEY,
    correlation_id    TEXT NOT NULL,
    account_id        TEXT NOT NULL,
    outcome           TEXT NOT NULL,
    notes             TEXT DEFAULT '',
    recorded_at       TEXT NOT NULL,
    recorded_by       TEXT DEFAULT 'system',
    churn_risk_before REAL,
    churn_risk_after  REAL
);

CREATE INDEX IF NOT EXISTS idx_fb_account   ON feedback (account_id);
CREATE INDEX IF NOT EXISTS idx_fb_outcome   ON feedback (outcome);
CREATE INDEX IF NOT EXISTS idx_fb_corr      ON feedback (correlation_id);
CREATE INDEX IF NOT EXISTS idx_fb_date      ON feedback (recorded_at DESC);
"""


def _conn(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(path), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


class FeedbackStore:
    """
    Armazena e consulta feedback de intervenções.

    Uso:
        store = FeedbackStore()
        entry = store.record(
            correlation_id="abc123",
            account_id="ACC_001",
            outcome=FeedbackOutcome.RENEWAL_HAPPENED,
            notes="Cliente renovou contrato",
            churn_risk_before=0.85,
            churn_risk_after=0.12,
        )
        history = store.for_account("ACC_001")
    """

    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        with _lock:
            conn = _conn(self.db_path)
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(
        self,
        correlation_id:    str,
        account_id:        str,
        outcome:           FeedbackOutcome | str,
        notes:             str = "",
        recorded_by:       str = "system",
        churn_risk_before: Optional[float] = None,
        churn_risk_after:  Optional[float] = None,
    ) -> FeedbackEntry:
        entry = FeedbackEntry(
            feedback_id       = uuid4().hex,
            correlation_id    = correlation_id,
            account_id        = account_id,
            outcome           = outcome.value if isinstance(outcome, FeedbackOutcome) else outcome,
            notes             = notes,
            recorded_at       = datetime.now(timezone.utc).isoformat(),
            recorded_by       = recorded_by,
            churn_risk_before = churn_risk_before,
            churn_risk_after  = churn_risk_after,
        )
        with _lock:
            conn = _conn(self.db_path)
            conn.execute(
                """INSERT INTO feedback
                   (feedback_id, correlation_id, account_id, outcome, notes,
                    recorded_at, recorded_by, churn_risk_before, churn_risk_after)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    entry.feedback_id, entry.correlation_id, entry.account_id,
                    entry.outcome, entry.notes, entry.recorded_at, entry.recorded_by,
                    entry.churn_risk_before, entry.churn_risk_after,
                ),
            )
            conn.commit()
            conn.close()
        return entry

    # ── Read ──────────────────────────────────────────────────────────────────

    def for_account(self, account_id: str) -> List[FeedbackEntry]:
        with _lock:
            conn = _conn(self.db_path)
            rows = conn.execute(
                "SELECT * FROM feedback WHERE account_id = ? ORDER BY recorded_at DESC",
                (account_id,),
            ).fetchall()
            conn.close()
        return [_row_to_entry(r) for r in rows]

    def for_correlation(self, correlation_id: str) -> List[FeedbackEntry]:
        with _lock:
            conn = _conn(self.db_path)
            rows = conn.execute(
                "SELECT * FROM feedback WHERE correlation_id = ? ORDER BY recorded_at DESC",
                (correlation_id,),
            ).fetchall()
            conn.close()
        return [_row_to_entry(r) for r in rows]

    def outcome_counts(self, since_days: int = 30) -> Dict[str, int]:
        """Contagem de outcomes dos últimos N dias."""
        cutoff = datetime.now(timezone.utc).isoformat()
        with _lock:
            conn = _conn(self.db_path)
            rows = conn.execute(
                """SELECT outcome, COUNT(*) as cnt FROM feedback
                   WHERE recorded_at >= datetime('now', ?)
                   GROUP BY outcome ORDER BY cnt DESC""",
                (f"-{since_days} days",),
            ).fetchall()
            conn.close()
        return {r["outcome"]: r["cnt"] for r in rows}

    def total(self) -> int:
        with _lock:
            conn = _conn(self.db_path)
            n = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            conn.close()
        return int(n)

    def recent(self, limit: int = 50) -> List[FeedbackEntry]:
        with _lock:
            conn = _conn(self.db_path)
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY recorded_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
        return [_row_to_entry(r) for r in rows]


def _row_to_entry(row: sqlite3.Row) -> FeedbackEntry:
    return FeedbackEntry(
        feedback_id       = row["feedback_id"],
        correlation_id    = row["correlation_id"],
        account_id        = row["account_id"],
        outcome           = row["outcome"],
        notes             = row["notes"] or "",
        recorded_at       = row["recorded_at"],
        recorded_by       = row["recorded_by"] or "system",
        churn_risk_before = row["churn_risk_before"],
        churn_risk_after  = row["churn_risk_after"],
    )
