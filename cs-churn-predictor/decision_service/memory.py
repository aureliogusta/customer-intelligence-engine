"""
decision_service/memory.py
===========================
VectorMemoryStore para histórico semântico de contas.

Portado do claw-code-main/src/vector_memory.py e adaptado para CS:
  - Provider "hash" funciona 100% offline (sem OpenAI, sem GPU)
  - Provider "openai" usa text-embedding-3-small para semântica real
  - Fallback gracioso: se não há DB, retorna listas vazias

Uso:
    store = AccountMemoryStore.from_env()
    store.init_schema()

    store.remember_prediction(
        account_id="ACC_000042",
        session_id="sess_abc",
        churn_risk=0.85,
        risk_level="HIGH",
        actions=["ESCALATE", "SCHEDULE_CALL"],
    )

    past = store.recall_account("ACC_000042", limit=3)
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_DIMENSIONS  = 384
TABLE_NAME          = "churn_memories"


@dataclass(frozen=True)
class MemoryMatch:
    id:          int
    text:        str
    account_id:  str
    distance:    float
    metadata:    dict[str, Any]


class AccountMemoryStore:
    """
    Armazena e recupera memórias semânticas por conta.

    Compatível com PostgreSQL + pgvector (extension vector).
    Cai graciosamente para operações noop se DSN não estiver configurado.
    """

    def __init__(
        self,
        dsn:        str | None = None,
        provider:   str = "hash",        # "hash" (offline) ou "openai"
        dimensions: int = DEFAULT_DIMENSIONS,
    ) -> None:
        self.dsn        = dsn or os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL")
        self.provider   = provider
        self.dimensions = dimensions

        if not self.dsn:
            raise ValueError("DSN não fornecido. Configure PGVECTOR_DSN ou DATABASE_URL.")

    @classmethod
    def from_env(cls) -> "AccountMemoryStore | None":
        """Cria store a partir de variáveis de ambiente. Retorna None se não configurado."""
        dsn = os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL")
        if not dsn:
            return None
        try:
            return cls(dsn=dsn)
        except Exception:
            return None

    # ── Schema ────────────────────────────────────────────────────────────────

    def init_schema(self) -> None:
        """Cria tabela e índices se não existirem."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        id          BIGSERIAL PRIMARY KEY,
                        account_id  TEXT NOT NULL,
                        session_id  TEXT,
                        text        TEXT NOT NULL,
                        embedding   vector({self.dimensions}) NOT NULL,
                        metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_embedding
                    ON {TABLE_NAME}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_account
                    ON {TABLE_NAME} (account_id, created_at DESC)
                """)

    # ── Write ─────────────────────────────────────────────────────────────────

    def remember(
        self,
        text:       str,
        account_id: str  = "default",
        session_id: str  | None = None,
        metadata:   dict[str, Any] | None = None,
    ) -> int:
        embedding = self._embed(text)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {TABLE_NAME} (account_id, session_id, text, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (
                        account_id,
                        session_id,
                        text,
                        self._vec_literal(embedding),
                        json.dumps(metadata or {}),
                    ),
                )
                row = cur.fetchone()
                return int(row[0])

    def remember_prediction(
        self,
        account_id:  str,
        session_id:  str,
        churn_risk:  float,
        risk_level:  str,
        actions:     list[str],
        extra:       dict[str, Any] | None = None,
    ) -> int:
        """Conveniência: salva o resultado de uma predição como memória."""
        text = (
            f"Conta {account_id}: churn_risk={churn_risk:.1%} [{risk_level}] "
            f"ações={','.join(actions)}"
        )
        metadata = {
            "churn_risk":  churn_risk,
            "risk_level":  risk_level,
            "actions":     actions,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            **(extra or {}),
        }
        return self.remember(text, account_id=account_id, session_id=session_id, metadata=metadata)

    # ── Read ──────────────────────────────────────────────────────────────────

    def recall(
        self,
        query:      str,
        account_id: str = "default",
        limit:      int = 5,
    ) -> list[MemoryMatch]:
        embedding = self._embed(query)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, text, account_id, metadata,
                           embedding <=> %s::vector AS distance
                    FROM {TABLE_NAME}
                    WHERE account_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        self._vec_literal(embedding),
                        account_id,
                        self._vec_literal(embedding),
                        limit,
                    ),
                )
                rows = cur.fetchall()

        return [
            MemoryMatch(
                id         = int(r[0]),
                text       = r[1],
                account_id = r[2],
                metadata   = r[3] if isinstance(r[3], dict) else {},
                distance   = float(r[4]),
            )
            for r in rows
        ]

    def recall_account(self, account_id: str, limit: int = 5) -> list[MemoryMatch]:
        """Retorna as últimas N memórias de uma conta sem query semântica."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, text, account_id, metadata, 0.0 AS distance
                    FROM {TABLE_NAME}
                    WHERE account_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (account_id, limit),
                )
                rows = cur.fetchall()

        return [
            MemoryMatch(
                id         = int(r[0]),
                text       = r[1],
                account_id = r[2],
                metadata   = r[3] if isinstance(r[3], dict) else {},
                distance   = float(r[4]),
            )
            for r in rows
        ]

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        if self.provider == "openai":
            return self._embed_openai(text)
        return self._embed_hash(text)

    def _embed_hash(self, text: str) -> list[float]:
        """
        Embedding determinístico baseado em hashing — zero dependências externas.
        Portado exatamente do claw-code para garantir comportamento idêntico.
        """
        tokens = [t for t in text.lower().split() if t]
        if not tokens:
            return [0.0] * self.dimensions

        vector = [0.0] * self.dimensions
        for ti, token in enumerate(tokens):
            digest = hashlib.sha256(f"{ti}:{token}".encode()).digest()
            for bi, byte in enumerate(digest):
                slot = (ti * 131 + bi * 17 + byte) % self.dimensions
                vector[slot] += (byte - 127.5) / 127.5

        norm = math.sqrt(sum(v * v for v in vector))
        return vector if norm <= 0 else [v / norm for v in vector]

    def _embed_openai(self, text: str) -> list[float]:
        try:
            openai = importlib.import_module("openai")
        except ImportError as exc:
            raise RuntimeError("Instale openai: pip install openai") from exc
        client  = openai.OpenAI()
        model   = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        resp    = client.embeddings.create(model=model, input=text, dimensions=self.dimensions)
        return [float(v) for v in resp.data[0].embedding]

    def _vec_literal(self, values: list[float]) -> str:
        return "[" + ",".join(f"{v:.8f}" for v in values) + "]"

    def _connect(self):
        try:
            psycopg = importlib.import_module("psycopg")
        except ImportError as exc:
            raise RuntimeError("Instale psycopg: pip install psycopg[binary]") from exc
        conn = psycopg.connect(self.dsn)
        conn.autocommit = True
        return conn


# ── Noop fallback ─────────────────────────────────────────────────────────────

class _NoopMemoryStore:
    """Substituto quando pgvector não está disponível. Silencia todas as chamadas."""

    def remember(self, *args, **kwargs) -> int | None:
        return None

    def remember_prediction(self, *args, **kwargs) -> int | None:
        return None

    def recall(self, *args, **kwargs) -> list:
        return []

    def recall_account(self, *args, **kwargs) -> list:
        return []

    def init_schema(self) -> None:
        pass


def get_memory_store() -> AccountMemoryStore | _NoopMemoryStore:
    """
    Factory que retorna AccountMemoryStore se pgvector disponível,
    ou _NoopMemoryStore como fallback silencioso.
    """
    store = AccountMemoryStore.from_env()
    return store if store is not None else _NoopMemoryStore()
