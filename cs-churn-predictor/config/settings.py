"""
config/settings.py
===================
Configuração centralizada via variáveis de ambiente.

Inspirado no padrão de WorkspaceSetup do claw-code-main/src/setup.py,
mas usando os tipos que já temos em vez de pydantic-settings (evita dep extra).

Uso:
    from config.settings import settings

    print(settings.models_dir)
    print(settings.ollama_url)
    if settings.pgvector_dsn:
        store = AccountMemoryStore(dsn=settings.pgvector_dsn)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent   # cs-churn-predictor/


@dataclass(frozen=True)
class ChurnSettings:
    # ── Paths ─────────────────────────────────────────────────────────────────
    models_dir:       Path
    data_dir:         Path
    logs_dir:         Path

    # ── Database (pgvector) ───────────────────────────────────────────────────
    pgvector_dsn:     str | None     # None = memória desabilitada
    embedding_provider: str          # "hash" | "openai"
    embedding_dimensions: int

    # ── LLM ──────────────────────────────────────────────────────────────────
    ollama_url:       str
    ollama_model:     str
    ollama_timeout:   int

    # ── API ───────────────────────────────────────────────────────────────────
    api_host:         str
    api_port:         int

    # ── ML ────────────────────────────────────────────────────────────────────
    high_risk_threshold:   float
    medium_risk_threshold: float
    retrain_on_drift:      bool
    perf_log_path:         Path

    # ── Monitoring ────────────────────────────────────────────────────────────
    mrr_alert_threshold:   float


def _env_path(key: str, default: Path) -> Path:
    return Path(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, "").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


def load_settings() -> ChurnSettings:
    return ChurnSettings(
        models_dir           = _env_path("MODELS_DIR",  ROOT / "ml" / "models"),
        data_dir             = _env_path("DATA_DIR",    ROOT / "data"),
        logs_dir             = _env_path("LOGS_DIR",    ROOT / "logs"),

        pgvector_dsn         = os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL"),
        embedding_provider   = os.getenv("EMBEDDING_PROVIDER", "hash"),
        embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "384")),

        ollama_url           = os.getenv("OLLAMA_URL",   "http://localhost:11434"),
        ollama_model         = os.getenv("OLLAMA_MODEL", "mistral:7b"),
        ollama_timeout       = int(os.getenv("OLLAMA_TIMEOUT", "120")),

        api_host             = os.getenv("API_HOST", "0.0.0.0"),
        api_port             = int(os.getenv("API_PORT", "8001")),

        high_risk_threshold   = float(os.getenv("HIGH_RISK_THRESHOLD",   "0.70")),
        medium_risk_threshold = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.40")),
        retrain_on_drift      = _env_bool("RETRAIN_ON_DRIFT", False),
        perf_log_path         = _env_path("PERF_LOG_PATH", ROOT / "logs" / "perf.jsonl"),

        mrr_alert_threshold   = float(os.getenv("MRR_ALERT_THRESHOLD", "50000")),
    )


# Singleton carregado na inicialização
settings = load_settings()
