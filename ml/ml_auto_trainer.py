"""
ml_auto_trainer.py
==================
Automação de retreino ML em background quando novas mãos chegam.
Trigerado pelo hand_history_watcher.py após ingestão.

Lógica
------
1. Após ingestão de N mãos novas, checa timestamp do último treino
2. Se intervalo >= TRAIN_COOLDOWN_SEC, dispara retreino em thread
3. Exporta dados do BD, executa pipeline ML, atualiza modelos
4. Log com timestamp e métricas de novo treino
"""

from __future__ import annotations

import os
import sys
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone

import psycopg2
import pandas as pd

_BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_BASE_DIR))

try:
    from leak_analysis.modules.db import load_db_config as _load_central_db_config
except Exception:
    _load_central_db_config = None

try:
    from decision_service.training import train_ex_ante_policy
    from mlops.registry import latest_manifest_path, load_manifest
except ImportError:
    print("ERRO: novos módulos de decisão não encontrados.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [ML-AUTO]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml_auto_trainer")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_COOLDOWN_SEC = 300  # retreina a cada 5 min se havia novos dados
HANDS_THRESHOLD    = 5    # se >= 5 novas mãos, permite retreino


def _build_db_config() -> dict:
    """Resolve DB config with same precedence used by main pipeline modules."""
    cfg: dict = {}

    if _load_central_db_config is not None:
        try:
            cfg = dict(_load_central_db_config() or {})
        except Exception:
            cfg = {}

    if not cfg:
        cfg_path = _BASE_DIR / "db_config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}

    cfg["host"] = os.getenv("POKER_DB_HOST", str(cfg.get("host", "localhost")))
    cfg["port"] = int(os.getenv("POKER_DB_PORT", str(cfg.get("port", 5432))))
    cfg["database"] = os.getenv("POKER_DB_NAME", str(cfg.get("database", "poker_dss")))
    cfg["user"] = os.getenv("POKER_DB_USER", str(cfg.get("user", "postgres")))
    cfg["password"] = os.getenv("POKER_DB_PASSWORD", str(cfg.get("password", "")))
    return cfg


DB_CONFIG: dict = _build_db_config()

_last_train_at: datetime | None = None
_training_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_last_training_timestamp() -> datetime | None:
    """Carrega timestamp do último treino do registry novo."""
    manifest_path = latest_manifest_path("ex_ante_policy")
    if manifest_path is None:
        return None
    try:
        meta = load_manifest(manifest_path)
        ts_str = meta.get("trained_at") or meta.get("created_at")
        if ts_str:
            return datetime.fromisoformat(ts_str)
    except Exception as e:
        log.warning("Erro ao carregar timestamp: %s", e)
    return None


def get_hand_count_since_timestamp(ts: datetime | None) -> int:
    """Conta mãos ingested após timestamp."""
    if not ts:
        return 0
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Ajustar timestamp para UTC se não estiver
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        cur.execute(
            "SELECT COUNT(*) FROM hands WHERE ingested_at > %s",
            (ts,)
        )
        count = cur.fetchone()[0]
        return count
    except Exception as e:
        log.error("Erro ao contar mãos: %s", e)
        return 0
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def export_hands_from_db() -> str:
    """Legacy export helper; prefer the registry-based dataset snapshot."""
    output_path = _BASE_DIR / "hands_latest_from_db.csv"
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM hands ORDER BY date_utc ASC", conn)
        conn.close()
        df.to_csv(output_path, index=False)
        log.info("Exportadas mãos para %s", output_path)
        return str(output_path)
    except Exception as e:
        log.error("Erro ao exportar: %s", e)
        return ""


def run_ml_training_bg():
    """Executa treino ex-ante em background (thread)."""
    global _last_train_at
    
    if not _training_lock.acquire(blocking=False):
        log.info("Treino já em progresso, skipando...")
        return
    
    try:
        log.info("Iniciando retreino ex-ante em background...")
        conn = get_db_connection()
        try:
            df = pd.read_sql_query("SELECT * FROM hands ORDER BY date_utc ASC", conn)
        finally:
            conn.close()
        
        if df.empty:
            log.error("Nenhuma mao disponivel no BD.")
            return

        try:
            metadata = train_ex_ante_policy(df)
            _last_train_at = datetime.now(timezone.utc)
            log.info(
                "Retreino concluido! acc=%.3f f1=%.3f confidence=%s",
                metadata.test_accuracy,
                metadata.test_f1_macro,
                metadata.confidence,
            )
        except Exception as e:
            log.error("Erro durante treino ex-ante: %s", e)
    finally:
        _training_lock.release()


def should_retrain() -> bool:
    """Checa se deve trigerar retreino (intervalo + novo volume)."""
    global _last_train_at
    
    # Carregar timestamp do arquivo metadados se não em memória
    if _last_train_at is None:
        _last_train_at = load_last_training_timestamp()
    
    # Se nunca treinou, avança
    if _last_train_at is None:
        return False

    # Normaliza timestamp para evitar comparacao naive vs aware.
    if _last_train_at.tzinfo is None:
        _last_train_at = _last_train_at.replace(tzinfo=timezone.utc)
    
    # Checa intervalo
    now = datetime.now(timezone.utc)
    elapsed = (now - _last_train_at).total_seconds()
    if elapsed < TRAIN_COOLDOWN_SEC:
        return False
    
    # Checa novo volume desde último treino
    new_hands = get_hand_count_since_timestamp(_last_train_at)
    if new_hands < HANDS_THRESHOLD:
        log.debug("Apenas %d mãos novas (threshold=%d), skipando", new_hands, HANDS_THRESHOLD)
        return False
    
    log.info("Treino triggerado: %d mãos novas, intervalo %.0fs", new_hands, elapsed)
    return True


def maybe_trigger_ml_training():
    """Verifica se deve retreinar e dispara em thread se sim."""
    if should_retrain():
        thread = threading.Thread(target=run_ml_training_bg, daemon=True)
        thread.start()
        log.info("ML training thread iniciada")

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def trigger_ml_check_after_ingest():
    """
    Called pelo hand_history_watcher após ingestão de um arquivo.
    Verifica se é hora de retreinar e dispara em background se necessário.
    """
    maybe_trigger_ml_training()


if __name__ == "__main__":
    # Debug mode
    log.info("ML Auto Trainer — debug mode")
    maybe_trigger_ml_training()
