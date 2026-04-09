"""
leak_analysis/modules/db.py
Gerenciamento de conexão e inicialização do banco de dados PostgreSQL.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

log = logging.getLogger(__name__)

# Carregar config do DB
_CONFIG_PATH = Path(__file__).parent.parent.parent / "db_config.json"

def load_db_config() -> dict:
    """Carrega configuração de banco de dados."""
    env_password = os.getenv("POKER_DB_PASSWORD", "")
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            config = json.load(f)
    else:
        config = {
        "host": os.getenv("POKER_DB_HOST", "localhost"),
        "port": int(os.getenv("POKER_DB_PORT", "5432")),
        "database": os.getenv("POKER_DB_NAME", "poker_dss"),
        "user": os.getenv("POKER_DB_USER", "postgres"),
        "password": os.getenv("POKER_DB_PASSWORD", ""),
    }
    if env_password:
        config["password"] = env_password
    return config

_connection_pool: Optional[pool.SimpleConnectionPool] = None

def init_db() -> None:
    """Inicializa pool de conexão."""
    global _connection_pool
    if _connection_pool is not None:
        return
    
    config = load_db_config()
    try:
        _connection_pool = pool.SimpleConnectionPool(
            1, 10,
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )
        log.info(f"DB pool initialized: {config['database']}@{config['host']}")
    except Exception as e:
        log.error(f"Falha ao inicializar pool de DB: {e}")
        raise

def get_db_connection():
    """Obtém conexão do pool."""
    if _connection_pool is None:
        init_db()
    return _connection_pool.getconn()

def close_db_connection(conn) -> None:
    """Retorna conexão ao pool."""
    if _connection_pool and conn:
        _connection_pool.putconn(conn)

def execute_query(query: str, params: tuple = (), fetch: str = "all") -> list | dict | None:
    """
    Executa query e retorna resultado.
    
    Args:
        query: SQL statement
        params: Parâmetros da query
        fetch: "all", "one", "none"
    
    Returns:
        list de dicts (fetch="all"), dict (fetch="one"), None (fetch="none")
    """
    conn = get_db_connection()
    cur = None
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, params)
        
        if fetch == "all":
            result = cur.fetchall()
        elif fetch == "one":
            result = cur.fetchone()
        else:
            conn.commit()
            result = None
        
        conn.commit()
        return result
    except Exception as e:
        conn.rollback()
        log.error(f"DB error: {e}\nQuery: {query}")
        raise
    finally:
        if cur is not None:
            cur.close()
        close_db_connection(conn)

def apply_schema_leak_analysis() -> None:
    """Aplica o schema de análise de leaks do arquivo SQL."""
    conn = get_db_connection()
    cur = None
    try:
        schema_path = Path(__file__).parent.parent.parent / "schema_leak_analysis.sql"
        
        if not schema_path.exists():
            log.warning(f"Schema file not found: {schema_path}")
            return
        
        with open(schema_path) as f:
            schema_sql = f.read()
        
        cur = conn.cursor()
        cur.execute(schema_sql)
        conn.commit()
        
        log.info("Leak analysis schema applied successfully")
    except Exception as e:
        conn.rollback()
        log.error(f"Falha ao aplicar schema: {e}")
        raise
    finally:
        if cur is not None:
            cur.close()
        close_db_connection(conn)
