"""
villain_tracker.py — Banco de dados de stats por villain em tempo real.

Stats rastreados:
  VPIP        % entrou no pote voluntariamente (pré-flop)
  PFR         % abriu/reraised pré-flop
  3bet%       % 3betou quando teve oportunidade
  Fold/3bet   % foldou para 3bet
  AGG         % agressão pós-flop (bet/raise sobre total de ações)
  hands_seen  total de mãos observadas

Escrita: após cada F9 quando villain_name está disponível.
Leitura: antes de chamar claw analyze (exibe HUD no terminal).
Latência alvo: < 5ms (SQLite WAL + índice por nome).
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[1] / "db" / "villains.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS villains (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,
    hands_seen      INTEGER NOT NULL DEFAULT 0,

    -- Acumuladores brutos (numerador / denominador) para recalculo exato
    vpip_num        INTEGER NOT NULL DEFAULT 0,
    vpip_den        INTEGER NOT NULL DEFAULT 0,
    pfr_num         INTEGER NOT NULL DEFAULT 0,
    pfr_den         INTEGER NOT NULL DEFAULT 0,
    tbet_num        INTEGER NOT NULL DEFAULT 0,   -- 3bet
    tbet_den        INTEGER NOT NULL DEFAULT 0,
    f3b_num         INTEGER NOT NULL DEFAULT 0,   -- fold to 3bet
    f3b_den         INTEGER NOT NULL DEFAULT 0,
    agg_num         INTEGER NOT NULL DEFAULT 0,   -- ações agressivas
    agg_den         INTEGER NOT NULL DEFAULT 0,   -- total ações pós-flop

    -- Stats derivados (cache recalculado a cada update)
    vpip            REAL    NOT NULL DEFAULT 0,
    pfr             REAL    NOT NULL DEFAULT 0,
    three_bet_pct   REAL    NOT NULL DEFAULT 0,
    fold_to_3bet    REAL    NOT NULL DEFAULT 0,
    agg_freq        REAL    NOT NULL DEFAULT 0,

    first_seen      TEXT,
    last_seen       TEXT,
    notes           TEXT    -- JSON livre
);

CREATE TABLE IF NOT EXISTS villain_actions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    villain_name    TEXT    NOT NULL,
    session_id      TEXT    NOT NULL,
    street          TEXT    NOT NULL,  -- preflop | flop | turn | river
    action          TEXT    NOT NULL,  -- fold | call | raise | check | bet | 3bet | 4bet | limp
    amount          REAL,
    pot_at_action   REAL,
    position        TEXT,
    recorded_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_villain_name
    ON villains(name);

CREATE INDEX IF NOT EXISTS idx_actions_villain
    ON villain_actions(villain_name, street);

CREATE INDEX IF NOT EXISTS idx_actions_session
    ON villain_actions(session_id);
"""


# ---------------------------------------------------------------------------
# Conexão
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=5, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    return c


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------

def get_villain(name: str) -> Optional[dict]:
    """Retorna dict com stats do villain ou None se nunca visto."""
    name = (name or "").strip()
    if not name:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM villains WHERE name = ?", (name,)
        ).fetchone()
    return dict(row) if row else None


def format_hud(name: str) -> str:
    """
    Linha de HUD compacta para exibir no terminal antes da decisão.
    Exemplo: [Villain42] VPIP:28% PFR:18% 3B:7% F3B:55% AGG:34% (47 mãos)
    """
    stats = get_villain(name)
    if stats is None or stats["hands_seen"] == 0:
        return f"[{name or 'Villain'}] Sem histórico — primeiro encontro"
    n = stats["hands_seen"]
    return (
        f"[{stats['name']}] "
        f"VPIP:{stats['vpip']*100:.0f}% "
        f"PFR:{stats['pfr']*100:.0f}% "
        f"3B:{stats['three_bet_pct']*100:.0f}% "
        f"F3B:{stats['fold_to_3bet']*100:.0f}% "
        f"AGG:{stats['agg_freq']*100:.0f}% "
        f"({n} mão{'s' if n != 1 else ''})"
    )


def all_villains() -> list[dict]:
    """Retorna todos os villains ordenados por hands_seen desc."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM villains ORDER BY hands_seen DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Escrita
# ---------------------------------------------------------------------------

def record_action(
    villain_name: str,
    session_id: str,
    street: str,
    action: str,
    amount: float = 0.0,
    pot_at_action: float = 0.0,
    position: str = "",
) -> None:
    """
    Registra uma ação observada do villain e atualiza suas stats.

    action esperado (case-insensitive):
      preflop: fold, limp, call, raise, 3bet, 4bet
      pós-flop: check, call, bet, raise, fold
    """
    name = (villain_name or "").strip()
    if not name:
        return

    action_l = action.strip().lower()
    street_l  = street.strip().lower()
    is_pre    = street_l == "preflop"
    is_post   = street_l in ("flop", "turn", "river")

    now = datetime.now(timezone.utc).isoformat()

    with _conn() as c:
        # Insere ação bruta
        c.execute(
            """INSERT INTO villain_actions
               (villain_name, session_id, street, action, amount, pot_at_action, position)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, session_id, street_l, action_l, amount, pot_at_action, position),
        )

        # Garante registro do villain
        c.execute(
            """INSERT OR IGNORE INTO villains (name, first_seen)
               VALUES (?, ?)""",
            (name, now),
        )

        row = dict(c.execute(
            "SELECT * FROM villains WHERE name = ?", (name,)
        ).fetchone())

        # Incrementa denominadores e numeradores conforme a ação
        vpip_num = row["vpip_num"]
        vpip_den = row["vpip_den"]
        pfr_num  = row["pfr_num"]
        pfr_den  = row["pfr_den"]
        tb_num   = row["tbet_num"]
        tb_den   = row["tbet_den"]
        f3b_num  = row["f3b_num"]
        f3b_den  = row["f3b_den"]
        agg_num  = row["agg_num"]
        agg_den  = row["agg_den"]

        if is_pre:
            vpip_den += 1
            if action_l in ("call", "raise", "3bet", "4bet", "limp"):
                vpip_num += 1

            pfr_den += 1
            if action_l in ("raise", "3bet", "4bet"):
                pfr_num += 1

            # Oportunidade de 3bet: simplificado — apenas quando ação é 3bet ou fold-para-raise
            if action_l in ("3bet", "fold", "call"):
                tb_den += 1
                if action_l == "3bet":
                    tb_num += 1

            # Fold to 3bet — simplificado: fold pré quando já há raise anterior
            if action_l == "fold":
                f3b_den += 1
                f3b_num += 1  # assume que fold pré é em resposta a raise

        if is_post:
            agg_den += 1
            if action_l in ("bet", "raise", "3bet", "4bet"):
                agg_num += 1

        def _rate(num: int, den: int) -> float:
            return num / den if den > 0 else 0.0

        hands_new = row["hands_seen"] + (1 if is_pre else 0)

        c.execute(
            """UPDATE villains SET
               hands_seen    = ?,
               vpip_num = ?, vpip_den = ?, vpip          = ?,
               pfr_num  = ?, pfr_den  = ?, pfr           = ?,
               tbet_num = ?, tbet_den = ?, three_bet_pct = ?,
               f3b_num  = ?, f3b_den  = ?, fold_to_3bet  = ?,
               agg_num  = ?, agg_den  = ?, agg_freq      = ?,
               last_seen = ?
               WHERE name = ?""",
            (
                hands_new,
                vpip_num, vpip_den, _rate(vpip_num, vpip_den),
                pfr_num,  pfr_den,  _rate(pfr_num,  pfr_den),
                tb_num,   tb_den,   _rate(tb_num,   tb_den),
                f3b_num,  f3b_den,  _rate(f3b_num,  f3b_den),
                agg_num,  agg_den,  _rate(agg_num,  agg_den),
                now,
                name,
            ),
        )


def infer_and_record(state: dict, session_id: str) -> None:
    """
    Infere a ação do villain a partir do estado capturado na tela e registra.

    Lógica simples:
      - Se call > 0  → villain bet/raised (ação agressiva pré ou pós)
      - Se call == 0 → villain checked (passivo)
    Pode ser expandida com leitura de histórico de ações na tela.
    """
    name = state.get("villain_name", "").strip()
    if not name:
        return

    call = state.get("call", 0) or 0
    pot  = state.get("pot",  0) or 0

    # Inferência básica
    if call > 0:
        action = "raise"   # villain gerou call obligation
        street = "preflop" # conservador — sem board = preflop
        if state.get("board"):
            board_len = len(state["board"]) // 2
            street = {1: "flop", 2: "turn", 3: "river", 4: "turn", 5: "river"}.get(board_len, "flop")
    else:
        action = "check"
        street = "flop" if state.get("board") else "preflop"

    record_action(
        villain_name=name,
        session_id=session_id,
        street=street,
        action=action,
        amount=call,
        pot_at_action=pot,
        position=state.get("position", ""),
    )


# ---------------------------------------------------------------------------
# CLI: python villain_tracker.py [list | reset <name>]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "list":
        rows = all_villains()
        if not rows:
            print("Nenhum villain no banco ainda.")
        else:
            print(f"{'Nome':<20} {'Mãos':>5} {'VPIP':>6} {'PFR':>5} {'3B':>4} {'F3B':>5} {'AGG':>5}")
            print("-" * 60)
            for r in rows:
                print(
                    f"{r['name']:<20} {r['hands_seen']:>5} "
                    f"{r['vpip']*100:>5.0f}% {r['pfr']*100:>4.0f}% "
                    f"{r['three_bet_pct']*100:>3.0f}% {r['fold_to_3bet']*100:>4.0f}% "
                    f"{r['agg_freq']*100:>4.0f}%"
                )

    elif cmd == "hud" and len(sys.argv) > 2:
        print(format_hud(sys.argv[2]))

    elif cmd == "reset" and len(sys.argv) > 2:
        name = sys.argv[2]
        with _conn() as c:
            c.execute("DELETE FROM villains WHERE name = ?", (name,))
            c.execute("DELETE FROM villain_actions WHERE villain_name = ?", (name,))
        print(f"Stats de '{name}' removidos.")

    else:
        print("Uso: villain_tracker.py [list | hud <nome> | reset <nome>]")
