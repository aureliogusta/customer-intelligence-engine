"""
Backfill inicial da tabela actions a partir das colunas hero_action_* da tabela hands.

Observacao: este backfill preenche apenas a trilha do hero.
Para acao completa de todos os atores, execute nova ingestao dos hand histories.
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values


def _load_db_config() -> dict:
    cfg = Path(__file__).resolve().parent / "db_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8"))
    return {
        "host": "localhost",
        "port": 5432,
        "database": "poker_dss",
        "user": "postgres",
        "password": "",
    }


def _parse_actions(action_text: str, street: str, hand_id_ref: int, hero_name: str, start_order: int) -> list[tuple]:
    if not action_text:
        return []
    tokens = [t.strip() for t in str(action_text).replace("→", ",").split(",") if t.strip()]
    out = []
    order = start_order
    for tok in tokens:
        upper = tok.upper()
        if "RAISE" in upper:
            action_type = "RAISE"
        elif "BET" in upper:
            action_type = "BET"
        elif "CALL" in upper:
            action_type = "CALL"
        elif "FOLD" in upper:
            action_type = "FOLD"
        elif "CHECK" in upper:
            action_type = "CHECK"
        else:
            action_type = "UNKNOWN"
        is_all_in = "ALL-IN" in upper or "ALL IN" in upper
        order += 1
        out.append((hand_id_ref, order, street, hero_name or "HERO", action_type, 0.0, 0.0, is_all_in))
    return out


def main() -> None:
    conn = psycopg2.connect(**_load_db_config())
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, hero_name, hero_action_preflop, hero_action_flop, hero_action_turn, hero_action_river
            FROM hands
            ORDER BY date_utc NULLS LAST
            """
        )
        rows = cur.fetchall()

        all_actions = []
        for hand_id_ref, hero_name, pre, flop, turn, river in rows:
            order = 0
            pre_rows = _parse_actions(pre, "PREFLOP", hand_id_ref, hero_name, order)
            all_actions.extend(pre_rows)
            order += len(pre_rows)

            flop_rows = _parse_actions(flop, "FLOP", hand_id_ref, hero_name, order)
            all_actions.extend(flop_rows)
            order += len(flop_rows)

            turn_rows = _parse_actions(turn, "TURN", hand_id_ref, hero_name, order)
            all_actions.extend(turn_rows)
            order += len(turn_rows)

            river_rows = _parse_actions(river, "RIVER", hand_id_ref, hero_name, order)
            all_actions.extend(river_rows)

        cur.execute("DELETE FROM actions")
        if all_actions:
            execute_values(
                cur,
                """
                INSERT INTO actions (
                    hand_id_ref, action_order, street, actor_id, action_type, amount, pot_size_before, is_all_in
                ) VALUES %s
                ON CONFLICT (hand_id_ref, action_order) DO UPDATE SET
                    street = EXCLUDED.street,
                    actor_id = EXCLUDED.actor_id,
                    action_type = EXCLUDED.action_type,
                    amount = EXCLUDED.amount,
                    pot_size_before = EXCLUDED.pot_size_before,
                    is_all_in = EXCLUDED.is_all_in,
                    created_at = NOW()
                """,
                all_actions,
            )
        conn.commit()
        print(f"Backfill concluido: {len(all_actions)} actions")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
