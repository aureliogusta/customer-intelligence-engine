from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

from hand_history_watcher import (
    DB_CONFIG,
    WATCH_DIRS,
    _extract_actions_from_block,
    get_connection,
    parse_file,
)
from leak_analysis.modules import execute_query


def _to_dt(date_utc: str | None):
    if not date_utc:
        return None
    try:
        dt = datetime.strptime(date_utc, "%Y/%m/%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _upsert_file(conn, filepath: Path, hero_name: str, site: str) -> int:
    parsed = parse_file(filepath, hero_name=hero_name, site=site)
    if not parsed:
        return 0

    cur = conn.cursor()
    try:
        source_file = str(filepath)
        tournament_name = filepath.stem
        recs = [p.record for p in parsed]
        net_chips = sum(r.hero_amount_won for r in recs)
        hands_won = sum(1 for r in recs if r.hero_result == "win")

        cur.execute(
            """
            INSERT INTO sessions
                (hero_name, source_file, tournament_name, hands_count, hands_won, net_chips)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_file) DO UPDATE SET
                hero_name = EXCLUDED.hero_name,
                tournament_name = EXCLUDED.tournament_name,
                hands_count = EXCLUDED.hands_count,
                hands_won = EXCLUDED.hands_won,
                net_chips = EXCLUDED.net_chips,
                ingested_at = NOW()
            RETURNING session_id
            """,
            (hero_name, source_file, tournament_name, len(recs), hands_won, net_chips),
        )
        session_id = cur.fetchone()[0]

        rows = []
        for p in parsed:
            r = p.record
            rows.append(
                (
                    r.hand_id,
                    str(session_id),
                    r.tournament_id,
                    r.tournament_name,
                    r.table_id,
                    _to_dt(r.date_utc),
                    r.level,
                    r.small_blind,
                    r.big_blind,
                    r.ante,
                    r.max_seats,
                    r.num_players,
                    r.btn_seat,
                    hero_name,
                    r.hero_seat,
                    r.hero_position,
                    r.hero_stack_start,
                    r.hero_stack_end,
                    r.hero_cards,
                    r.hero_action_preflop,
                    r.hero_action_flop,
                    r.hero_action_turn,
                    r.hero_action_river,
                    r.hero_vpip,
                    r.hero_pfr,
                    r.hero_aggressor,
                    r.hero_went_allin,
                    r.board_flop,
                    r.board_turn,
                    r.board_river,
                    r.hero_result,
                    r.hero_amount_won,
                    r.pot_final,
                    r.went_to_showdown,
                    r.m_ratio,
                )
            )

        execute_values(
            cur,
            """
            INSERT INTO hands (
                hand_id, session_id, tournament_id, tournament_name, table_id,
                date_utc, level, small_blind, big_blind, ante, max_seats,
                num_players, btn_seat, hero_name, hero_seat, hero_position,
                hero_stack_start, hero_stack_end, hero_cards,
                hero_action_preflop, hero_action_flop,
                hero_action_turn, hero_action_river,
                hero_vpip, hero_pfr, hero_aggressor, hero_went_allin,
                board_flop, board_turn, board_river,
                hero_result, hero_amount_won, pot_final,
                went_to_showdown, m_ratio
            ) VALUES %s
            ON CONFLICT (hand_id) DO UPDATE SET
                session_id = EXCLUDED.session_id,
                level = EXCLUDED.level,
                small_blind = EXCLUDED.small_blind,
                big_blind = EXCLUDED.big_blind,
                ante = EXCLUDED.ante,
                num_players = EXCLUDED.num_players,
                hero_position = EXCLUDED.hero_position,
                hero_stack_start = EXCLUDED.hero_stack_start,
                hero_stack_end = EXCLUDED.hero_stack_end,
                hero_action_preflop = EXCLUDED.hero_action_preflop,
                hero_action_flop = EXCLUDED.hero_action_flop,
                hero_action_turn = EXCLUDED.hero_action_turn,
                hero_action_river = EXCLUDED.hero_action_river,
                hero_amount_won = EXCLUDED.hero_amount_won,
                pot_final = EXCLUDED.pot_final,
                m_ratio = EXCLUDED.m_ratio,
                ingested_at = NOW()
            """,
            rows,
        )

        hand_ids = [p.record.hand_id for p in parsed if p.record.hand_id]
        cur.execute("SELECT id, hand_id FROM hands WHERE hand_id = ANY(%s)", (hand_ids,))
        id_rows = cur.fetchall()
        hand_id_to_pk = {row[1]: row[0] for row in id_rows}

        db_hand_ids = list(hand_id_to_pk.values())
        if db_hand_ids:
            cur.execute("DELETE FROM actions WHERE hand_id_ref = ANY(%s)", (db_hand_ids,))

        action_rows = []
        for item in parsed:
            hand_pk = hand_id_to_pk.get(item.record.hand_id)
            if hand_pk is None:
                continue
            for _, action_order, street, actor, action_type, amount, pot_before, is_all_in in _extract_actions_from_block(
                hand_id=item.record.hand_id,
                raw_block=item.raw_block,
                site=item.site,
            ):
                action_rows.append(
                    (
                        hand_pk,
                        action_order,
                        street,
                        actor,
                        action_type,
                        amount,
                        pot_before,
                        is_all_in,
                    )
                )

        if action_rows:
            execute_values(
                cur,
                """
                INSERT INTO actions (
                    hand_id_ref, action_order, street, actor_id, action_type,
                    amount, pot_size_before, is_all_in
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
                action_rows,
            )

        conn.commit()
        return len(recs)
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def run_backfill() -> None:
    total = 0
    conn = get_connection()
    try:
        for entry in WATCH_DIRS:
            watch_path = Path(entry["dir"])
            if not watch_path.exists():
                continue
            files = sorted(watch_path.glob("**/*.txt"))
            for f in files:
                total += _upsert_file(conn, f, hero_name=entry["hero"], site=entry["site"])
    finally:
        conn.close()

    try:
        execute_query("SELECT refresh_leak_analysis_materialized_views(TRUE)", fetch="none")
    except Exception:
        execute_query("SELECT refresh_leak_analysis_materialized_views(FALSE)", fetch="none")

    print(f"[BACKFILL] {total} maos reprocessadas com UPSERT sem perda de historico.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill/Upsert seguro de hand histories")
    parser.add_argument("--password", type=str, default="")
    args = parser.parse_args()
    if args.password:
        DB_CONFIG["password"] = args.password
    if not DB_CONFIG.get("password"):
        DB_CONFIG["password"] = os.getenv("POKER_DB_PASSWORD", "")
    run_backfill()


if __name__ == "__main__":
    main()
