"""
hand_history_watcher.py
=======================
Serviço de ingestão contínua — Poker DSS Data Pipeline.
Suporta ACR (America's Cardroom) e PokerStars PT-BR.

Execução
--------
  python hand_history_watcher.py          # monitoramento contínuo
  python hand_history_watcher.py --once   # processa tudo e sai
"""

from __future__ import annotations

import sys
import os
import re
import time
import logging
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("ERRO: pip install psycopg2-binary")
    sys.exit(1)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("ERRO: pip install watchdog")
    sys.exit(1)

_BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_BASE_DIR))

try:
    from hand_history_parser import ACRHandParser, HandRecord
except ImportError:
    print("ERRO: hand_history_parser.py não encontrado.")
    sys.exit(1)

try:
    from pokerstars_parser import PSHandParser
    PS_AVAILABLE = True
except ImportError:
    PS_AVAILABLE = False

try:
    from ml_auto_trainer import trigger_ml_check_after_ingest
    ML_AVAILABLE = True
except ImportError:
    log_warning = True
    ML_AVAILABLE = False

try:
    from ingestion_service.diagnostics import record_ingestion_event, touch_watcher_heartbeat
except Exception:
    record_ingestion_event = None
    touch_watcher_heartbeat = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("watcher")

if not PS_AVAILABLE:
    log.warning("pokerstars_parser.py nao encontrado - PS sera ignorado.")

# =============================================================================
# CONFIGURACAO — edite aqui
# =============================================================================

WATCH_DIRS: list[dict] = []

DB_CONFIG: dict = {
    "host":     os.getenv("POKER_DB_HOST", "localhost"),
    "port":     int(os.getenv("POKER_DB_PORT", "5432")),
    "database": os.getenv("POKER_DB_NAME", "poker_dss"),
    "user":     os.getenv("POKER_DB_USER", "postgres"),
    "password": os.getenv("POKER_DB_PASSWORD", ""),
}

WATCH_EXTENSION:  str = ".txt"
COOLDOWN_SECONDS: int = 5


def _expand_existing_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp).lower()
        if key in seen:
            continue
        if rp.exists() and rp.is_dir():
            out.append(rp)
            seen.add(key)
    return out


def auto_detect_watch_dirs() -> list[dict]:
    home = Path.home()
    local = home / "AppData" / "Local"
    program_files_x86 = Path(os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    program_files = Path(os.getenv("ProgramFiles", r"C:\Program Files"))

    hero_acr = os.getenv("POKER_HERO_ACR", "AurelioDizzy")
    hero_ps = os.getenv("POKER_HERO_PS", "PioDizzy")

    acr_candidates = _expand_existing_dirs([
        program_files_x86 / "ACR Poker" / "handHistory",
        program_files / "ACR Poker" / "handHistory",
        local / "AmericasCardroom" / "HandHistory",
        local / "AmericasCardroom" / "handHistory",
    ])

    ps_roots = _expand_existing_dirs([
        local / "PokerStars.ES" / "HandHistory",
        local / "PokerStars" / "HandHistory",
        local / "PokerStars.FR" / "HandHistory",
        local / "PokerStars.EU" / "HandHistory",
    ])
    ps_candidates: list[Path] = []
    for root in ps_roots:
        children = [p for p in root.iterdir() if p.is_dir()]
        if children:
            ps_candidates.extend(children)
        else:
            ps_candidates.append(root)
    ps_candidates = _expand_existing_dirs(ps_candidates)

    dirs: list[dict] = []
    for p in acr_candidates:
        dirs.append({"dir": str(p), "hero": hero_acr, "site": "acr"})
    for p in ps_candidates:
        dirs.append({"dir": str(p), "hero": hero_ps, "site": "ps"})

    if not dirs:
        dirs = [
            {
                "dir": r"C:\Program Files (x86)\ACR Poker\handHistory",
                "hero": hero_acr,
                "site": "acr",
            },
            {
                "dir": str(local / "PokerStars.ES" / "HandHistory"),
                "hero": hero_ps,
                "site": "ps",
            },
        ]
    return dirs


WATCH_DIRS = auto_detect_watch_dirs()

# =============================================================================
# Utilitarios
# =============================================================================

def detect_site(text: str) -> str:
    if "Mao PokerStars #" in text or "PokerStars Hand #" in text or "Mão PokerStars #" in text:
        return "ps"
    if "Game Hand #" in text:
        return "acr"
    return "unknown"


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def test_connection() -> bool:
    try:
        conn = get_connection()
        conn.close()
        return True
    except Exception as e:
        log.error("Falha na conexao: %s", e)
        return False


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class ParsedHandBlock:
    record: HandRecord
    raw_block: str
    site: str


_RE_ACR_ACTION = re.compile(
    r"^(?P<actor>[^:]+?)(?::)?\s+(?P<action>folds|checks|calls|bets|raises)(?:\s+\$?(?P<a1>[\d.,]+))?(?:\s+to\s+\$?(?P<a2>[\d.,]+))?",
    re.IGNORECASE,
)

_RE_PS_ACTION = re.compile(
    r"^(?P<actor>[^:]+):\s+(?P<action>desiste|passa|iguala|aposta|aumenta)(?:\s+\$?(?P<a1>[\d.,]+))?(?:\s+para\s+\$?(?P<a2>[\d.,]+))?",
    re.IGNORECASE,
)

_RE_ACR_ANTE = re.compile(r"^(?P<actor>\S+)\s+posts\s+ante\s+\$?(?P<amount>[\d.,]+)", re.IGNORECASE)
_RE_ACR_BLIND = re.compile(r"^(?P<actor>\S+)\s+posts\s+the\s+(small|big)\s+blind\s+\$?(?P<amount>[\d.,]+)", re.IGNORECASE)
_RE_PS_ANTE = re.compile(r"^(?P<actor>\S+):\s+coloca\s+ante\s+\$?(?P<amount>[\d.,]+)", re.IGNORECASE)
_RE_PS_BLIND = re.compile(r"^(?P<actor>\S+):\s+paga\s+o\s+(small blind|big blind)\s+\$?(?P<amount>[\d.,]+)", re.IGNORECASE)


def _normalize_amount(raw: str | None) -> float:
    if not raw:
        return 0.0
    txt = raw.strip().replace("$", "")
    if "," in txt and "." not in txt:
        txt = txt.replace(",", ".")
    else:
        txt = txt.replace(",", "")
    try:
        return float(txt)
    except Exception:
        return 0.0


def _normalize_action_type(action: str, site: str) -> str:
    token = (action or "").strip().lower()
    if site == "ps":
        mapping = {
            "desiste": "FOLD",
            "passa": "CHECK",
            "iguala": "CALL",
            "aposta": "BET",
            "aumenta": "RAISE",
        }
        return mapping.get(token, token.upper())
    mapping = {
        "folds": "FOLD",
        "checks": "CHECK",
        "calls": "CALL",
        "bets": "BET",
        "raises": "RAISE",
    }
    return mapping.get(token, token.upper())


def _split_hand_blocks(text: str, site: str) -> list[str]:
    if site == "acr":
        blocks = [b.strip() for b in text.split("Game Hand #") if b.strip()]
        return ["Game Hand #" + b for b in blocks]
    if site == "ps":
        blocks = re.split(r"(?=Mão PokerStars #|Mao PokerStars #)", text)
        return [b.strip() for b in blocks if b.strip()]
    return []


def _extract_actions_from_block(hand_id: str, raw_block: str, site: str) -> list[tuple]:
    rows: list[tuple] = []
    street = "PREFLOP"
    pot = 0.0
    order = 0

    for line in raw_block.splitlines():
        line = line.strip()
        if not line:
            continue

        up = line.upper()
        if "*** FLOP ***" in up:
            street = "FLOP"
            continue
        if "*** TURN ***" in up:
            street = "TURN"
            continue
        if "*** RIVER ***" in up:
            street = "RIVER"
            continue
        if "*** SUMMARY ***" in up or "*** RESUMO ***" in up:
            break

        if site == "acr":
            m_ante = _RE_ACR_ANTE.match(line)
            if m_ante:
                pot += _normalize_amount(m_ante.group("amount"))
                continue
            m_blind = _RE_ACR_BLIND.match(line)
            if m_blind:
                pot += _normalize_amount(m_blind.group("amount"))
                continue
        else:
            m_ante = _RE_PS_ANTE.match(line)
            if m_ante:
                pot += _normalize_amount(m_ante.group("amount"))
                continue
            m_blind = _RE_PS_BLIND.match(line)
            if m_blind:
                pot += _normalize_amount(m_blind.group("amount"))
                continue

        matcher = _RE_ACR_ACTION if site == "acr" else _RE_PS_ACTION
        m = matcher.match(line)
        if not m:
            continue

        actor = m.group("actor").strip()
        action_type = _normalize_action_type(m.group("action"), site)
        amount = _normalize_amount(m.group("a2") or m.group("a1"))
        pot_before = pot
        is_all_in = "all-in" in line.lower() or "all in" in line.lower()

        order += 1
        rows.append((hand_id, order, street, actor, action_type, amount, pot_before, is_all_in))

        if action_type in {"CALL", "BET", "RAISE"}:
            pot += amount

    return rows


def parse_file(filepath: Path, hero_name: str, site: str) -> list[ParsedHandBlock]:
    tournament_name = filepath.stem
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.error("Erro ao ler %s: %s", filepath.name, e)
        return []

    if site == "auto":
        site = detect_site(text)

    records: list[ParsedHandBlock] = []

    if site == "acr":
        parser = ACRHandParser(hero_name=hero_name)
        blocks = _split_hand_blocks(text, "acr")
        for block in blocks:
            try:
                rec = parser.parse(block, tournament_name)
                if rec and rec.hand_id:
                    records.append(ParsedHandBlock(record=rec, raw_block=block, site="acr"))
            except Exception as e:
                log.debug("ACR parse error: %s", e)

    elif site == "ps" and PS_AVAILABLE:
        parser = PSHandParser(hero_name=hero_name)
        blocks = _split_hand_blocks(text, "ps")
        for block in blocks:
            try:
                rec = parser.parse(block, tournament_name)
                if rec and rec.hand_id:
                    records.append(ParsedHandBlock(record=rec, raw_block=block, site="ps"))
            except Exception as e:
                log.debug("PS parse error: %s", e)

    return records


def ingest_file(filepath: Path, conn, hero_name: str, site: str = "auto") -> int:

    records = parse_file(filepath, hero_name, site)
    if not records:
        log.debug("Nenhuma mao do hero encontrada em %s.", filepath.name)
        return 0

    source_file     = str(filepath)
    tournament_name = filepath.stem
    cur = conn.cursor()

    try:
        net_chips = sum(p.record.hero_amount_won for p in records)
        hands_won = sum(1 for p in records if p.record.hero_result == "win")

        cur.execute("""
            INSERT INTO sessions
                (hero_name, source_file, tournament_name, hands_count, hands_won, net_chips)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_file) DO UPDATE SET
                hands_count = EXCLUDED.hands_count,
                hands_won   = EXCLUDED.hands_won,
                net_chips   = EXCLUDED.net_chips,
                ingested_at = NOW()
            RETURNING session_id
        """, (hero_name, source_file, tournament_name,
              len(records), hands_won, net_chips))

        session_id = cur.fetchone()[0]

        rows = []
        for p in records:
            r = p.record
            date_utc = None
            if r.date_utc:
                try:
                    date_utc = datetime.strptime(r.date_utc, "%Y/%m/%d %H:%M:%S")
                    date_utc = date_utc.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass

            rows.append((
                r.hand_id, str(session_id), r.tournament_id,
                r.tournament_name, r.table_id, date_utc,
                r.level, r.small_blind, r.big_blind, r.ante,
                r.max_seats, r.num_players, r.btn_seat,
                hero_name, r.hero_seat, r.hero_position,
                r.hero_stack_start, r.hero_stack_end, r.hero_cards,
                r.hero_action_preflop, r.hero_action_flop,
                r.hero_action_turn, r.hero_action_river,
                r.hero_vpip, r.hero_pfr, r.hero_aggressor, r.hero_went_allin,
                r.board_flop, r.board_turn, r.board_river,
                r.hero_result, r.hero_amount_won, r.pot_final,
                r.went_to_showdown, r.m_ratio,
            ))

        execute_values(cur, """
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
            ON CONFLICT (hand_id) DO NOTHING
        """, rows)
        inserted_hands = max(0, cur.rowcount)

        hand_ids = [p.record.hand_id for p in records if p.record.hand_id]
        cur.execute(
            "SELECT id, hand_id FROM hands WHERE hand_id = ANY(%s)",
            (hand_ids,),
        )
        id_rows = cur.fetchall()
        hand_id_to_pk = {row[1]: row[0] for row in id_rows}

        db_hand_ids = list(hand_id_to_pk.values())
        if db_hand_ids:
            cur.execute("DELETE FROM actions WHERE hand_id_ref = ANY(%s)", (db_hand_ids,))

        action_rows = []
        for parsed in records:
            hid = parsed.record.hand_id
            hand_pk = hand_id_to_pk.get(hid)
            if hand_pk is None:
                continue
            for hand_id_txt, action_order, street, actor, action_type, amount, pot_before, is_all_in in _extract_actions_from_block(
                hand_id=hid,
                raw_block=parsed.raw_block,
                site=parsed.site,
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
                    hand_id_ref,
                    action_order,
                    street,
                    actor_id,
                    action_type,
                    amount,
                    pot_size_before,
                    is_all_in
                ) VALUES %s
                ON CONFLICT (hand_id_ref, action_order) DO NOTHING
                """,
                action_rows,
            )

        conn.commit()
        if inserted_hands > 0:
            last_ingested_at = None
            for parsed in records:
                if parsed.record.date_utc:
                    try:
                        last_ingested_at = datetime.strptime(parsed.record.date_utc, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass
            log.info(
                "[WATCHER] file=%s hands_count=%d site=%s hero=%s last_hand_ingested_at=%s",
                filepath.name,
                inserted_hands,
                site,
                hero_name,
                last_ingested_at.isoformat() if last_ingested_at else "unknown",
            )
            if record_ingestion_event is not None:
                try:
                    record_ingestion_event(
                        source_file=filepath.name,
                        inserted_hands=inserted_hands,
                        site=site,
                        hero_name=hero_name,
                        last_hand_ingested_at=last_ingested_at,
                    )
                except Exception as e:
                    log.debug("Falha ao registrar telemetria de ingestao: %s", e)
        
        # Dispara retreino somente quando houve insercao real.
        if ML_AVAILABLE and inserted_hands > 0:
            try:
                trigger_ml_check_after_ingest()
            except Exception as e:
                log.warning("Erro ao trigerar ML: %s", e)
        
        return inserted_hands

    except Exception as e:
        conn.rollback()
        log.error("Erro no upsert de %s: %s", filepath.name, e)
        return 0
    finally:
        cur.close()

# =============================================================================
# Modo batch
# =============================================================================

def run_batch() -> None:
    conn  = get_connection()
    total = 0
    try:
        for entry in WATCH_DIRS:
            watch_path = Path(entry["dir"])
            if not watch_path.exists():
                log.warning("Pasta nao encontrada: %s", watch_path)
                continue
            files = sorted(watch_path.glob(f"**/*{WATCH_EXTENSION}"))
            log.info("[%s] %d arquivos em %s", entry["site"].upper(), len(files), watch_path)
            for f in files:
                total += ingest_file(f, conn, hero_name=entry["hero"], site=entry["site"])
    finally:
        conn.close()
    if total == 0:
        log.info("[WATCHER] Banco sincronizado (Nenhuma mão nova detectada).")
    else:
        log.info("Batch concluido: %d maos no total.", total)

# =============================================================================
# Modo watch
# =============================================================================

class HandHistoryHandler(FileSystemEventHandler):
    def __init__(self, hero_name: str, site: str) -> None:
        self.hero_name = hero_name
        self.site      = site
        self._cache: dict[str, tuple[float, str]] = {}

    def _should_process(self, path: str) -> bool:
        now  = time.time()
        h    = file_hash(Path(path))
        prev = self._cache.get(path)
        if prev and now - prev[0] < COOLDOWN_SECONDS and h == prev[1]:
            return False
        self._cache[path] = (now, h)
        return True

    def _handle(self, path: str) -> None:
        if not path.endswith(WATCH_EXTENSION):
            return
        fpath = Path(path)
        if not fpath.exists() or not self._should_process(path):
            return
        conn = None
        try:
            conn = get_connection()
            ingest_file(fpath, conn, hero_name=self.hero_name, site=self.site)
        except Exception as e:
            log.error("Erro: %s", e)
        finally:
            if conn is not None:
                conn.close()

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle(event.src_path)


def run_watch() -> None:
    log.info("=" * 56)
    log.info("  Poker DSS — Hand History Watcher")
    for e in WATCH_DIRS:
        log.info("  [%s] %s (hero: %s)", e["site"].upper(), e["dir"], e["hero"])
    log.info("=" * 56)

    run_batch()
    log.info("Monitorando novas sessoes...")

    if touch_watcher_heartbeat is not None:
        try:
            touch_watcher_heartbeat(mode="watch")
        except Exception as e:
            log.debug("Falha ao iniciar heartbeat: %s", e)

    observers = []
    for entry in WATCH_DIRS:
        watch_path = Path(entry["dir"])
        if not watch_path.exists():
            continue
        handler  = HandHistoryHandler(hero_name=entry["hero"], site=entry["site"])
        observer = Observer()
        observer.schedule(handler, str(watch_path), recursive=True)
        observer.start()
        observers.append(observer)

    try:
        while True:
            if touch_watcher_heartbeat is not None:
                try:
                    touch_watcher_heartbeat(mode="watch")
                except Exception as e:
                    log.debug("Falha ao atualizar heartbeat: %s", e)
            time.sleep(1)
    except KeyboardInterrupt:
        for obs in observers:
            obs.stop()
        log.info("Watcher encerrado.")
    for obs in observers:
        obs.join()

# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once",     action="store_true")
    ap.add_argument("--password", type=str)
    args = ap.parse_args()

    if args.password:
        DB_CONFIG["password"] = args.password

    if not test_connection():
        log.error("Nao foi possivel conectar ao PostgreSQL.")
        sys.exit(1)
    log.info("Conexao OK.")

    if args.once:
        run_batch()
    else:
        run_watch()
