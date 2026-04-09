"""
hand_history_parser.py
======================
Core Data Manipulation — Roadmap Block 1.

Responsabilidades
-----------------
  ACRHandParser     → lê e parseia arquivos .txt do ACR/WPN
  HandRecord        → estrutura tipada de uma única mão
  ParserPipeline    → orquestra leitura → limpeza → DataFrame → CSV

Formato suportado
-----------------
  ACR Poker / WPN Network (America's Cardroom)
  Linha de cabeçalho: "Game Hand #XXXX - Tournament #XXXX - ..."

Campos extraídos por mão
------------------------
  Identificação    : hand_id, tournament_id, table_id, date_utc
  Estrutura        : level, small_blind, big_blind, ante, max_seats
  Hero             : hero_seat, hero_position, hero_stack_start,
                     hero_cards, hero_vpip, hero_pfr, hero_aggressor
  Ações            : hero_action_preflop, hero_action_flop,
                     hero_action_turn, hero_action_river
  Board            : board_flop, board_turn, board_river
  Resultado        : hero_result, hero_amount_won, hero_stack_end,
                     went_to_showdown, num_players, pot_final
  Torneio          : tournament_name (extraído do nome do arquivo)

Run:  python hand_history_parser.py [pasta_com_txt]
      Sem argumento: usa /mnt/user-data/uploads/
"""

from __future__ import annotations

import os
import re
import sys
import glob
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("parser")

# ── Configuração ─────────────────────────────────────────────────────────────

HERO_NAME: str = "AurelioDizzy"   # ← altere para seu screen name se mudar

# Posições em ordem de assento relativo ao botão (6-max e 8-max)
_POSITION_MAP_8: dict[int, str] = {
    0: "BTN", 1: "SB", 2: "BB",
    3: "UTG", 4: "UTG+1", 5: "MP",
    6: "HJ",  7: "CO",
}
_POSITION_MAP_6: dict[int, str] = {
    0: "BTN", 1: "SB", 2: "BB",
    3: "UTG", 4: "HJ", 5: "CO",
}

# ── Regex pré-compilados (O(1) por uso após compilação) ──────────────────────

_RE_HEADER = re.compile(
    r"Game Hand #(\d+) - Tournament #(\d+) - .+? - Level (\d+) "
    r"\((\d+\.?\d*)/(\d+\.?\d*)\) - (\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) UTC"
)
_RE_TABLE   = re.compile(r"Table '(.+?)' (\d+)-max Seat #(\d+) is the button")
_RE_SEAT    = re.compile(r"Seat (\d+): (\S+) \((\d+\.?\d*)\)")
_RE_ANTE    = re.compile(r"(\S+) posts ante (\d+\.?\d*)")
_RE_BLIND   = re.compile(r"(\S+) posts the (small|big) blind (\d+\.?\d*)")
_RE_DEALT   = re.compile(r"Dealt to " + re.escape(HERO_NAME) + r" \[(.+?)\]")
_RE_ACTION  = re.compile(
    r"(\S+) (folds|checks|calls|bets|raises)(?: (\d+\.?\d*))?(?: to (\d+\.?\d*))?"
    r"(?: and is all-in)?"
)
_RE_BOARD_FLOP  = re.compile(r"\*\*\* FLOP \*\*\* \[(.+?)\]")
_RE_BOARD_TURN  = re.compile(r"\*\*\* TURN \*\*\* \[.+?\] \[(.+?)\]")
_RE_BOARD_RIVER = re.compile(r"\*\*\* RIVER \*\*\* \[.+?\] \[(.+?)\]")
_RE_COLLECTED   = re.compile(
    r"(\S+) collected (\d+\.?\d*) from (main pot|side pot|pot)"
)
_RE_SHOWDOWN_WIN = re.compile(
    r"Seat \d+: " + re.escape(HERO_NAME) + r".*?(?:showed .+? and won|did not show and won) (\d+\.?\d*)"
)
_RE_SHOWDOWN_LOSS = re.compile(
    r"Seat \d+: " + re.escape(HERO_NAME) + r".*?showed .+? and lost"
)
_RE_TOTAL_POT = re.compile(r"Total pot (\d+\.?\d*)")
_RE_ALLIN   = re.compile(r"and is all-in")


# ── Estrutura de dados de uma mão ─────────────────────────────────────────────

@dataclass
class HandRecord:
    # Identificação
    hand_id:          str  = ""
    tournament_id:    str  = ""
    tournament_name:  str  = ""
    table_id:         str  = ""
    date_utc:         str  = ""

    # Estrutura do nível
    level:            int   = 0
    small_blind:      float = 0.0
    big_blind:        float = 0.0
    ante:             float = 0.0
    max_seats:        int   = 0
    num_players:      int   = 0
    btn_seat:         int   = 0

    # Hero
    hero_seat:        int   = 0
    hero_position:    str   = ""
    hero_stack_start: float = 0.0
    hero_stack_end:   float = 0.0
    hero_cards:       str   = ""

    # Ações do hero por rua
    hero_action_preflop: str = ""
    hero_action_flop:    str = ""
    hero_action_turn:    str = ""
    hero_action_river:   str = ""

    # Flags derivadas
    hero_vpip:       int = 0   # 1 se voluntariamente colocou fichas pré-flop
    hero_pfr:        int = 0   # 1 se abriu ou 3-betou pré-flop
    hero_aggressor:  int = 0   # 1 se teve bet/raise em qualquer rua
    hero_went_allin: int = 0   # 1 se foi all-in

    # Board
    board_flop:  str = ""
    board_turn:  str = ""
    board_river: str = ""

    # Resultado
    hero_result:     str   = "fold"   # fold | win | loss | allin_loss
    hero_amount_won: float = 0.0      # fichas líquidas ganhas (pode ser negativo)
    pot_final:       float = 0.0
    went_to_showdown: int  = 0

    # M-ratio (stack / (BB + SB + ante*players)) — útil para análise de push/fold
    m_ratio:         float = 0.0


# ── Parser principal ──────────────────────────────────────────────────────────

class ACRHandParser:
    """
    Parseia um bloco de texto correspondente a UMA mão do ACR.

    Uso:
        parser = ACRHandParser(hero_name="AurelioDizzy")
        record = parser.parse(hand_text)
    """

    def __init__(self, hero_name: str = HERO_NAME) -> None:
        self.hero = hero_name

    def parse(self, raw: str, tournament_name: str = "") -> Optional[HandRecord]:
        """
        Parseia o texto bruto de uma mão e retorna um HandRecord.
        Retorna None se o bloco não contiver cartas do hero (mão não jogada).
        """
        lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if not lines or not lines[0].strip():
            return None
        rec   = HandRecord(tournament_name=tournament_name)

        # ── 1. Cabeçalho ──
        m = _RE_HEADER.search(lines[0])
        if not m:
            return None
        rec.hand_id       = m.group(1)
        rec.tournament_id = m.group(2)
        rec.level         = int(m.group(3))
        rec.small_blind   = float(m.group(4))
        rec.big_blind     = float(m.group(5))
        rec.date_utc      = m.group(6)

        # ── 2. Mesa ──
        if len(lines) > 1:
            m = _RE_TABLE.search(lines[1])
            if m:
                rec.table_id  = m.group(1)
                rec.max_seats = int(m.group(2))
                rec.btn_seat  = int(m.group(3))

        # ── 3. Assentos e stacks ──
        seats: dict[int, tuple[str, float]] = {}  # seat_num → (name, stack)
        for line in lines:
            m = _RE_SEAT.match(line)
            if m:
                seats[int(m.group(1))] = (m.group(2), float(m.group(3)))

        rec.num_players = len(seats)

        if self.hero not in [v[0] for v in seats.values()]:
            return None   # hero não está nesta mão

        hero_seat_num = next(k for k, v in seats.items() if v[0] == self.hero)
        rec.hero_seat        = hero_seat_num
        rec.hero_stack_start = seats[hero_seat_num][1]

        # ── 4. Posição do hero relativa ao botão ──
        rec.hero_position = self._calc_position(
            hero_seat_num, rec.btn_seat,
            sorted(seats.keys()), rec.max_seats
        )

        # ── 5. Ante ──
        for line in lines:
            m = _RE_ANTE.match(line)
            if m and m.group(1) == self.hero:
                rec.ante = float(m.group(2))
                break

        # ── 6. Cartas do hero ──
        for line in lines:
            m = _RE_DEALT.search(line)
            if m:
                rec.hero_cards = m.group(1)
                break

        if not rec.hero_cards:
            return None   # mão sem dealt = hero sentado fora

        # ── 7. Ruas e ações ──
        street        = "preflop"
        hero_actions: dict[str, list[str]] = {
            "preflop": [], "flop": [], "turn": [], "river": []
        }
        in_summary = False

        for line in lines:
            if line.startswith("*** SUMMARY ***"):
                in_summary = True
                continue
            if in_summary:
                continue

            # Detecta mudança de rua
            if "*** HOLE CARDS ***" in line:
                street = "preflop"
            elif "*** FLOP ***" in line:
                street = "flop"
                m = _RE_BOARD_FLOP.search(line)
                if m:
                    rec.board_flop = m.group(1)
            elif "*** TURN ***" in line:
                street = "turn"
                m = _RE_BOARD_TURN.search(line)
                if m:
                    rec.board_turn = m.group(1)
            elif "*** RIVER ***" in line:
                street = "river"
                m = _RE_BOARD_RIVER.search(line)
                if m:
                    rec.board_river = m.group(1)
            elif "*** SHOW DOWN ***" in line:
                rec.went_to_showdown = 1

            # Ações do hero
            m = _RE_ACTION.match(line)
            if m and m.group(1) == self.hero:
                action = m.group(2)
                amount = m.group(4) or m.group(3) or ""
                tag    = action if not amount else f"{action} {amount}"
                if _RE_ALLIN.search(line):
                    tag += " (all-in)"
                    rec.hero_went_allin = 1
                hero_actions[street].append(tag)

        # ── 8. Serializa ações por rua ──
        rec.hero_action_preflop = " → ".join(hero_actions["preflop"])
        rec.hero_action_flop    = " → ".join(hero_actions["flop"])
        rec.hero_action_turn    = " → ".join(hero_actions["turn"])
        rec.hero_action_river   = " → ".join(hero_actions["river"])

        # ── 9. Flags VPIP / PFR / Aggressor ──
        pf_actions = hero_actions["preflop"]
        is_bb = rec.hero_position == "BB"
        is_sb = rec.hero_position == "SB"

        # VPIP: entrou voluntariamente (call ou raise pré-flop, exceto apenas check no BB)
        if any(a.startswith(("calls", "raises")) for a in pf_actions):
            rec.hero_vpip = 1
        elif is_bb and not any(a.startswith(("calls", "raises", "folds")) for a in pf_actions):
            # BB checked/sem ação = não é VPIP
            rec.hero_vpip = 0

        # PFR: abriu, 3-betou ou 4-betou
        if any(a.startswith("raises") for a in pf_actions):
            rec.hero_pfr = 1

        # Aggressor: qualquer bet/raise em qualquer rua
        all_actions = [a for street_acts in hero_actions.values() for a in street_acts]
        if any(a.startswith(("bets", "raises")) for a in all_actions):
            rec.hero_aggressor = 1

        # ── 10. Resultado ──
        pot_m = _RE_TOTAL_POT.search(raw)
        if pot_m:
            rec.pot_final = float(pot_m.group(1))

        # Verifica se hero ganhou (collected ou won no summary)
        won_amount = 0.0
        for line in lines:
            m = _RE_COLLECTED.search(line)
            if m and m.group(1) == self.hero:
                won_amount += float(m.group(2))

        if not won_amount:
            m = _RE_SHOWDOWN_WIN.search(raw)
            if m:
                won_amount = float(m.group(1))

        # Calcula resultado líquido
        invested = rec.ante
        if any(a.startswith("calls") for a in pf_actions):
            # Aproximação: o que foi chamado + cegues
            if is_sb:
                invested += rec.small_blind
            elif is_bb:
                invested += rec.big_blind
            # Calls adicionais são difíceis de rastrear sem parsing completo
            # A camada de stats vai refinar isso
        elif is_sb:
            invested += rec.small_blind
        elif is_bb:
            invested += rec.big_blind

        if won_amount > 0:
            rec.hero_amount_won = won_amount
            rec.hero_result = "win"
        elif _RE_SHOWDOWN_LOSS.search(raw):
            rec.hero_amount_won = -invested
            rec.hero_result = "allin_loss" if rec.hero_went_allin else "loss"
        else:
            rec.hero_amount_won = -invested
            rec.hero_result = "fold"

        # Mantém consistência: stack final deriva do stack inicial + resultado líquido.
        rec.hero_stack_end = rec.hero_stack_start + rec.hero_amount_won

        # ── 11. M-Ratio ──
        orbit_cost = rec.big_blind + rec.small_blind + (rec.ante * rec.num_players)
        if orbit_cost > 0:
            rec.m_ratio = round(rec.hero_stack_start / orbit_cost, 2)

        return rec

    @staticmethod
    def _calc_position(
        hero_seat: int,
        btn_seat:  int,
        active_seats: list[int],
        max_seats: int,
    ) -> str:
        """
        Calcula a posição do hero relativa ao botão.
        Usa os assentos ativos (não todos os slots da mesa).
        """
        n = len(active_seats)
        if n == 0:
            return "?"

        # Ordena assentos ativos a partir do botão
        btn_idx   = active_seats.index(btn_seat) if btn_seat in active_seats \
                    else min(range(n), key=lambda i: (active_seats[i] - btn_seat) % max_seats)

        # Posição relativa: 0 = BTN, 1 = SB, 2 = BB, 3 = UTG ...
        hero_idx = active_seats.index(hero_seat) if hero_seat in active_seats else 0
        rel_pos  = (hero_idx - btn_idx) % n

        pos_map = _POSITION_MAP_6 if n <= 6 else _POSITION_MAP_8
        return pos_map.get(rel_pos, f"MP{rel_pos - 3}")


# ── Pipeline completo ─────────────────────────────────────────────────────────

class ParserPipeline:
    """
    Orquestra: leitura de arquivos → parse → limpeza → DataFrame → CSV.

    Uso:
        pipeline = ParserPipeline(hero_name="AurelioDizzy")
        df = pipeline.run(input_dir="./hand_histories/")
        pipeline.export(df, output_path="hands.csv")
    """

    def __init__(self, hero_name: str = HERO_NAME) -> None:
        self.parser = ACRHandParser(hero_name=hero_name)

    def run(self, input_dir: str) -> pd.DataFrame:
        """
        Lê todos os .txt em input_dir e retorna DataFrame limpo.
        """
        txt_files = glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
        if not txt_files:
            log.warning("Nenhum arquivo .txt encontrado em: %s", input_dir)
            return pd.DataFrame()

        records: list[dict] = []
        total_hands = parsed = skipped = 0

        for filepath in sorted(txt_files):
            tournament_name = self._extract_tournament_name(filepath)
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError as e:
                log.error("Erro lendo %s: %s", filepath, e)
                continue

            # Divide em blocos de mão individuais
            hand_blocks = re.split(
                r"(?=Game Hand #\d+ - Tournament)", content
            )

            for block in hand_blocks:
                block = block.strip()
                if not block:
                    continue
                total_hands += 1
                try:
                    rec = self.parser.parse(block, tournament_name)
                    if rec:
                        records.append(asdict(rec))
                        parsed += 1
                    else:
                        skipped += 1
                except Exception as e:
                    log.debug("Erro parseando mão: %s", e)
                    skipped += 1

            log.info(
                "✓ %s  (%d mãos)",
                os.path.basename(filepath),
                len(re.findall(r"Game Hand #\d+", content)),
            )

        log.info(
            "Total: %d blocos | %d parseados | %d sem hero / ignorados",
            total_hands, parsed, skipped,
        )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = self._clean(df)
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpeza e otimização de tipos — Pandas best practices.
        """
        # Tipos numéricos
        float_cols = [
            "small_blind", "big_blind", "ante",
            "hero_stack_start", "hero_amount_won", "pot_final", "m_ratio",
        ]
        int_cols = [
            "level", "max_seats", "num_players", "btn_seat",
            "hero_seat", "hero_vpip", "hero_pfr", "hero_aggressor",
            "hero_went_allin", "went_to_showdown",
        ]
        for c in float_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")

        # Datetime
        if "date_utc" in df.columns:
            df["date_utc"] = pd.to_datetime(df["date_utc"], format="%Y/%m/%d %H:%M:%S", errors="coerce")

        # Categoricals — reduz memória ~4x em colunas de baixa cardinalidade
        cat_cols = ["hero_position", "hero_result", "tournament_name"]
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype("category")

        # Ordena por data
        if "date_utc" in df.columns:
            df = df.sort_values("date_utc").reset_index(drop=True)

        # Remove duplicatas (mesmo hand_id em múltiplos arquivos)
        if "hand_id" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset="hand_id").reset_index(drop=True)
            if len(df) < before:
                log.info("Removidas %d mãos duplicadas", before - len(df))

        log.info(
            "DataFrame final: %d mãos | %d colunas | %.1f KB",
            len(df), len(df.columns),
            df.memory_usage(deep=True).sum() / 1024,
        )
        return df

    def export(
        self,
        df: pd.DataFrame,
        output_path: str = "hands.csv",
        also_parquet: bool = True,
    ) -> None:
        """
        Exporta o DataFrame para CSV e opcionalmente Parquet.
        Parquet é ~5x menor e carrega 10x mais rápido que CSV.
        """
        if df.empty:
            log.warning("DataFrame vazio — nada exportado.")
            return

        df.to_csv(output_path, index=False)
        log.info("CSV exportado: %s  (%d linhas)", output_path, len(df))

        if also_parquet:
            try:
                parquet_path = output_path.replace(".csv", ".parquet")
                df.to_parquet(parquet_path, index=False)
                log.info("Parquet exportado: %s", parquet_path)
            except ImportError:
                log.info("pyarrow não instalado — apenas CSV exportado.")

    @staticmethod
    def _extract_tournament_name(filepath: str) -> str:
        """Extrai o nome do torneio do nome do arquivo."""
        name = os.path.basename(filepath)
        m = re.search(r"TN-(.+?)_GAMETYPE", name)
        if m:
            return m.group(1).replace("_", " ").replace("-", " ").strip()
        return name.split("_TN-")[-1].split("_")[0] if "_TN-" in name else ""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    input_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads"

    print("=" * 60)
    print(f"  ACR Hand History Parser")
    print(f"  Hero: {HERO_NAME}")
    print(f"  Diretório: {input_dir}")
    print("=" * 60)

    pipeline = ParserPipeline(hero_name=HERO_NAME)
    df = pipeline.run(input_dir=input_dir)

    if df.empty:
        print("Nenhuma mão parseada.")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  Mãos parseadas : {len(df)}")
    print(f"  Colunas        : {len(df.columns)}")
    print(f"\n  Primeiras 3 mãos:")
    print(df[["hand_id", "date_utc", "hero_position", "hero_cards",
              "hero_action_preflop", "hero_result", "hero_vpip",
              "hero_pfr", "m_ratio"]].head(3).to_string(index=False))
    print(f"\n  Distribuição de posições:")
    print(df["hero_position"].value_counts().to_string())
    print(f"\n  Distribuição de resultados:")
    print(df["hero_result"].value_counts().to_string())
    print("=" * 60)

    # Exporta
    out = os.path.join("/mnt/user-data/outputs", "hands.csv")
    pipeline.export(df, output_path=out)
