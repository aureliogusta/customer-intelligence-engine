"""
study_agent.py — Agente de estudo baseado nos dados do Postgres (poker_dss).

Pipeline:
  Postgres (hands/sessions) → análise matemática → Ollama streaming → terminal

Uso:
  python study_agent.py
  python study_agent.py --sessoes 20       (analisa últimas N sessões)
  python study_agent.py --posicao BB       (foca em posição específica)
  python study_agent.py --maos-erradas     (identifica mãos mal jogadas + explicação)
  python study_agent.py --maos-erradas --sessoes 5  (só últimas 5 sessões)
"""

from __future__ import annotations

import sys
import json
import os
import urllib.request
from pathlib import Path

# Fix encoding no Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Range manager do projeto statistics (GTO ranges)
sys.path.insert(0, r"C:\projeto-spade\statistics")
try:
    from range_manager import (
        classify_stack, is_in_range, get_push_range,
        StackRange, _LOW_RANGES, _MID_RANGES, _HIGH_RANGES, _RANGE_MAP,
    )
    _RANGES_OK = True
except ImportError:
    _RANGES_OK = False

# ─── Config ──────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("PG_HOST",     "localhost"),
    "port":     int(os.getenv("PG_PORT", "5432")),
    "database": os.getenv("PG_DB",       "poker_dss"),
    "user":     os.getenv("PG_USER",     "postgres"),
    "password": os.getenv("PG_PASSWORD", "aurelio"),
}

OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("POKER_LLM_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = 60  # segundos — análise é mais longa que decisão em jogo


# ─── Conexão ─────────────────────────────────────────────────────────────────

def _connect():
    try:
        import psycopg2
    except ImportError:
        print("[StudyAgent] ERRO: psycopg2 não instalado — pip install psycopg2-binary")
        sys.exit(1)
    return psycopg2.connect(**DB_CONFIG)


# ─── Coleta de dados ─────────────────────────────────────────────────────────

def _fetch_stats(n_sessions: int = 0, posicao: str = "") -> dict:
    conn = _connect()
    cur  = conn.cursor()

    def q(sql, params=None):
        cur.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── 1. Resumo geral ──────────────────────────────────────────────────────
    cur.execute("""
        SELECT
            COUNT(*)                                                      AS total_maos,
            ROUND(AVG(hero_vpip)*100, 1)                                  AS vpip,
            ROUND(AVG(hero_pfr)*100, 1)                                   AS pfr,
            ROUND(AVG(hero_aggressor)*100, 1)                             AS agg,
            ROUND(AVG(went_to_showdown)*100, 1)                           AS wtsd,
            ROUND(SUM(hero_amount_won / NULLIF(big_blind,0))
                  / NULLIF(COUNT(*),0) * 100, 2)                          AS bb_per_100
        FROM hands WHERE big_blind > 0
    """)
    geral = dict(zip([d[0] for d in cur.description], cur.fetchone()))

    # ── 2. Stats por posição ─────────────────────────────────────────────────
    posicoes = q("""
        SELECT
            hero_position                                                  AS posicao,
            COUNT(*)                                                       AS maos,
            ROUND(AVG(hero_vpip)*100, 1)                                   AS vpip,
            ROUND(AVG(hero_pfr)*100, 1)                                    AS pfr,
            ROUND(AVG(hero_aggressor)*100, 1)                              AS agg,
            ROUND(AVG(went_to_showdown)*100, 1)                            AS wtsd,
            ROUND(COUNT(*) FILTER (WHERE hero_vpip=0)::numeric
                  / COUNT(*) * 100, 1)                                     AS fold_pct,
            ROUND(SUM(hero_amount_won / NULLIF(big_blind,0))
                  / NULLIF(COUNT(*),0) * 100, 2)                           AS bb_per_100
        FROM hands
        WHERE big_blind > 0
          AND (%(pos)s = '' OR hero_position = %(pos)s)
        GROUP BY hero_position
        ORDER BY maos DESC
    """, {"pos": posicao.upper()})

    # ── 3. Últimas sessões ───────────────────────────────────────────────────
    sessoes_sql = """
        SELECT
            s.ingested_at::DATE                                            AS data,
            s.hands_count                                                  AS maos,
            ROUND(s.net_chips, 0)                                          AS net_chips,
            ROUND(AVG(h.hero_vpip)*100, 1)                                 AS vpip,
            ROUND(AVG(h.hero_pfr)*100, 1)                                  AS pfr,
            ROUND(SUM(h.hero_amount_won / NULLIF(h.big_blind,0))
                  / NULLIF(s.hands_count,0) * 100, 1)                      AS bb100
        FROM sessions s
        JOIN hands h ON h.session_id = s.session_id
        WHERE h.big_blind > 0
        GROUP BY s.session_id, s.ingested_at, s.hands_count, s.net_chips
        ORDER BY s.ingested_at DESC
    """
    if n_sessions > 0:
        sessoes_sql += f" LIMIT {n_sessions}"
    sessoes = q(sessoes_sql)

    # ── 4. Tendência recente vs geral ─────────────────────────────────────────
    cur.execute("""
        SELECT
            ROUND(SUM(h.hero_amount_won / NULLIF(h.big_blind,0))
                  / NULLIF(COUNT(h.*),0) * 100, 2)                         AS bb100_recente,
            ROUND(AVG(h.hero_vpip)*100, 1)                                  AS vpip_recente
        FROM (
            SELECT session_id FROM sessions ORDER BY ingested_at DESC LIMIT 10
        ) s
        JOIN hands h ON h.session_id = s.session_id
        WHERE h.big_blind > 0
    """)
    recente = dict(zip([d[0] for d in cur.description], cur.fetchone()))

    # ── 5. Padrão de VPIP/PFR por street ─────────────────────────────────────
    cur.execute("""
        SELECT
            ROUND(AVG(CASE WHEN hero_vpip=1 AND hero_pfr=0 THEN 1.0 ELSE 0.0 END)*100,1) AS call_only_pct,
            ROUND(AVG(CASE WHEN hero_action_flop IS NOT NULL
                           AND hero_action_flop != '' THEN 1.0 ELSE 0.0 END)*100,1)       AS viu_flop_pct,
            ROUND(AVG(CASE WHEN went_to_showdown=1
                           AND hero_result='loss' THEN 1.0 ELSE 0.0 END)*100,1)           AS wtsd_loss_pct
        FROM hands WHERE big_blind > 0
    """)
    streets = dict(zip([d[0] for d in cur.description], cur.fetchone()))

    conn.close()
    return {
        "geral":    geral,
        "posicoes": posicoes,
        "sessoes":  sessoes,
        "recente":  recente,
        "streets":  streets,
    }


# ─── Formatação do contexto ───────────────────────────────────────────────────

def _build_context(data: dict, posicao: str = "") -> str:
    g  = data["geral"]
    r  = data["recente"]
    s  = data["streets"]
    bb = g["bb_per_100"]

    # Identifica posições problemáticas
    pos_negativas = [p for p in data["posicoes"] if p["bb_per_100"] and p["bb_per_100"] < 0]
    pos_baixas    = [p for p in data["posicoes"]
                    if p["bb_per_100"] and 0 <= p["bb_per_100"] < 50 and p["maos"] >= 30]

    lines = [
        "RELATÓRIO DE PERFORMANCE — POKER DSS",
        f"Total de mãos analisadas: {g['total_maos']}",
        "",
        "── MÉTRICAS GERAIS ─────────────────────────────",
        f"  BB/100 geral    : {bb:+.2f}",
        f"  BB/100 recente  : {r['bb100_recente']:+.2f}  (últimas 10 sessões)",
        f"  VPIP            : {g['vpip']}%  (referência MTT: 18–24%)",
        f"  PFR             : {g['pfr']}%   (referência MTT: 14–20%)",
        f"  VPIP/PFR gap    : {round(float(g['vpip'] or 0) - float(g['pfr'] or 0), 1)} pp"
        f"  (ideal < 4 pp — gap alto = jogo passivo)",
        f"  Agressividade   : {g['agg']}%",
        f"  WTSD            : {g['wtsd']}%  (referência MTT: 25–33%)",
        f"  Call sem raise  : {s['call_only_pct']}%  (entradas passivas, sem iniciativa)",
        f"  Viu flop        : {s['viu_pct'] if 'viu_pct' in s else s['viu_flop_pct']}%",
        f"  Perdeu showdown : {s['wtsd_loss_pct']}%",
        "",
        "── BB/100 POR POSIÇÃO ───────────────────────────",
    ]

    for p in sorted(data["posicoes"], key=lambda x: x["bb_per_100"] or 0):
        flag = ""
        if (p["bb_per_100"] or 0) < 0:
            flag = " ◄ DÉFICIT"
        elif (p["bb_per_100"] or 0) < 50 and (p["maos"] or 0) >= 30:
            flag = " ◄ ABAIXO DO ESPERADO"
        lines.append(
            f"  {p['posicao']:6s} | {p['maos']:4d} mãos | "
            f"VPIP {p['vpip']}% PFR {p['pfr']}% "
            f"Fold {p['fold_pct']}% WTSD {p['wtsd']}% | "
            f"{p['bb_per_100']:+.1f} bb/100{flag}"
        )

    if pos_negativas:
        lines += ["", "── POSIÇÕES EM DÉFICIT ──────────────────────────"]
        for p in pos_negativas:
            lines.append(f"  {p['posicao']}: {p['bb_per_100']:+.1f} bb/100 "
                         f"(Fold {p['fold_pct']}%, PFR {p['pfr']}%, WTSD {p['wtsd']}%)")

    lines += ["", "── ÚLTIMAS SESSÕES ──────────────────────────────"]
    for s in data["sessoes"][:10]:
        sinal = "+" if (s["bb100"] or 0) >= 0 else ""
        lines.append(
            f"  {s['data']} | {s['maos']:3d} mãos | "
            f"VPIP {s['vpip']}% PFR {s['pfr']}% | "
            f"{sinal}{s['bb100']} bb/100"
        )

    if posicao:
        lines += ["", f"── FOCO: {posicao} ──────────────────────────────"]
        alvo = next((p for p in data["posicoes"]
                     if p["posicao"] == posicao.upper()), None)
        if alvo:
            for k, v in alvo.items():
                lines.append(f"  {k}: {v}")

    return "\n".join(lines)


# ─── Identificação de mãos mal jogadas ───────────────────────────────────────

_RANKS = "AKQJT98765432"
_R2I   = {r: i for i, r in enumerate(_RANKS)}


def _canonicalize(raw_cards: str) -> str:
    """'Ad 6d' → 'A6s'  |  'Kh 3c' → 'K3o'"""
    parts = raw_cards.strip().split()
    if len(parts) != 2:
        return ""
    try:
        r1, s1 = parts[0][0].upper(), parts[0][1].lower()
        r2, s2 = parts[1][0].upper(), parts[1][1].lower()
    except IndexError:
        return ""
    if r1 not in _R2I or r2 not in _R2I:
        return ""
    # Garante ordem decrescente de rank
    if _R2I[r1] > _R2I[r2]:
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return f"{r1}{r2}"
    suit = "s" if s1 == s2 else "o"
    return f"{r1}{r2}{suit}"


def _action_type(action_str: str) -> str:
    """Extrai tipo principal da ação: fold, call, raise, check, allin."""
    if not action_str:
        return ""
    a = action_str.lower()
    if "all-in" in a or "all in" in a:
        return "allin"
    if "raises" in a or "bets" in a:
        return "raise"
    if "calls" in a:
        return "call"
    if "checks" in a:
        return "check"
    if "folds" in a or "desiste" in a:
        return "fold"
    return ""


def _pot_odds_bb_defense(pot_before_bb: float, call_bb: float) -> float:
    """Retorna equity mínima necessária para defender (pot odds)."""
    total = pot_before_bb + call_bb
    if total <= 0:
        return 0.33
    return call_bb / total


_MISTAKE_SYSTEM = """Você é um coach de poker MTT especializado em revisão de mãos.
Para cada mão listada, explique em 2-3 linhas:
- Por que foi um erro (em termos de GTO/EV)
- O que deveria ter sido feito
- Conceito-chave envolvido (ex: fold equity, pot odds, ICM, range disadvantage)

Seja direto e técnico. Use os dados da mão. Sem introdução, vá direto para cada caso."""


def _fetch_wrong_hands(n_sessions: int = 0) -> list[dict]:
    """
    Busca mãos candidatas a erro e classifica por tipo de problema.

    Tipos detectados:
      FOLD_GTO_RANGE   — foldou mão que está no range GTO para posição/stack
      LIMP_INITIATIVE  — entrou sem iniciativa (call) quando deveria raise
      BB_FOLD_CHEAP    — foldou BB pagando < 33% do pot (pot odds favorecem defesa)
      PASSIVE_POSTFLOP — viu flop mas não teve agressão no flop/turn
      PUSH_FOLD_MISS   — stack < 10bb e foldou mão de push (M ratio baixo)
    """
    if not _RANGES_OK:
        print("[StudyAgent] range_manager não disponível — análise de mãos desabilitada.")
        return []

    conn = _connect()
    cur  = conn.cursor()

    session_filter = ""
    if n_sessions > 0:
        session_filter = f"""
            AND session_id IN (
                SELECT session_id FROM sessions
                ORDER BY ingested_at DESC LIMIT {n_sessions}
            )
        """

    cur.execute(f"""
        SELECT
            hand_id, date_utc, hero_position, hero_cards,
            hero_stack_start, big_blind, m_ratio,
            hero_vpip, hero_pfr, hero_aggressor,
            hero_action_preflop, hero_action_flop, hero_action_turn,
            hero_result, hero_amount_won, pot_final
        FROM hands
        WHERE hero_cards IS NOT NULL AND hero_cards != ''
          AND big_blind > 0
          {session_filter}
        ORDER BY date_utc DESC
        LIMIT 500
    """)
    cols  = [d[0] for d in cur.description]
    rows  = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()

    mistakes = []

    for h in rows:
        canon    = _canonicalize(h["hero_cards"])
        if not canon:
            continue

        pos      = (h["hero_position"] or "BTN").upper()
        stack_bb = float(h["hero_stack_start"] or 0) / float(h["big_blind"] or 1)
        m_ratio  = float(h["m_ratio"] or 99)
        bb_val   = float(h["big_blind"] or 1)
        pot      = float(h["pot_final"] or 0)
        won      = float(h["hero_amount_won"] or 0)

        pf_action  = _action_type(h["hero_action_preflop"] or "")
        fl_action  = _action_type(h["hero_action_flop"] or "")

        sr         = classify_stack(stack_bb)
        gto_range  = _RANGE_MAP[sr].get(pos, frozenset())
        in_range   = canon in gto_range

        erro = None

        # ── 1. Foldou mão que está no range GTO ──────────────────────────────
        if pf_action == "fold" and in_range and m_ratio > 3:
            erro = {
                "tipo":    "FOLD_GTO_RANGE",
                "titulo":  f"Fold com {canon} no range GTO ({pos}, {sr.value})",
                "detalhe": (
                    f"Mao: {h['hero_cards']} ({canon}) | Posicao: {pos} | "
                    f"Stack: {stack_bb:.1f}bb [{sr.value}] | M={m_ratio:.1f}\n"
                    f"Acao: fold | {canon} esta no range GTO de {pos} para stack {sr.value}.\n"
                    f"Range GTO tem {len(gto_range)} combos nessa situacao."
                ),
            }

        # ── 2. Entrou sem iniciativa quando poderia raise (limp/call) ─────────
        elif (h["hero_vpip"] == 1 and h["hero_pfr"] == 0
              and pos not in ("BB",) and m_ratio > 8):
            erro = {
                "tipo":    "LIMP_INITIATIVE",
                "titulo":  f"Call passivo com {canon} ({pos}) sem iniciativa",
                "detalhe": (
                    f"Mao: {h['hero_cards']} ({canon}) | Posicao: {pos} | "
                    f"Stack: {stack_bb:.1f}bb [{sr.value}] | M={m_ratio:.1f}\n"
                    f"Acao: call sem raise | Entrou passivo sem tomar iniciativa.\n"
                    f"In GTO range: {'SIM' if in_range else 'NAO — mao fora do range'}."
                ),
            }

        # ── 3. BB foldou pagando pouco (pot odds favoráveis) ─────────────────
        elif (pos == "BB" and pf_action == "fold"
              and h["hero_action_preflop"] and bb_val > 0):
            raw_pf = (h["hero_action_preflop"] or "").lower()
            # Tenta extrair valor do raise do oponente a partir do pot_final
            if pot > 0 and pot < bb_val * 6:  # raise pequeno (< 3x)
                call_bb = pot / bb_val / 2    # estimativa
                needed  = _pot_odds_bb_defense(pot / bb_val, call_bb)
                if needed < 0.30:             # precisava de < 30% equity
                    erro = {
                        "tipo":    "BB_FOLD_CHEAP",
                        "titulo":  f"BB foldou {canon} pagando preco baixo",
                        "detalhe": (
                            f"Mao: {h['hero_cards']} ({canon}) | Posicao: BB | "
                            f"Stack: {stack_bb:.1f}bb | M={m_ratio:.1f}\n"
                            f"Pot: {pot/bb_val:.1f}bb | Equity minima necessaria: ~{needed*100:.0f}%.\n"
                            f"Defender BB com {canon} provavelmente tem equity suficiente."
                        ),
                    }

        # ── 4. Passivo no flop — viu flop mas nao agrediu ─────────────────────
        elif (h["hero_vpip"] == 1 and h["hero_action_flop"]
              and fl_action in ("check", "call")
              and h["hero_aggressor"] == 0 and stack_bb > 15):
            erro = {
                "tipo":    "PASSIVE_POSTFLOP",
                "titulo":  f"Passivo no flop com {canon} ({pos})",
                "detalhe": (
                    f"Mao: {h['hero_cards']} ({canon}) | Posicao: {pos} | "
                    f"Stack: {stack_bb:.1f}bb [{sr.value}]\n"
                    f"Preflop: {h['hero_action_preflop']} | Flop: {h['hero_action_flop']}\n"
                    f"Nenhuma agressao no flop — check/call passivo."
                ),
            }

        # ── 5. Stack curto e foldou mao de push ──────────────────────────────
        elif (pf_action == "fold" and m_ratio <= 7 and in_range):
            erro = {
                "tipo":    "PUSH_FOLD_MISS",
                "titulo":  f"Push/fold: deveria ter shoved {canon} ({pos}, M={m_ratio:.1f})",
                "detalhe": (
                    f"Mao: {h['hero_cards']} ({canon}) | Posicao: {pos} | "
                    f"Stack: {stack_bb:.1f}bb | M={m_ratio:.1f}\n"
                    f"Com M<7, push/fold puro. {canon} esta no range de push de {pos}.\n"
                    f"Fold aqui perde fold equity e desperdicou stack."
                ),
            }

        if erro:
            erro["hand_id"] = h["hand_id"]
            erro["data"]    = str(h["date_utc"])[:10] if h["date_utc"] else ""
            erro["won"]     = won
            mistakes.append(erro)

    # Ordena por impacto: erros com maior perda primeiro
    mistakes.sort(key=lambda x: x["won"])
    return mistakes


def _format_wrong_hands(mistakes: list[dict]) -> str:
    if not mistakes:
        return "Nenhuma mao errada identificada nas sessoes analisadas."

    # Agrupa por tipo para resumo
    from collections import Counter
    contagem = Counter(m["tipo"] for m in mistakes)

    lines = [
        f"MAOS MAL JOGADAS — {len(mistakes)} casos identificados",
        "",
        "RESUMO POR TIPO:",
    ]
    tipo_label = {
        "FOLD_GTO_RANGE":   "Fold dentro do range GTO",
        "LIMP_INITIATIVE":  "Call passivo sem iniciativa",
        "BB_FOLD_CHEAP":    "BB foldou pagando pouco",
        "PASSIVE_POSTFLOP": "Passivo no flop/turn",
        "PUSH_FOLD_MISS":   "Deveria ter shoved (M baixo)",
    }
    for tipo, n in contagem.most_common():
        lines.append(f"  {tipo_label.get(tipo, tipo):40s} {n:3d} casos")

    lines += ["", "DETALHES (ordenado por prejuizo):"]
    for i, m in enumerate(mistakes[:15], 1):  # limita a 15 para o prompt
        lines.append(f"\n[{i}] {m['titulo']}  ({m['data']})  won={m['won']:+.0f}")
        lines.append(f"    {m['detalhe']}")

    return "\n".join(lines)


def _stream_wrong_hands_analysis(wrong_hands_text: str) -> None:
    """Envia as mãos erradas ao Llama para explicação pedagógica."""
    payload = json.dumps({
        "model":    OLLAMA_MODEL,
        "messages": [
            {"role": "system",  "content": _MISTAKE_SYSTEM},
            {"role": "user",    "content": wrong_hands_text},
        ],
        "stream":  True,
        "options": {
            "temperature": 0.3,
            "num_predict": 700,
            "stop": ["---"],
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj   = json.loads(line)
                    token = obj.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                    if obj.get("done"):
                        break
                except json.JSONDecodeError:
                    pass
        print()
    except urllib.error.URLError as exc:
        print(f"\n[StudyAgent] Ollama indisponível ({exc}).")


# ─── Ollama streaming ─────────────────────────────────────────────────────────

_SYSTEM = """Você é um coach de poker especializado em MTT (torneios).
Analise os dados de performance abaixo e produza um relatório de estudo estruturado com:

1. DIAGNÓSTICO PRINCIPAL — o maior problema identificado nos dados (seja específico)
2. TOP 3 LEAKS — os 3 erros com maior impacto no BB/100, em ordem de prioridade
3. PLANO DE ESTUDO — para cada leak, 1-2 ações concretas de estudo/correção
4. MÉTRICAS ALVO — onde cada indicador deveria estar após correção

Seja direto, use os números dos dados, não generalize. Máximo 350 palavras."""


def _stream_ollama(context: str) -> None:
    payload = json.dumps({
        "model":    OLLAMA_MODEL,
        "messages": [
            {"role": "system",  "content": _SYSTEM},
            {"role": "user",    "content": context},
        ],
        "stream":  True,
        "options": {
            "temperature": 0.3,
            "num_predict": 500,
            "stop": ["---"],
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj   = json.loads(line)
                    token = obj.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                    if obj.get("done"):
                        break
                except json.JSONDecodeError:
                    pass
        print()
    except urllib.error.URLError as exc:
        print(f"\n[StudyAgent] Ollama indisponível ({exc}).")
        print("  → Execute: ollama serve")
        print("  → Modelo : ollama pull llama3.2:3b")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    n_sessoes   = 0
    posicao     = ""
    maos_erradas = "--maos-erradas" in args

    for i, a in enumerate(args):
        if a == "--sessoes" and i + 1 < len(args):
            n_sessoes = int(args[i + 1])
        if a == "--posicao" and i + 1 < len(args):
            posicao = args[i + 1].upper()

    print()
    print("══════════════════════════════════════════════════════")
    print("  STUDY AGENT — Poker DSS")
    print("══════════════════════════════════════════════════════")
    print("  Conectando ao banco de dados...")

    if maos_erradas:
        # ── Modo: mãos erradas ───────────────────────────────────────────────
        print("  Identificando mãos mal jogadas...\n")
        mistakes = _fetch_wrong_hands(n_sessions=n_sessoes)
        wrong_text = _format_wrong_hands(mistakes)

        print(wrong_text)
        print()
        print("══════════════════════════════════════════════════════")
        print("  EXPLICAÇÃO DO COACH (Llama)")
        print("══════════════════════════════════════════════════════")
        print()
        _stream_wrong_hands_analysis(wrong_text)
    else:
        # ── Modo padrão: análise geral ───────────────────────────────────────
        data    = _fetch_stats(n_sessions=n_sessoes, posicao=posicao)
        context = _build_context(data, posicao=posicao)

        print()
        print(context)
        print()
        print("══════════════════════════════════════════════════════")
        print("  ANÁLISE DO COACH (Llama)")
        print("══════════════════════════════════════════════════════")
        print()
        _stream_ollama(context)

    print()
    print("══════════════════════════════════════════════════════")
    print()


if __name__ == "__main__":
    main()
