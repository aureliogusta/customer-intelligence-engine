"""
websocket_server.py
===================
Servidor WebSocket de produção — Poker DSS.

Roadmap blocks implementados aqui
----------------------------------
  ✅ Real Time Integration   — WebSocket persistente, sem overhead HTTP
  ✅ Latency Optimization    — Dual mode MC, pool de conexões, compressão off
  ✅ Deployment Architecture — health check, graceful shutdown, reload config
  ✅ Data Compliance         — audit trail JSON, sessão por cliente, PII-free logs

Protocolo
---------
  Cliente → Servidor : string de tokens separados por espaço
    Formato base:  "MÃO POSIÇÃO OPEN_BB STACK_BB MULTIWAY [CARTAS...]"
    Modo review:   "REVIEW MÃO POSIÇÃO OPEN_BB STACK_BB MULTIWAY [CARTAS...]"
    Ping:          "PING"
    Status:        "STATUS"

  Servidor → Cliente : JSON estruturado
    {
      "decision":    str,    "BET/RAISE (...)"
      "ev":          str,    "+2.10" | "—"
      "color_code":  str,    "GREEN_INTENSE" | "RED_FOLD" | "PURPLE_ALLIN" |
                             "GRAY_ERROR"
      "equity":      str,    "68.0%"
      "hand_label":  str,    "Flush" | ""
      "tier":        str,    "Tier 2"
      "has_board":   bool
      "latency_ms":  float
      "mc_mode":     str,    "fast" | "deep" | "preflop"
      "mc_n":        int,    número de iterações MC usadas
      "mc_ci":       str,    "[62.1%–74.3%]" | ""   intervalo de confiança 95%
      "mc_error_pp": float   margem de erro em pp
      "session_id":  str     identificador da sessão (para audit trail)
    }

Modo Dual MC
------------
  fast  (padrão durante o grind):
    N=600, ±3pp, ~18ms — decisão rápida sob pressão de timebank

  deep  (modo review, flag "REVIEW" no início da string):
    N=20000, multiprocessing com todos os cores disponíveis
    ±0.5pp, ~300ms — análise pós-sessão com precisão máxima

Deployment
----------
  Iniciar:          python websocket_server.py
  Com port custom:  python websocket_server.py --port 8001
  Health check:     curl http://localhost:8000/health   (via HTTP separado)
  Graceful stop:    Ctrl+C  (fecha conexões abertas antes de sair)

Data Compliance
---------------
  Audit trail em: ./logs/audit_<DATA>.jsonl
  Cada evento registra: session_id, timestamp_utc, hand (PII-free),
  position, decision, color_code, latency_ms
  PII: screen names e valores de pot NÃO são logados.

Instalação:
  pip install websockets --break-system-packages

"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import time
import logging
import uuid
import argparse
import multiprocessing
import re
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_BASE_DIR))

import websockets

from decision_engine import evaluate_action, DECISION_POLICY_VERSION
from monte_carlo_engine import ITERATIONS_PRESETS
from range_manager import analyze_vs_3bet, calc_open_raise_chips, classify_stack
from math_validator import MathValidator

# ── Configuração ──────────────────────────────────────────────────────────────

_HOST        = "0.0.0.0"
_PORT        = 8000
_LOG_DIR     = _BASE_DIR / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_AUDIT_RETENTION_DAYS = 30

# Número de cores para modo deep
_N_CORES = max(1, multiprocessing.cpu_count() - 1)   # deixa 1 core livre para OS

# Iterações por modo
_FAST_N  = ITERATIONS_PRESETS["fast"]       # 600  — durante o grind
_DEEP_N  = ITERATIONS_PRESETS["deep"]       # 20000 — modo review
RESPONSE_SCHEMA_VERSION = "1.1.0"

# ── Logging estruturado ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("poker_ws")

# Audit trail — JSONL separado para não misturar com logs operacionais
_audit_path = _LOG_DIR / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
_audit_fh   = open(_audit_path, "a", encoding="utf-8", buffering=1)  # line-buffered


def _cleanup_old_audit_logs() -> None:
    """Remove arquivos de audit antigos para evitar crescimento indefinido em disco."""
    now = datetime.now(timezone.utc)
    for p in _LOG_DIR.glob("audit_*.jsonl"):
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            age_days = (now - mtime).days
            if age_days > _AUDIT_RETENTION_DAYS:
                p.unlink(missing_ok=True)
        except OSError:
            # Falha de limpeza nunca deve derrubar o servidor.
            continue

def _audit(event: dict) -> None:
    """
    Registra evento no audit trail (JSONL).
    PII-free: não loga screen names, valores de pot ou stack absoluto.
    Loga apenas: session, timestamp, mão canônica, posição, decisão, cor, latência.
    """
    entry = {
        "ts":        datetime.now(timezone.utc).isoformat(),
        "session":   event.get("session_id", "?"),
        "hand":      event.get("hand_canonical", event.get("hand", "?")),
        "position":  event.get("position", "?"),
        "has_board": event.get("has_board", False),
        "decision":  event.get("decision", "?")[:60],
        "color":     event.get("color_code", "?"),
        "mc_mode":   event.get("mc_mode", "?"),
        "latency":   round(event.get("latency_ms", 0), 2),
    }
    try:
        _audit_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass   # audit nunca quebra o servidor


# ── Estatísticas de sessão (em memória, sem PII) ──────────────────────────────

# ── Contexto persistente de sessão ────────────────────────────────────────────

class SessionContext:
    """
    Estado persistente por conexão WebSocket.
    Sobrevive entre múltiplas mensagens de análise.

    Configurável via comando:  SET SB=800 BB=1200 ANTE=200
                               SET OPEN=2.5
    """
    __slots__ = ("sb", "bb", "ante", "open_mult")

    def __init__(self) -> None:
        self.sb        = 0.0   # small blind em fichas (0 = não configurado)
        self.bb        = 0.0   # big blind em fichas
        self.ante      = 0.0   # ante em fichas
        self.open_mult = 0.0   # multiplicador open raise (0 = auto por posição)

    def update_from_tokens(self, tokens: list[str]) -> list[str]:
        """
        Parseia tokens do comando SET.
        Formato: SET SB=800 BB=1200 ANTE=200  ou  SET OPEN=2.5
        Retorna lista de erros (vazia = ok).
        """
        errors = []
        for token in tokens:
            if "=" not in token:
                continue
            key, _, val = token.partition("=")
            key = key.upper().strip()
            try:
                v = float(val.strip())
            except ValueError:
                errors.append(f"Valor inválido para {key}: '{val}'")
                continue

            if key in ("SB", "SMALL_BLIND"):
                self.sb = v
            elif key in ("BB", "BIG_BLIND"):
                self.bb = v
            elif key in ("ANTE",):
                self.ante = v
            elif key in ("OPEN", "OPEN_MULT", "OPEN_SIZE"):
                self.open_mult = v
            else:
                errors.append(f"Parâmetro desconhecido: '{key}'")

        return errors

    def to_dict(self) -> dict:
        return {
            "sb":        self.sb,
            "bb":        self.bb,
            "ante":      self.ante,
            "open_mult": self.open_mult,
            "configured": self.bb > 0,
        }


# ── Stats de sessão ───────────────────────────────────────────────────────────

class SessionStats:
    """Acumula métricas por conexão cliente para o endpoint STATUS."""
    __slots__ = ("session_id", "connected_at", "requests", "errors",
                 "total_latency_ms", "green", "yellow", "red")

    def __init__(self) -> None:
        self.session_id    = str(uuid.uuid4())[:8]
        self.connected_at  = time.perf_counter()
        self.requests      = 0
        self.errors        = 0
        self.total_latency_ms = 0.0
        self.green = self.yellow = self.red = 0

    def record(self, color: str, latency_ms: float) -> None:
        self.requests      += 1
        self.total_latency_ms += latency_ms
        if color == "GREEN_INTENSE": self.green  += 1
        elif color == "RED_FOLD":    self.red    += 1
        else:                        self.yellow += 1

    @property
    def avg_latency(self) -> float:
        return self.total_latency_ms / max(self.requests, 1)

    def to_dict(self) -> dict:
        uptime = time.perf_counter() - self.connected_at
        return {
            "session_id":    self.session_id,
            "uptime_s":      round(uptime, 1),
            "requests":      self.requests,
            "errors":        self.errors,
            "avg_latency_ms": round(self.avg_latency, 2),
            "green":  self.green,
            "yellow": self.yellow,
            "red":    self.red,
        }


# ── Mapeamento decisão → color_code ──────────────────────────────────────────

def _map_color(result: dict) -> str:
    """
    Mapeia resultado para color_code.

    Hierarquia (primeira que casar vence):
      1. ALL IN               → PURPLE_ALLIN
      2. FOLD puro            → RED_FOLD
      3. EV numérico positivo → GREEN_INTENSE
      4. EV numérico negativo → RED_FOLD
      5. Ação agressiva/call  → GREEN_INTENSE
      6. Default              → GREEN_INTENSE (sem amarelo/indecisão)
    """
    decision_clean = result.get("decision", "").upper().split(" | ")[0].strip()

    # ALL IN: roxo
    if "ALL IN" in decision_clean or "ALL-IN" in decision_clean:
        return "PURPLE_ALLIN"

    # FOLD: vermelho
    if decision_clean.startswith("FOLD"):
        return "RED_FOLD"

    # Path EV numérico (pós-flop)
    ev = result.get("ev_adjusted") or result.get("ev")
    if isinstance(ev, (int, float)):
        if ev <= 0:
            return "RED_FOLD"
        return "GREEN_INTENSE"

    # Default para decisões de texto: GREEN_INTENSE
    return "GREEN_INTENSE"


def _build_response(
    result: dict,
    latency_ms: float,
    mc_mode: str,
    session_id: str,
    ctx: "SessionContext | None" = None,
) -> dict:
    """
    Constrói payload JSON para o app — SIMPLIFICADO E LIMPO.
    
    OUTPUT MINIMALISTA:
      - Ação principal APENAS (FOLD, CALL, BET/RAISE, ALL-IN)
      - Sem "AVALIAR" (ambiguidade removida)
      - Stack range e sizing em BB
      - EQ% como info secundária (não priority)
      - Sem ação distribution (poluição visual)
    """
    decision_clean = result["decision"].split(" | ")[0].strip()

    # ─────────────────────────────────────────────────────────────────────────
    # NORMALIZA DECISÃO PARA 4 AÇÕES SIMPLES
    # ─────────────────────────────────────────────────────────────────────────
    decision_upper = decision_clean.upper()
    
    # Extrai ação principal
    if "ALL" in decision_upper and "IN" in decision_upper:
        action_main = "ALL-IN"
    elif "FOLD" in decision_upper:
        action_main = "FOLD"
    elif "CALL" in decision_upper:
        action_main = "CALL"
    elif any(x in decision_upper for x in ["RAISE", "BET", "RFI", "3-BET", "4-BET"]):
        action_main = "BET"
    else:
        action_main = decision_clean

    # ─────────────────────────────────────────────────────────────────────────
    # SIZING EM BB (conversão automática fichas → BB)
    # ─────────────────────────────────────────────────────────────────────────
    sizing_str = ""
    stack_bb = float(result.get("eff_stack_bb", 0) or 0)

    if action_main == "ALL-IN" and stack_bb > 0:
        if ctx and ctx.bb > 0:
            sizing_str = f"{stack_bb:.1f} BB ({int(stack_bb * ctx.bb)} fichas)"
        else:
            sizing_str = f"{stack_bb:.1f} BB"
    elif result.get("sizing_bb"):
        sizing_bb = float(result.get("sizing_bb"))
        if ctx and ctx.bb > 0:
            sizing_str = f"{sizing_bb:.1f} BB ({int(sizing_bb * ctx.bb)} fichas)"
        else:
            sizing_str = f"{sizing_bb:.1f} BB"
    elif ctx and ctx.bb > 0:
        open_raise_chips = result.get("open_raise_chips")
        if open_raise_chips and open_raise_chips > 0:
            open_raise_bb = round(open_raise_chips / ctx.bb, 1)
            sizing_str = f"{open_raise_bb:.1f} BB ({int(open_raise_chips)} fichas)"

    ev_raw = result.get("ev_adjusted") or result.get("ev")
    ev_str = f"{ev_raw:+.2f}" if isinstance(ev_raw, (int, float)) else "—"

    eq_raw = (result.get("equity_adjusted")
              or result.get("equity")
              or result.get("preflop_equity"))
    equity_str = f"{eq_raw}%" if eq_raw is not None else "—"

    if action_main == "ALL-IN":
        color = "PURPLE_ALLIN"
    elif action_main == "FOLD":
        color = "RED_FOLD"
    else:
        color = "GREEN_INTENSE"

    # IC do Monte Carlo (mantém para análise pós-flop)
    mc_ci  = ""
    mc_err = 0.0
    mc_n   = 0
    if result.get("mc_ci_lower") is not None:
        mc_ci  = f"[{result['mc_ci_lower']}%–{result['mc_ci_upper']}%]"
        mc_err = result.get("mc_margin_of_error", 0.0)
        mc_n   = result.get("mc_iterations", 0)

    # ─────────────────────────────────────────────────────────────────────────
    # RESPOSTA SIMPLIFICADA
    # ─────────────────────────────────────────────────────────────────────────

    return {
        # CAMPO PRINCIPAL — ação direta, sem interpretação
        "decision":          action_main,
        "sizing":            sizing_str,
        "action":            action_main,
        
        # Info secundária (não priority visual)
        "equity":            equity_str,
        "ev":                ev_str,
        "stack_bb":          round(stack_bb, 1) if stack_bb > 0 else None,
        "hand":              result.get("hand_canonical") or result.get("hand"),
        "position":          result.get("position"),
        "decision_policy_version": result.get("decision_policy_version", DECISION_POLICY_VERSION),
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        
        # Contexto
        "color_code":        color,
        "hand_label":        result.get("hand_label", ""),
        "tier":              result.get("tier_label", "—"),
        "stack_range":       result.get("stack_range", "HIGH"),
        "has_board":         result.get("has_board", False),
        
        # Performance
        "latency_ms":        round(latency_ms, 2),
        "mc_mode":           mc_mode,
        "mc_n":              mc_n,
        "mc_ci":             mc_ci,
        "mc_error_pp":       mc_err,
        
        # Sessão e validação
        "session_id":        session_id,
        "blind_level":       ctx.to_dict() if ctx else None,
        "math_valid":        result.get("_math_valid", True),
        "math_warnings":     result.get("_validation_errors", [])[:3] if result.get("_validation_errors") else [],
        
        # ML (secundário)
        "ml_enabled":        result.get("ml_enabled", False),
        "ml_confidence":     result.get("ml_confidence", "no_model"),
        
        # REMOVER DO OUTPUT: action_distribution (poluição visual)
        # REMOVER DO OUTPUT: open_raise_chips/str (agora em 'sizing')
    }


# ── Modo Deep: worker isolado em processo separado ────────────────────────────

def _deep_worker(args: tuple) -> dict:
    """
    Roda no ProcessPoolExecutor — isolado do event loop.
    Recebe tokens já validados, retorna result dict.
    """
    hand, position, open_bb, stack_bb, is_multiway, board = args
    # Re-importa no processo filho (necessário por pickling)
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from decision_engine import evaluate_action
    from monte_carlo_engine import MCIntegrator

    if board:
        return MCIntegrator.full_analysis_mc(
            hero_hand_str=hand,
            board_strs=board,
            call_size_bb=open_bb,
            pot_size_bb=stack_bb,
            villain_position=position,
            stack_bb=stack_bb,
            mode="deep",
        )
    else:
        return evaluate_action(hand, position, open_bb, stack_bb, is_multiway, None)


# ── Parser de mensagem ────────────────────────────────────────────────────────

def _normalize_hand_token(hand: str) -> str:
    """
    Normaliza o token de mão vindo do Android (tudo maiúsculo).
    AKS → AKs  |  86O → 86o  |  AAS → AA  |  KKO → KK  |  AA → AA
    """
    if len(hand) == 3:
        r1 = hand[0].upper()
        r2 = hand[1].upper()
        q  = hand[2].upper()
        # Par com qualifier (AAS, AAO, KKS, etc.) → descarta qualifier
        if r1 == r2 and q in ('S', 'O'):
            return f"{r1}{r2}"
        # Suited/offsuit normal (AKS → AKs, 86O → 86o)
        if q in ('S', 'O'):
            return f"{r1}{r2}{q.lower()}"
    return hand.upper() if len(hand) == 2 else hand


_CARD_RE = re.compile(r"^[2-9TJQKA][hdcs]$", re.IGNORECASE)


def _is_valid_card(card: str) -> bool:
    return bool(_CARD_RE.match(card.strip()))


def _parse_message(raw: str) -> dict:
    """
    Parseia string de input do cliente.

    Formatos suportados:
      Pré-flop RFI:    MÃO POS OPEN_BB STACK_BB MULTIWAY
      Pré-flop vs open: MÃO POS VILLAIN_OPEN STACK_BB MULTIWAY
      Resposta 3-bet:  MÃO POS VILLAIN_3BET STACK_BB MULTIWAY 3BET
      Pós-flop:        MÃO POS BB_APOSTA POT MULTIWAY C1 C2 C3 [C4] [C5]
      Review (deep):   REVIEW MÃO POS BB POT MULTIWAY [CARTAS...]

    Retorna dict com todas as chaves necessárias.
    """
    tokens = raw.strip().split()
    if not tokens:
        raise ValueError("Mensagem vazia.")

    # Flag REVIEW
    is_review = tokens[0].upper() == "REVIEW"
    if is_review:
        tokens = tokens[1:]

    n = len(tokens)
    if n < 5:
        raise ValueError(
            f"Tokens insuficientes ({n}). "
            f"Formato: MÃO POSIÇÃO OPEN_BB STACK_BB MULTIWAY [CARTAS...] [3BET]"
        )

    hand     = _normalize_hand_token(tokens[0])
    position = tokens[1].upper()

    try:
        open_bb  = float(tokens[2])
        stack_bb = float(tokens[3])
    except ValueError:
        raise ValueError(f"Valores numéricos inválidos: '{tokens[2]}' / '{tokens[3]}'.")

    if stack_bb <= 0:
        raise ValueError("STACK_BB deve ser positivo.")
    if open_bb < 0:
        raise ValueError("OPEN_BB não pode ser negativo. Use 0 para RFI (abrir o pot).")

    mway_tok = tokens[4].upper()
    if mway_tok not in ("Y", "N"):
        raise ValueError(f"MULTIWAY inválido: '{tokens[4]}'. Use Y ou N.")
    is_multiway = mway_tok == "Y"

    # Tokens restantes: cartas do board e/ou flag 3BET
    rest         = tokens[5:]
    is_3bet_spot = False
    board        = None

    if rest:
        # Detecta flag 3BET no final
        if rest[-1].upper() == "3BET":
            is_3bet_spot = True
            rest = rest[:-1]
        if rest:
            board = rest

    # Validação do board
    if board is not None:
        if len(board) in (1, 2):
            raise ValueError(
                f"Board incompleto ({len(board)} carta(s)). Use 3 (flop), 4 (turn) ou 5 (river)."
            )
        if len(board) > 5:
            raise ValueError(f"Board com muitas cartas ({len(board)}). Máximo: 5.")
        norm_board = [c.strip() for c in board]
        if not all(_is_valid_card(c) for c in norm_board):
            raise ValueError("Board inválido: use cartas no formato Ah Kd 2c.")
        if len(set(c.lower() for c in norm_board)) != len(norm_board):
            raise ValueError("Board inválido: cartas duplicadas detectadas.")

    return {
        "is_review":    is_review,
        "hand":         hand,
        "position":     position,
        "open_bb":      open_bb,
        "stack_bb":     stack_bb,
        "is_multiway":  is_multiway,
        "board":        board,
        "is_3bet_spot": is_3bet_spot,
    }


# ── Helper: análise vs 3-bet ──────────────────────────────────────────────────

def _handle_3bet(tokens: list[str], ctx: "SessionContext", session_id: str) -> dict:
    """
    Processa comando:  3BET HAND POS STACK_BB VILLAIN_3BET_BB [HERO_OPEN_BB]

    Exemplos:
      3BET AKs BTN 35 9
      3BET TT CO 28 10 2.5
    """
    if len(tokens) < 4:
        return {
            "decision":   "ERRO: 3BET requer: HAND POS STACK_BB VILLAIN_3BET_BB [HERO_OPEN_BB]",
            "color_code": "GRAY_ERROR",
            "ev": "—", "equity": "—", "has_board": False,
            "decision_policy_version": DECISION_POLICY_VERSION,
            "response_schema_version": RESPONSE_SCHEMA_VERSION,
            "session_id": session_id,
        }

    hand         = tokens[0]
    if len(hand) == 3 and hand[-1].upper() in ("S", "O"):
        hand = hand[:2].upper() + hand[-1].lower()

    position     = tokens[1].upper()
    try:
        stack_bb     = float(tokens[2])
        v3bet_bb     = float(tokens[3])
        hero_open_bb = float(tokens[4]) if len(tokens) > 4 else 2.5
    except (ValueError, IndexError):
        return {
            "decision":   "ERRO: valores numéricos inválidos no comando 3BET",
            "color_code": "GRAY_ERROR",
            "ev": "—", "equity": "—", "has_board": False,
            "decision_policy_version": DECISION_POLICY_VERSION,
            "response_schema_version": RESPONSE_SCHEMA_VERSION,
            "session_id": session_id,
        }

    # IP = In Position se BTN ou CO
    is_ip = position in ("BTN", "CO", "HJ")

    analysis = analyze_vs_3bet(
        hand         = hand,
        position     = position,
        stack_bb     = stack_bb,
        hero_open_bb = hero_open_bb,
        villain_3bet_bb = v3bet_bb,
        bb_chips     = ctx.bb,
        is_ip        = is_ip,
    )

    # Monta sizing string do 4-bet
    four_bet_str = f"{analysis['four_bet_bb']:.1f}bb"
    if ctx.bb > 0:
        four_bet_chips = round(analysis["four_bet_bb"] * ctx.bb)
        four_bet_str  += f" = {four_bet_chips:,} chips"

    decision_raw = analysis["decision"].upper()
    if "ALL" in decision_raw and "IN" in decision_raw:
        action_main = "ALL-IN"
        color = "PURPLE_ALLIN"
    elif "FOLD" in decision_raw:
        action_main = "FOLD"
        color = "RED_FOLD"
    elif "CALL" in decision_raw:
        action_main = "CALL"
        color = "GREEN_INTENSE"
    elif "4BET" in decision_raw or "BET" in decision_raw or "RAISE" in decision_raw:
        action_main = "BET"
        color = "GREEN_INTENSE"
    else:
        action_main = "FOLD"
        color = "RED_FOLD"

    sizing = ""
    if action_main == "BET":
        sizing = four_bet_str
    elif action_main == "ALL-IN":
        if ctx.bb > 0:
            sizing = f"{stack_bb:.1f} BB ({int(stack_bb * ctx.bb)} fichas)"
        else:
            sizing = f"{stack_bb:.1f} BB"

    return {
        "decision":          action_main,
        "action":            action_main,
        "sizing":            sizing,
        "stack_bb":          round(stack_bb, 1),
        "hand":              hand,
        "position":          position,
        "decision_policy_version": DECISION_POLICY_VERSION,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "ev":                "—",
        "color_code":        color,
        "equity":            "—",
        "hand_label":        hand,
        "tier":              "—",
        "has_board":         False,
        "mc_mode":           "preflop",
        "mc_n":              0,
        "mc_ci":             "",
        "mc_error_pp":       0.0,
        "session_id":        session_id,
        "stack_range":       classify_stack(stack_bb).value,
        "four_bet_sizing":   four_bet_str,
        "pot_odds_pct":      analysis["pot_odds_pct"],
        "blind_level":       ctx.to_dict(),
        "math_valid":        True,
        "math_warnings":     [],
        "is_vs_3bet":        True,
    }


# ── Handler WebSocket ─────────────────────────────────────────────────────────

async def handle(websocket, executor: ProcessPoolExecutor) -> None:
    """
    Processa conexão de um cliente.
    Uma conexão permanece aberta — sem handshake por mensagem.

    Comandos suportados:
      PING
      STATUS
      SET SB=800 BB=1200 ANTE=200    → configura blind level
      SET OPEN=2.5                   → configura open raise multiplier
      3BET HAND POS STACK VILLAIN_3BET_BB → análise vs 3-bet
      HAND POS OPEN_BB STACK_BB MULTIWAY [BOARD...]
      REVIEW HAND POS OPEN_BB STACK_BB MULTIWAY [BOARD...]
    """
    stats = SessionStats()
    ctx   = SessionContext()        # contexto persistente desta sessão
    peer  = websocket.remote_address
    log.info("Conexão  %s:%s  session=%s", peer[0], peer[1], stats.session_id)

    try:
        async for raw_message in websocket:
            t0  = time.perf_counter()
            raw = raw_message.strip() if isinstance(raw_message, str) else ""

            if not raw:
                continue

            # ── Comandos especiais ──
            if raw.upper() == "PING":
                await websocket.send(json.dumps({"pong": True, "ts": time.time()}))
                continue

            if raw.upper() == "STATUS":
                payload = {**stats.to_dict(), "blind_context": ctx.to_dict()}
                await websocket.send(json.dumps(payload))
                continue

            # ── Comando SET: configura contexto da sessão ──
            tokens_raw = raw.strip().split()
            if tokens_raw and tokens_raw[0].upper() == "SET":
                errs = ctx.update_from_tokens(tokens_raw[1:])
                resp = {
                    "command":      "SET",
                    "context":      ctx.to_dict(),
                    "errors":       errs,
                    "ok":           len(errs) == 0,
                    "session_id":   stats.session_id,
                }
                await websocket.send(json.dumps(resp, ensure_ascii=False))
                log.info("[%s] SET context: %s", stats.session_id, ctx.to_dict())
                continue

            log.info("← [%s] %s", stats.session_id, raw[:80])

            try:
                # ── Comando 3BET: análise vs 3-bet do vilão ──
                if tokens_raw and tokens_raw[0].upper() == "3BET":
                    response   = _handle_3bet(tokens_raw[1:], ctx, stats.session_id)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    response["latency_ms"] = round(latency_ms, 2)
                    stats.record(response.get("color_code", "GREEN_INTENSE"), latency_ms)
                    await websocket.send(json.dumps(response, ensure_ascii=False))
                    log.info("→ [%s] 3BET decision=%s lat=%.1fms",
                             stats.session_id, response.get("decision", "?")[:30], latency_ms)
                    continue

                # ── Análise padrão ──
                parsed      = _parse_message(raw)
                is_review   = parsed["is_review"]
                hand        = parsed["hand"]
                position    = parsed["position"]
                open_bb     = parsed["open_bb"]
                stack_bb    = parsed["stack_bb"]
                is_multiway = parsed["is_multiway"]
                board       = parsed["board"]
                is_3bet_spot = parsed["is_3bet_spot"]

                mc_mode = "preflop"

                if board:
                    if is_review:
                        mc_mode = "deep"
                        loop    = asyncio.get_event_loop()
                        result  = await loop.run_in_executor(
                            executor,
                            _deep_worker,
                            (hand, position, open_bb, stack_bb, is_multiway, board),
                        )
                    else:
                        mc_mode = "fast"
                        result  = await asyncio.to_thread(
                            evaluate_action,
                            hand, position, open_bb, stack_bb, is_multiway, board,
                            ctx.bb, ctx.ante, ctx.open_mult or None,
                            0, is_3bet_spot,
                        )
                else:
                    # Pré-flop: O(1), direto no event loop
                    result = evaluate_action(
                        hand, position, open_bb, stack_bb, is_multiway, None,
                        ctx.bb, ctx.ante, ctx.open_mult or None,
                        0, is_3bet_spot,
                    )

                latency_ms = (time.perf_counter() - t0) * 1000
                response   = _build_response(result, latency_ms, mc_mode, stats.session_id, ctx)

                _audit({**response, "hand": hand, "hand_canonical": result.get("hand_canonical", hand)})
                stats.record(response["color_code"], latency_ms)

            except ValueError as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                response   = {
                    "decision":    f"ERRO: {str(exc)[:80]}",
                    "ev":          "—",
                    "color_code":  "GRAY_ERROR",
                    "equity":      "—",
                    "hand_label":  "",
                    "tier":        "—",
                    "has_board":   False,
                    "latency_ms":  round(latency_ms, 2),
                    "mc_mode":     "error",
                    "mc_n":        0,
                    "mc_ci":       "",
                    "mc_error_pp": 0.0,
                    "decision_policy_version": DECISION_POLICY_VERSION,
                    "response_schema_version": RESPONSE_SCHEMA_VERSION,
                    "session_id":  stats.session_id,
                    "stack_range": "—",
                    "open_raise_chips": None,
                    "open_raise_str":   None,
                    "blind_level":      ctx.to_dict(),
                    "math_valid":       False,
                    "math_warnings":    [],
                }
                stats.errors += 1
                log.warning("[%s] Parse error: %s", stats.session_id, exc)

            payload = json.dumps(response, ensure_ascii=False)
            await websocket.send(payload)
            log.info(
                "→ [%s] color=%-14s ev=%-8s lat=%.1fms  mode=%s  range=%s",
                stats.session_id,
                response.get("color_code","?"),
                response.get("ev","?"),
                response.get("latency_ms", 0),
                response.get("mc_mode","?"),
                response.get("stack_range","?"),
            )

    except websockets.exceptions.ConnectionClosed:
        log.info(
            "Desconectado [%s]  %d req / %d err / avg=%.1fms",
            stats.session_id, stats.requests, stats.errors, stats.avg_latency,
        )


# ── Health check HTTP (porta separada) ────────────────────────────────────────

async def _health_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """
    Responde requisições HTTP simples na porta 8001.
    Permite: curl http://VM_IP:8001/health
    Retorna JSON com status e métricas básicas.
    """
    try:
        await asyncio.wait_for(reader.readline(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    body = json.dumps({
        "status":   "ok",
        "service":  "poker-dss",
        "cores":    _N_CORES,
        "fast_n":   _FAST_N,
        "deep_n":   _DEEP_N,
        "audit":    str(_audit_path.name),
        "ts":       datetime.now(timezone.utc).isoformat(),
    })
    http_response = (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
        f"{body}"
    )
    writer.write(http_response.encode())
    await writer.drain()
    writer.close()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(port: int = _PORT) -> None:
    _cleanup_old_audit_logs()
    log.info("=" * 56)
    log.info("  Poker DSS — WebSocket Server")
    log.info("  WS  : ws://0.0.0.0:%d", port)
    log.info("  HTTP health: http://0.0.0.0:%d/health", port + 1)
    log.info("  Cores disponíveis: %d  (deep mode usa todos)", _N_CORES)
    log.info("  Fast N=%d  Deep N=%d", _FAST_N, _DEEP_N)
    log.info("  Audit trail: %s", _audit_path)
    log.info("=" * 56)

    # Pool de processos para modo deep — inicializado uma vez
    with ProcessPoolExecutor(max_workers=_N_CORES) as executor:

        # Handler sem path — compatível com websockets 10+
        async def ws_handler(websocket):
            await handle(websocket, executor)

        # Servidor WebSocket
        async with websockets.serve(
            ws_handler,
            _HOST, port,
            ping_interval=20,
            ping_timeout=10,
            max_size=2048,
            compression=None,
        ):
            # Health check HTTP na porta seguinte
            health_srv = await asyncio.start_server(
                _health_handler, _HOST, port + 1
            )
            async with health_srv:
                log.info("Servidor ativo. Ctrl+C para encerrar.")
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    pass

    _audit_fh.flush()
    _audit_fh.close()
    log.info("Servidor encerrado com segurança.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poker DSS WebSocket Server")
    parser.add_argument("--port", type=int, default=_PORT, help=f"Porta WS (default: {_PORT})")
    args = parser.parse_args()

    try:
        from realtime.server import main as realtime_main

        asyncio.run(realtime_main(port=args.port))
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — encerrando...")