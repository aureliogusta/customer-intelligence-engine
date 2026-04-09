"""
decision_engine.py
==================
Motor de decisão — Poker DSS v5.

Mudanças desta versão
---------------------
  ✅ Decisões binárias — sem AVALIAR/amarelo. Sempre FOLD ou ação agressiva.
  ✅ Bug fix — AAs/AAo/KKs/KKo agora normalizados corretamente
  ✅ Sizing exato — RFI = 2BB + 2BB por limp (configurável)
  ✅ Resposta ao 3-bet do vilão — FOLD / CALL / 4-BET (com sizing)
  ✅ Sizing do 3-bet = 3.5x o raise do vilão
  ✅ Alertas de perfil ML por posição (HeroProfileAdvisor)
  ✅ Ações: FOLD / RFI / FLAT / 3-BET / ALL-IN

Protocolo WebSocket (entrada)
------------------------------
  Pré-flop normal:   "MÃO POS OPEN_BB STACK_BB MULTIWAY"
  Resposta a 3-bet:  "MÃO POS VILLAIN_3BET STACK_BB MULTIWAY 3BET"
  Pós-flop:          "MÃO POS BB_APOSTA POT_BB MULTIWAY C1 C2 C3 [C4] [C5]"
"""

from __future__ import annotations

import sys
import os
import re
import time
import logging
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base import (
    FLAT_RULES_CONFIG,
    HAND_RANKINGS,
    HAND_KEY_BY_LABEL,
    BOARD_TEXTURE_WEIGHTS,
    POSITION_GROUP,
    RFI_RANGES,
)
from equity_engine import EquityCalculator, DeckManager
from range_manager import (
    StackRange,
    classify_stack,
    is_in_range,
    should_push,
    calc_open_raise_chips,
    get_action_distribution,
    analyze_vs_3bet,
)
from math_validator import MathValidator

try:
    from monte_carlo_engine import MCIntegrator as _MCIntegrator
    _MC_AVAILABLE = True
except ImportError:
    _MC_AVAILABLE = False

log = logging.getLogger("decision_engine")

# =============================================================================
# ML runtime (lazy load, fail-safe)
# =============================================================================

_ML_RUNTIME = None
_ML_ENABLED = os.getenv("POKER_ML_ENABLE", "1") == "1"
_ML_MIN_SAMPLES_FOR_FLIP = 80
_ML_FACTOR_MIN = 0.80
_ML_FACTOR_MAX = 1.20
_CANONICAL_ACTIONS = frozenset({"FOLD", "CALL", "BET", "ALL-IN"})
DECISION_POLICY_VERSION = "2026.03-medium-1"


def _get_ml_runtime():
    return None


def _is_aggressive_action(action: str) -> bool:
    a = (action or "").upper()
    return any(k in a for k in ("RFI", "3-BET", "4-BET", "RAISE", "BET", "ALL-IN"))


def _canonicalize_action(action: str, decision_text: str = "") -> str:
    """Converte qualquer ação legada para o contrato canônico de 4 ações."""
    a = (action or "").upper().strip()
    d = (decision_text or "").upper().strip()

    if "ALL" in a and "IN" in a:
        return "ALL-IN"
    if "ALL" in d and "IN" in d:
        return "ALL-IN"

    if a.startswith("FOLD") or d.startswith("FOLD"):
        return "FOLD"

    if a.startswith("CALL") or "FLAT" in a or "CALL" in d or "FLAT" in d:
        return "CALL"

    if any(k in a for k in ("BET", "RAISE", "RFI", "3-BET", "4-BET")):
        return "BET"
    if any(k in d for k in ("BET", "RAISE", "RFI", "3-BET", "4-BET")):
        return "BET"

    return "FOLD"


def _finalize_decision_payload(result: dict) -> dict:
    """Padroniza payload final do engine para decisão direta e estável."""
    detail = str(result.get("decision", "") or "").strip()
    action = _canonicalize_action(result.get("action", ""), detail)

    result["action"] = action
    result["decision"] = action
    result["decision_detail"] = detail if detail and detail.upper() != action else ""

    if action == "ALL-IN":
        result["color_code"] = "PURPLE_ALLIN"
    elif action == "FOLD":
        result["color_code"] = "RED_FOLD"
    else:
        result["color_code"] = "GREEN_INTENSE"

    # Garante contrato estável
    if result["decision"] not in _CANONICAL_ACTIONS:
        result["decision"] = "FOLD"
        result["action"] = "FOLD"
        result["color_code"] = "RED_FOLD"

    # Rastreabilidade da policy ativa para auditoria de regressão em produção.
    result["decision_policy_version"] = DECISION_POLICY_VERSION

    return result


def _build_ml_context(
    hand_canonical: str,
    position: str,
    result: dict,
    eff_stack_bb: float,
    is_multiway: bool,
    board: list[str] | None,
    bb_chips: float,
) -> dict:
    action = result.get("action", "")
    bb_val = float(bb_chips) if bb_chips and bb_chips > 0 else 100.0
    board_flop = ""
    if board:
        board_flop = " ".join(board[:3])

    return {
        "hero_position": position,
        "hero_stack_start": max(eff_stack_bb * bb_val, 0.0),
        "big_blind": bb_val,
        "m_ratio": max(float(eff_stack_bb), 0.0),
        "hero_vpip": int(action not in ("", "FOLD")),
        "hero_pfr": int(_is_aggressive_action(action)),
        "hero_aggressor": int(_is_aggressive_action(action)),
        "hero_went_allin": int("ALL-IN" in (result.get("decision", "").upper())),
        "board_flop": board_flop,
        "num_players": 3 if is_multiway else 2,
        "hero_cards": hand_canonical,
        "level": 1,
        "went_to_showdown": 0,
        "hero_action_preflop": action.lower() if action else "fold",
    }


def _apply_ml_adjustment(
    result: dict,
    hand_canonical: str,
    position: str,
    open_size_bb: float,
    eff_stack_bb: float,
    is_multiway: bool,
    board: list[str] | None,
    bb_chips: float,
) -> dict:
    result.update({
        "ml_enabled": False,
        "ml_factor": 1.0,
        "ml_win_prob": 0.5,
        "ml_confidence": "disabled",
        "ml_samples": 0,
        "ml_insight": "legacy_ml_path_disabled",
    })
    return result

# =============================================================================
# Configuração de sizing
# =============================================================================

RFI_BASE_BB:       float = 2.0   # raise base em BB
RFI_LIMP_ADD_BB:   float = 2.0   # +2BB por limp antes
THREEBET_MULT:     float = 3.5   # 3-bet = 3.5x o raise do vilão
FOURBET_MULT:      float = 2.5   # 4-bet = 2.5x o 3-bet
MIN_STACK_ALLIN:   float = 15.0  # abaixo de X BB → push/fold direto
LOW_STACK_MAX_BB:  float = 20.0  # janela low stack principal

# Positions que têm posição tardia (steal/wide range)
_LATE_POSITIONS  = frozenset({"BTN", "CO"})
_BLIND_POSITIONS = frozenset({"SB", "BB"})
_EARLY_POSITIONS = frozenset({"UTG", "UTG+1", "LJ", "MP"})

# =============================================================================
# Rank table
# =============================================================================

_RANK_ORDER  = "AKQJT98765432"
_RANK_VAL    = {r: 14 - i for i, r in enumerate(_RANK_ORDER)}
_VALID_RANKS = set(_RANK_ORDER)
_VALID_SUITS = set("hdcs")

# =============================================================================
# Normalizador — corrigido para AAs/AAo/KKs etc.
# =============================================================================

def normalize_hand(hand: str) -> str:
    """
    Converte qualquer notação para forma canônica.

    Aceita:
      'AA'    → 'AA'     (par — pass-through)
      'AAs'   → 'AA'     (par com qualifier — CORRIGIDO)
      'AAo'   → 'AA'     (par com qualifier — CORRIGIDO)
      'AKs'   → 'AKs'   (já canônica)
      'AKo'   → 'AKo'   (já canônica)
      'JsAs'  → 'AJs'   (notação explícita suited)
      'Ac6d'  → 'A6o'   (notação explícita offsuit)
      '4c4h'  → '44'    (par com naipes)
    """
    hand = hand.strip()

    # ── 3 chars: par + qualifier (AAs, AAo, KKs, KKo, etc.) ──
    if len(hand) == 3:
        r1 = hand[0].upper()
        r2 = hand[1].upper()
        q  = hand[2].lower()
        if r1 in _VALID_RANKS and r2 in _VALID_RANKS and r1 == r2:
            return f"{r1}{r2}"   # descarta qualifier — par é sempre par
        # suited/offsuit com ranks diferentes (ex: AKs já está ok)
        if r1 in _VALID_RANKS and r2 in _VALID_RANKS and q in ('s', 'o'):
            # garante high card primeiro
            if _RANK_ORDER.index(r1) > _RANK_ORDER.index(r2):
                r1, r2 = r2, r1
            return f"{r1}{r2}{q}"
        return hand

    # ── 4 chars: notação explícita com naipes (JsAs, 4c4h, Ac6d) ──
    if len(hand) == 4:
        r1 = hand[0].upper(); s1 = hand[1].lower()
        r2 = hand[2].upper(); s2 = hand[3].lower()
        if (r1 not in _VALID_RANKS or r2 not in _VALID_RANKS or
                s1 not in _VALID_SUITS or s2 not in _VALID_SUITS):
            return hand
        if r1 == r2:
            return f"{r1}{r2}"
        if _RANK_ORDER.index(r1) > _RANK_ORDER.index(r2):
            r1, s1, r2, s2 = r2, s2, r1, s1
        return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"

    # ── 2 chars: par canônico (AA, KK, 77) ou suited/offsuit sem qualifier ──
    if len(hand) == 2:
        r1 = hand[0].upper(); r2 = hand[1].upper()
        if r1 in _VALID_RANKS and r2 in _VALID_RANKS:
            if r1 == r2:
                return hand.upper()
            # Ex: "AK" → ambíguo — trata como suited por default
            if _RANK_ORDER.index(r1) > _RANK_ORDER.index(r2):
                r1, r2 = r2, r1
            return f"{r1}{r2}s"

    return hand


# =============================================================================
# Lookup no knowledge_base
# =============================================================================

def _get_tier_info(hand: str) -> dict | None:
    """O(1) lookup via HAND_KEY_BY_LABEL."""
    key = HAND_KEY_BY_LABEL.get(hand)
    if key is None:
        return None
    return HAND_RANKINGS.get(key)


# =============================================================================
# Sizing calculators
# =============================================================================

def calc_rfi_size(limpers: int = 0) -> float:
    """RFI = 2BB + 2BB por limp."""
    return RFI_BASE_BB + limpers * RFI_LIMP_ADD_BB


def calc_3bet_size(villain_raise: float) -> float:
    """3-bet = 3.5x o raise do vilão."""
    return round(villain_raise * THREEBET_MULT, 1)


def calc_4bet_size(villain_3bet: float) -> float:
    """4-bet = 2.5x o 3-bet do vilão."""
    return round(villain_3bet * FOURBET_MULT, 1)


# =============================================================================
# Expansor de ranges (ex: "77+" → ["77","88","99","TT","JJ","QQ","KK","AA"])
# =============================================================================

_PAIRS_ORDER = ["22","33","44","55","66","77","88","99","TT","JJ","QQ","KK","AA"]
_RANKS_ASC   = list(reversed(_RANK_ORDER))  # 2,3,...,K,A

def _expand_range(notation: str) -> set[str]:
    """Expande notação de range para conjunto de mãos canônicas."""
    hands: set[str] = set()
    n = notation.strip()

    # Par com ladder: "77+" → 77,88,...,AA
    m = re.match(r'^(\d{1}|\w{1})(\1)\+$', n)
    if not m:
        m = re.match(r'^([AKQJT])(\1)\+$', n)
    if m:
        r = m.group(1).upper()
        start_idx = _PAIRS_ORDER.index(f"{r}{r}") if f"{r}{r}" in _PAIRS_ORDER else -1
        if start_idx >= 0:
            hands.update(_PAIRS_ORDER[start_idx:])
        return hands

    # Par exato: "77"
    if re.match(r'^([AKQJT2-9])\1$', n) or re.match(r'^(\d)\1$', n):
        r = n[0].upper()
        hands.add(f"{r}{r}")
        return hands

    # Suited ladder: "AJs+" → AJs, AQs, AKs
    m = re.match(r'^([AKQJT])([AKQJT2-9])s\+$', n)
    if m:
        high = m.group(1).upper()
        low  = m.group(2).upper()
        low_idx  = _RANK_ORDER.index(low)
        high_idx = _RANK_ORDER.index(high)
        for i in range(high_idx, low_idx):   # from low rank up
            r2 = _RANK_ORDER[i]
            if r2 != high:
                hands.add(f"{high}{r2}s")
        return hands

    # Offsuit ladder: "AQo+" → AQo, AKo
    m = re.match(r'^([AKQJT])([AKQJT2-9])o\+$', n)
    if m:
        high = m.group(1).upper()
        low  = m.group(2).upper()
        low_idx  = _RANK_ORDER.index(low)
        high_idx = _RANK_ORDER.index(high)
        for i in range(high_idx, low_idx):
            r2 = _RANK_ORDER[i]
            if r2 != high:
                hands.add(f"{high}{r2}o")
        return hands

    # Exato suited: "AKs", "T9s"
    m = re.match(r'^([AKQJT2-9])([AKQJT2-9])([so])$', n)
    if m:
        r1, r2, q = m.group(1).upper(), m.group(2).upper(), m.group(3)
        if _RANK_ORDER.index(r1) > _RANK_ORDER.index(r2):
            r1, r2 = r2, r1
        hands.add(f"{r1}{r2}{q}")
        return hands

    return hands


# Cache de ranges expandidos por posição
_RFI_EXPANDED: dict[str, set[str]] = {}

_GTO_CACHE: dict | None = None
_PUSH_CACHE: dict[tuple[str, str], set[str]] = {}


def _load_gto_ranges() -> dict:
    """Carrega gto_ranges.json sob demanda e retorna dict vazio em falhas."""
    global _GTO_CACHE
    if _GTO_CACHE is not None:
        return _GTO_CACHE

    try:
        p = Path(__file__).parent / "gto_ranges.json"
        _GTO_CACHE = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        _GTO_CACHE = {}
    return _GTO_CACHE


def _nearest_depth_key(keys: list[str], stack_bb: float) -> str | None:
    """Escolhe depth mais próxima em chaves no formato '<n>bb'."""
    parsed: list[tuple[float, str]] = []
    for k in keys:
        try:
            parsed.append((float(str(k).lower().replace("bb", "")), k))
        except Exception:
            continue
    if not parsed:
        return None
    return min(parsed, key=lambda x: abs(x[0] - float(stack_bb)))[1]


def _expand_weighted_notation_map(weighted_map: dict[str, float], threshold: float = 0.5) -> set[str]:
    hands: set[str] = set()
    for notation, weight in weighted_map.items():
        if str(notation).startswith("_"):
            continue
        try:
            w = float(weight)
        except Exception:
            continue
        if w >= threshold:
            hands.update(_expand_range(notation))
    return hands


def _is_push_from_gto_chart(hand: str, position: str, stack_bb: float) -> bool | None:
    """Retorna True/False quando há chart de push/fold para a posição; None se não houver cobertura."""
    gto = _load_gto_ranges()
    push_fold = gto.get("push_fold", {})
    pos_data = push_fold.get(position.upper())
    if not isinstance(pos_data, dict):
        return None

    depth_key = _nearest_depth_key([k for k in pos_data.keys() if not str(k).startswith("_")], stack_bb)
    if not depth_key:
        return None

    cache_key = (position.upper(), depth_key)
    if cache_key not in _PUSH_CACHE:
        raw = pos_data.get(depth_key, {})
        if not isinstance(raw, dict):
            _PUSH_CACHE[cache_key] = set()
        else:
            _PUSH_CACHE[cache_key] = _expand_weighted_notation_map(raw, threshold=0.5)

    return hand in _PUSH_CACHE.get(cache_key, set())


def _rfi_action_from_gto_chart(hand: str, position: str, stack_bb: float) -> str | None:
    """Retorna ação dominante do chart RFI: 'BET'|'CALL'|'FOLD' ou None sem cobertura."""
    gto = _load_gto_ranges()
    rfi = gto.get("RFI", {})
    pos_data = rfi.get(position.upper())
    if not isinstance(pos_data, dict):
        return None

    depth_key = _nearest_depth_key([k for k in pos_data.keys() if not str(k).startswith("_")], stack_bb)
    if not depth_key:
        return None

    depth_table = pos_data.get(depth_key, {})
    if not isinstance(depth_table, dict):
        return None

    entry = depth_table.get(hand)
    if not isinstance(entry, dict):
        return "FOLD"

    r = float(entry.get("r", 0.0) or 0.0)
    c = float(entry.get("c", 0.0) or 0.0)
    f = float(entry.get("f", 0.0) or 0.0)
    top = max(("BET", r), ("CALL", c), ("FOLD", f), key=lambda x: x[1])[0]
    return top


def _vs_3bet_action_from_gto_chart(hand: str, position: str, stack_bb: float) -> str | None:
    """Retorna ação dominante vs 3-bet: 'BET'|'CALL'|'FOLD' ou None sem cobertura."""
    gto = _load_gto_ranges()
    vs_3bet = gto.get("vs_3bet", {})
    if not isinstance(vs_3bet, dict):
        return None

    pos = position.upper()
    spot_key = None
    for k in vs_3bet.keys():
        if str(k).startswith("_"):
            continue
        if str(k).upper().startswith(f"{pos}_VS_"):
            spot_key = k
            break
    if not spot_key:
        return None

    spot = vs_3bet.get(spot_key, {})
    if not isinstance(spot, dict):
        return None

    depth_key = _nearest_depth_key([k for k in spot.keys() if not str(k).startswith("_")], stack_bb)
    if not depth_key:
        return None

    depth_table = spot.get(depth_key, {})
    if not isinstance(depth_table, dict):
        return None

    entry = depth_table.get(hand)
    if not isinstance(entry, dict):
        return "FOLD"

    r = float(entry.get("r", 0.0) or 0.0)
    c = float(entry.get("c", 0.0) or 0.0)
    f = float(entry.get("f", 0.0) or 0.0)
    return max(("BET", r), ("CALL", c), ("FOLD", f), key=lambda x: x[1])[0]

def _get_rfi_range(position: str) -> set[str]:
    """Retorna o conjunto expandido de mãos do RFI para a posição."""
    if position not in _RFI_EXPANDED:
        raw = RFI_RANGES.get(position, [])
        expanded: set[str] = set()
        for notation in raw:
            expanded.update(_expand_range(notation))
        _RFI_EXPANDED[position] = expanded
    return _RFI_EXPANDED[position]


def is_in_rfi_range(hand: str, position: str) -> bool:
    """Verifica se a mão está no range de RFI da posição."""
    return hand in _get_rfi_range(position)


# =============================================================================
# 3-bet ranges (mãos que 3-betam vs abertura do vilão)
# =============================================================================

# Value 3-bet: mãos premium que 3-betam para valor
_3BET_VALUE: dict[str, set[str]] = {
    "UTG": {"AA","KK","QQ","AKs","AKo"},
    "LJ":  {"AA","KK","QQ","JJ","AKs","AKo","AQs"},
    "HJ":  {"AA","KK","QQ","JJ","TT","AKs","AKo","AQs","AQo"},
    "CO":  {"AA","KK","QQ","JJ","TT","AKs","AKo","AQs","AQo","AJs"},
    "BTN": {"AA","KK","QQ","JJ","TT","AKs","AKo","AQs","AQo","AJs","KQs"},
    "SB":  {"AA","KK","QQ","JJ","TT","AKs","AKo","AQs","AQo"},
    "BB":  {"AA","KK","QQ","JJ","TT","AKs","AKo","AQs","AQo","AJs"},
}

# Bluff 3-bet: suited aces e suited connectors com blockers
_3BET_BLUFF: dict[str, set[str]] = {
    "UTG": set(),
    "LJ":  {"A5s","A4s"},
    "HJ":  {"A5s","A4s","A3s"},
    "CO":  {"A5s","A4s","A3s","A2s","K5s"},
    "BTN": {"A5s","A4s","A3s","A2s","K5s","K4s","65s","54s"},
    "SB":  {"A5s","A4s","A3s","A2s","K5s"},
    "BB":  {"A5s","A4s","A3s","A2s"},
}

def is_3bet_hand(hand: str, position: str) -> tuple[bool, str]:
    """
    Retorna (deve_3bet, motivo).
    motivo: 'value' | 'bluff' | ''
    """
    value_range = _3BET_VALUE.get(position, set())
    bluff_range = _3BET_BLUFF.get(position, set())
    if hand in value_range:
        return True, "value"
    if hand in bluff_range:
        return True, "bluff"
    return False, ""


# =============================================================================
# Flat (call) ranges vs abertura
# =============================================================================

_FLAT_RANGE: dict[str, set[str]] = {
    "BTN": {"JJ","TT","99","88","77","66","55","44","33","22",
            "AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
            "KQs","KJs","KTs","K9s",
            "QJs","QTs","Q9s","JTs","J9s","T9s","98s","87s","76s","65s",
            "AQo","AJo","ATo","KQo","KJo","QJo"},
    "CO":  {"JJ","TT","99","88","77","66","55","44",
            "AJs","ATs","A9s","A8s","A7s","A6s","A5s",
            "KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s",
            "AQo","AJo","KQo"},
    "HJ":  {"JJ","TT","99","88","77","66","55",
            "AJs","ATs","A9s","KQs","KJs","QJs","JTs","T9s",
            "AQo","KQo"},
    "SB":  {"JJ","TT","99","88","77","66","55","44","33","22",
            "AJs","ATs","A9s","A8s","A5s","A4s","A3s","A2s",
            "KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s","87s","76s",
            "AQo","AJo","KQo","KJo"},
    "BB":  {"JJ","TT","99","88","77","66","55","44","33","22",
            "AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
            "KQs","KJs","KTs","K9s","K8s",
            "QJs","QTs","Q9s","JTs","J9s","T9s","98s","87s","76s","65s","54s",
            "AQo","AJo","ATo","KQo","KJo","QJo","JTo"},
}

def is_flat_hand(hand: str, position: str) -> bool:
    return hand in _FLAT_RANGE.get(position, set())


# =============================================================================
# Push/fold ranges (stack curto — jam direto)
# =============================================================================

def is_push_hand(hand: str, stack_bb: float, position: str) -> bool:
    """Retorna True se deve jam dado o stack."""
    chart_push = _is_push_from_gto_chart(hand, position, stack_bb)
    if chart_push is not None:
        return chart_push

    tier_info = _get_tier_info(hand)
    if not tier_info:
        return False
    tier = tier_info["tier"]
    # Com menos de 10bb qualquer Tier 1-2 vai all-in
    if stack_bb <= 10 and tier <= 2:
        return True
    # Com menos de 15bb Tier 1-3 vai all-in de todas as posições
    if stack_bb <= 15 and tier <= 3:
        return True
    # Com menos de 20bb Tier 1-2 vai all-in de posições tardias
    if stack_bb <= 20 and tier <= 2 and position in _LATE_POSITIONS | _BLIND_POSITIONS:
        return True
    return False


# =============================================================================
# HeroProfileAdvisor — alertas de perfil ML
# =============================================================================

class HeroProfileAdvisor:
    _PROFILE: dict = {
        "UTG": {"vpip_real": 24.5, "vpip_gto": 14.0, "pfr_real": 18.7, "pfr_gto": 13.0, "leak": "loose"},
        "MP":  {"vpip_real": 23.9, "vpip_gto": 17.0, "pfr_real": 19.7, "pfr_gto": 15.0, "leak": "loose"},
        "HJ":  {"vpip_real": 20.8, "vpip_gto": 20.0, "pfr_real": 14.8, "pfr_gto": 18.0, "leak": "pfr_low"},
        "CO":  {"vpip_real": 20.0, "vpip_gto": 26.0, "pfr_real": 21.9, "pfr_gto": 23.0, "leak": "tight"},
        "BTN": {"vpip_real": 28.8, "vpip_gto": 45.0, "pfr_real": 23.3, "pfr_gto": 38.0, "leak": "tight"},
        "SB":  {"vpip_real": 13.2, "vpip_gto": 35.0, "pfr_real":  5.6, "pfr_gto": 28.0, "leak": "passive"},
        "BB":  {"vpip_real": 24.0, "vpip_gto": 40.0, "pfr_real": 14.4, "pfr_gto": 10.0, "leak": "tight"},
    }
    _THRESHOLD = 4.0

    @classmethod
    def get_alert(cls, position: str, action: str) -> str:
        data = cls._PROFILE.get(position)
        if not data:
            return ""
        leak       = data["leak"]
        vpip_diff  = data["vpip_real"] - data["vpip_gto"]
        pfr_diff   = data["pfr_real"]  - data["pfr_gto"]
        action_up  = action.upper()

        if leak in ("tight", "passive") and abs(vpip_diff) >= cls._THRESHOLD:
            if "FOLD" in action_up:
                return f" | ⚠ PERFIL: {position} tight ({data['vpip_real']:.0f}% vs GTO {data['vpip_gto']:.0f}%)"
        if leak == "loose" and vpip_diff >= cls._THRESHOLD:
            if any(k in action_up for k in ("RFI","3-BET","FLAT","ALL-IN")):
                return f" | ⚠ PERFIL: {position} loose ({data['vpip_real']:.0f}% vs GTO {data['vpip_gto']:.0f}%)"
        if leak == "pfr_low" and abs(pfr_diff) >= cls._THRESHOLD:
            if "FLAT" in action_up:
                return f" | ⚠ PERFIL: PFR baixo no {position} — considere 3-bet"
        return ""

    @classmethod
    def update_profile(cls, desvios: dict) -> None:
        for pos, d in desvios.items():
            if pos not in cls._PROFILE:
                continue
            cls._PROFILE[pos]["vpip_real"] = d.get("vpip_real", cls._PROFILE[pos]["vpip_real"])
            cls._PROFILE[pos]["pfr_real"]  = d.get("pfr_real",  cls._PROFILE[pos]["pfr_real"])


def sync_profile_from_db(limit: int = 5000) -> bool:
    """
    Atualiza perfil do HeroProfileAdvisor a partir do histórico no PostgreSQL.

    Execução é explícita (não ocorre no import) para evitar side-effects,
    latência de startup e acoplamento obrigatório ao banco.
    """
    log.info("sync_profile_from_db desativado: caminho legado sem papel no ex-ante.")
    return False


# =============================================================================
# DECISÃO PRÉ-FLOP — lógica binária
# =============================================================================

def evaluate_preflop(
    hand: str,
    position: str,
    open_size_bb: float,
    stack_bb: float,
    is_multiway: bool,
    bb_chips: float = 0.0,
    ante_chips: float = 0.0,
    custom_open_mult: float | None = None,
    limpers: int = 0,
    is_3bet_spot: bool = False,
) -> dict:
    canonical = normalize_hand(hand)
    stack_range = classify_stack(stack_bb)
    rfi_size = calc_rfi_size(limpers) if custom_open_mult is None else calc_rfi_size(limpers) * float(custom_open_mult)
    open_raise_chips = round(float(rfi_size) * float(bb_chips or 100.0), 2)
    open_raise_str = f"RFI → {rfi_size:.1f}BB"

    from decision_service.inference import predict_ex_ante
    from decision_service.models import ExAnteDecisionRequest

    result = predict_ex_ante(
        ExAnteDecisionRequest(
            hand=canonical,
            position=position,
            stack_bb=float(stack_bb),
            pot_bb_before=float(open_size_bb or 0.0),
            num_players=3 if is_multiway else 2,
            limpers=int(limpers or 0),
            open_size_bb=float(open_size_bb or 0.0),
            street="PREFLOP",
            ante_bb=float(ante_chips or 0.0) / float(bb_chips or 1.0) if bb_chips else 0.0,
            bb_chips=float(bb_chips or 100.0),
            is_3bet_spot=bool(is_3bet_spot),
        )
    )

    legacy_action = str(result.action)
    if legacy_action == "RAISE":
        legacy_action = "ALL-IN" if float(stack_bb) <= MIN_STACK_ALLIN else ("RFI" if open_size_bb <= 1.0 else "BET")

    payload = result.__dict__.copy()
    payload.update({
        "action": legacy_action,
        "decision": legacy_action,
        "decision_detail": result.rationale,
        "hand": hand,
        "hand_canonical": canonical,
        "position": position,
        "has_board": False,
        "open_raise_chips": open_raise_chips,
        "open_raise_str": open_raise_str,
        "stack_range": stack_range.value,
        "action_distribution": get_action_distribution(legacy_action, stack_range),
        "tier": (_get_tier_info(canonical) or {}).get("tier", 6),
    })
    return payload


def _decide_rfi(
    hand: str, position: str,
    rfi_size: float, stack_bb: float, is_multiway: bool,
) -> dict:
    """Hero é o primeiro a abrir."""
    in_rfi = is_in_rfi_range(hand, position)

    if in_rfi:
        action   = "RFI"
        decision = f"RFI → {rfi_size:.1f}BB"
        color    = "GREEN_INTENSE"
    else:
        action   = "FOLD"
        decision = "FOLD"
        color    = "RED_FOLD"

    return _build_preflop_result(hand, position, action, decision, color, stack_bb, sizing=rfi_size)


def _decide_rfi_chart_driven(
    hand: str, position: str,
    rfi_size: float, stack_bb: float,
) -> dict | None:
    """Decisão RFI orientada por chart GTO (ação dominante) quando disponível."""
    chart_action = _rfi_action_from_gto_chart(hand, position, stack_bb)
    if chart_action is None:
        return None

    if chart_action == "BET":
        action = "BET"
        decision = f"BET: {rfi_size:.1f}BB"
        color = "GREEN_INTENSE"
    elif chart_action == "CALL":
        action = "CALL"
        decision = "CALL"
        color = "GREEN_INTENSE"
    else:
        action = "FOLD"
        decision = "FOLD"
        color = "RED_FOLD"

    return _build_preflop_result(hand, position, action, decision, color, stack_bb, sizing=rfi_size if action == "BET" else 0.0)


def _decide_vs_open(
    hand: str, position: str,
    villain_open: float, stack_bb: float, is_multiway: bool,
) -> dict:
    """Vilão abriu — hero decide: FOLD / FLAT / 3-BET."""
    do_3bet, reason_3bet = is_3bet_hand(hand, position)
    do_flat              = is_flat_hand(hand, position)

    if do_3bet:
        size_3bet = calc_3bet_size(villain_open)
        tag       = "valor" if reason_3bet == "value" else "bluff"
        action    = "3-BET"
        decision  = f"3-BET → {size_3bet:.1f}BB ({tag})"
        color     = "GREEN_INTENSE"
    elif do_flat and not is_multiway:
        action   = "FLAT"
        decision = f"FLAT → call {villain_open:.1f}BB"
        color    = "GREEN_INTENSE"
    elif do_flat and is_multiway:
        # Multiway — apenas mãos de valor especulativo com implied odds
        tier_info = _get_tier_info(hand)
        tier      = tier_info["tier"] if tier_info else 6
        if tier <= 3:
            action   = "FLAT"
            decision = f"FLAT → call {villain_open:.1f}BB (multiway)"
            color    = "GREEN_INTENSE"
        else:
            action   = "FOLD"
            decision = "FOLD (multiway — implied odds insuficientes)"
            color    = "RED_FOLD"
    else:
        action   = "FOLD"
        decision = "FOLD"
        color    = "RED_FOLD"

    return _build_preflop_result(
        hand, position, action, decision, color, stack_bb,
        sizing=calc_3bet_size(villain_open) if do_3bet else villain_open
    )


def _respond_to_3bet(
    hand: str, position: str,
    villain_3bet: float, stack_bb: float,
) -> dict:
    """Hero responde ao 3-bet do vilão: FOLD / CALL / 4-BET / ALL-IN."""
    chart_action = _vs_3bet_action_from_gto_chart(hand, position, stack_bb)
    if chart_action is not None:
        if chart_action == "BET":
            size_4bet = calc_4bet_size(villain_3bet)
            if size_4bet >= stack_bb * 0.65:
                return _build_preflop_result(
                    hand, position, "ALL-IN", f"ALL-IN ({stack_bb:.0f}BB)", "GREEN_INTENSE", stack_bb
                )
            return _build_preflop_result(
                hand, position, "BET", f"BET: {size_4bet:.1f}BB", "GREEN_INTENSE", stack_bb, sizing=size_4bet
            )
        if chart_action == "CALL":
            return _build_preflop_result(
                hand, position, "CALL", f"CALL ({villain_3bet:.1f}BB)", "GREEN_INTENSE", stack_bb, sizing=villain_3bet
            )
        return _build_preflop_result(
            hand, position, "FOLD", "FOLD", "RED_FOLD", stack_bb
        )

    tier_info = _get_tier_info(hand)
    tier      = tier_info["tier"] if tier_info else 6

    # Stack curto → push direto com premiums
    if stack_bb <= 25 and tier <= 2:
        action   = "ALL-IN"
        decision = f"ALL-IN → {stack_bb:.0f}BB (vs 3-bet)"
        color    = "GREEN_INTENSE"
        return _build_preflop_result(hand, position, action, decision, color, stack_bb)

    # Premium — 4-bet para valor
    if tier == 1 or hand in {"AKs", "AKo", "QQ"}:
        size_4bet = calc_4bet_size(villain_3bet)
        if size_4bet >= stack_bb * 0.65:
            action   = "ALL-IN"
            decision = f"ALL-IN → {stack_bb:.0f}BB (4-bet jam vs 3-bet)"
            color    = "GREEN_INTENSE"
        else:
            action   = "4-BET"
            decision = f"4-BET → {size_4bet:.1f}BB"
            color    = "GREEN_INTENSE"
        return _build_preflop_result(hand, position, action, decision, color, stack_bb, sizing=size_4bet)

    # Tier 2 — call com pot odds favoráveis
    if tier == 2:
        pot_odds = villain_3bet / (villain_3bet * 2 + villain_3bet)
        eq       = tier_info.get("preflop_equity", 0.5) if tier_info else 0.5
        if eq > pot_odds:
            action   = "CALL"
            decision = f"CALL → {villain_3bet:.1f}BB (vs 3-bet)"
            color    = "GREEN_INTENSE"
        else:
            action   = "FOLD"
            decision = "FOLD (vs 3-bet — pot odds insuficientes)"
            color    = "RED_FOLD"
        return _build_preflop_result(hand, position, action, decision, color, stack_bb)

    # Tier 3+ — fold vs 3-bet (OOP) ou call IP com pairs
    is_pair = len(hand) == 2 and hand[0] == hand[1]
    if tier == 3 and is_pair and position in _LATE_POSITIONS:
        action   = "CALL"
        decision = f"CALL → {villain_3bet:.1f}BB (set mining vs 3-bet)"
        color    = "GREEN_INTENSE"
    else:
        action   = "FOLD"
        decision = "FOLD (vs 3-bet)"
        color    = "RED_FOLD"

    return _build_preflop_result(hand, position, action, decision, color, stack_bb)


def _build_preflop_result(
    hand: str, position: str, action: str, decision: str,
    color: str, stack_bb: float, sizing: float = 0.0,
) -> dict:
    tier_info     = _get_tier_info(hand)
    tier_label    = f"Tier {tier_info['tier']}" if tier_info else "—"
    return {
        "has_board":      False,
        "hand":           hand,
        "hand_canonical": hand,
        "position":       position,
        "action":         action,
        "sizing_bb":      sizing,
        "decision":       decision,
        "color_code":     color,
        "ev":             "—",
        "equity":         f"{round(tier_info['preflop_equity'] * 100, 1)}%" if tier_info else "—",
        "tier_label":     tier_label,
        "hand_label":     tier_info["label"] if tier_info else hand,
    }


# =============================================================================
# DECISÃO PÓS-FLOP
# =============================================================================

def evaluate_postflop(
    hand: str, position: str,
    call_size_bb: float, pot_size_bb: float,
    board: list[str], is_multiway: bool = False,
) -> dict:
    """Pós-flop: Monte Carlo → decisão binária FOLD / BET."""
    villain_pos = "BB" if position in _LATE_POSITIONS else "BTN"
    # Sem stack efetivo explícito no protocolo pós-flop, usa proxy conservador do tamanho do pote/ação.
    stack_proxy_bb = max(float(pot_size_bb), float(call_size_bb) * 3.0, 20.0)

    # Monte Carlo
    if _MC_AVAILABLE:
        try:
            result = _MCIntegrator.full_analysis_mc(
                hero_hand_str=hand, board_strs=board,
                call_size_bb=call_size_bb, pot_size_bb=pot_size_bb,
                villain_position=villain_pos, stack_bb=stack_proxy_bb,
                mode="fast",
            )
            return _postflop_from_mc(result, hand, position, call_size_bb, pot_size_bb, board, is_multiway)
        except Exception:
            pass

    # Fallback — enumeração
    analysis = EquityCalculator.full_analysis(
        hero_hand_str=hand, board_strs=board,
        call_size_bb=call_size_bb, pot_size_bb=pot_size_bb,
    )
    return _postflop_from_analysis(analysis, hand, position, call_size_bb, pot_size_bb, board, is_multiway)


def _postflop_from_mc(
    mc: dict, hand: str, position: str,
    call_bb: float, pot_bb: float, board: list[str], multiway: bool,
) -> dict:
    """Constrói resultado binário a partir do MC."""
    ev_raw  = mc.get("ev_adjusted") if isinstance(mc.get("ev_adjusted"), (int, float)) else mc.get("ev")
    ev      = float(ev_raw) if isinstance(ev_raw, (int, float)) else 0.0
    equity_raw = mc.get("equity_adjusted") if mc.get("equity_adjusted") is not None else mc.get("equity")
    equity = float(equity_raw) if isinstance(equity_raw, (int, float)) else 0.0
    label  = mc.get("hand_label", "")

    # Normaliza escalas para coerência matemática interna.
    equity_ratio = equity / 100.0 if equity > 1.0 else max(0.0, min(1.0, equity))
    pot_odds = EquityCalculator.pot_odds(call_bb, pot_bb)
    pot_if_win = pot_bb + call_bb
    ev_consistent = EquityCalculator.expected_value(equity_ratio, pot_if_win, call_bb)

    mc["pot_odds"] = pot_odds
    mc["equity_adjusted"] = round(equity_ratio * 100.0, 1)
    mc["ev_mc_raw"] = round(ev, 2)
    mc["ev_adjusted"] = round(ev_consistent, 2)

    # Decisão binária
    if ev_consistent > 0:
        # Calcula bet sizing sugerido (33-75% do pote)
        bet_size  = _suggest_bet(pot_bb, equity_ratio * 100.0)
        decision  = f"BET/RAISE → {bet_size:.1f}BB"
        color     = "GREEN_INTENSE"
    else:
        decision = "FOLD"
        color    = "RED_FOLD"

    tier_info  = _get_tier_info(normalize_hand(hand))
    tier_label = f"Tier {tier_info['tier']}" if tier_info else "—"

    texture_key  = _detect_texture(board)

    mc["action"]      = "BET" if decision.startswith("BET") else "FOLD"
    mc["decision"]    = decision
    mc["color_code"]  = color
    mc["has_board"]   = True
    mc["hand"]        = hand
    mc["position"]    = position
    mc["tier_label"]  = tier_label
    mc["texture_key"] = texture_key
    return mc


def _postflop_from_analysis(
    analysis: dict, hand: str, position: str,
    call_bb: float, pot_bb: float, board: list[str], multiway: bool,
) -> dict:
    texture_key  = _detect_texture(board)
    texture_meta = BOARD_TEXTURE_WEIGHTS[texture_key]
    equity_raw   = analysis["equity"] / 100.0

    outs_data = analysis.get("outs_data", {})
    if outs_data and outs_data.get("outs_total", 0) > 0:
        adj_equity = min(equity_raw * texture_meta["draw_mod"], 1.0)
    else:
        adj_equity = min(equity_raw * texture_meta["made_hand_mod"], 1.0)

    pot_odds_val = EquityCalculator.pot_odds(call_bb, pot_bb)
    pot_if_win   = pot_bb + call_bb
    ev_adj       = EquityCalculator.expected_value(adj_equity, pot_if_win, call_bb)

    if ev_adj > 0:
        bet_size = _suggest_bet(pot_bb, adj_equity * 100)
        decision = f"BET/RAISE → {bet_size:.1f}BB"
        color    = "GREEN_INTENSE"
    else:
        decision = "FOLD"
        color    = "RED_FOLD"

    tier_info  = _get_tier_info(normalize_hand(hand))
    tier_label = f"Tier {tier_info['tier']}" if tier_info else "—"
    action = "BET" if decision.startswith("BET") else "FOLD"
    stack_range = classify_stack(pot_bb)
    return {
        **analysis,
        "equity_adjusted": round(adj_equity * 100, 1),
        "ev_adjusted":     round(ev_adj, 2),
        "texture_key":     texture_key,
        "texture_note":    f"Textura: {texture_meta['label']}",
        "action":          action,
        "decision":        decision,
        "color_code":      color,
        "tier_label":      tier_label,
        "has_board":       True,
        "hand":            hand,
        "position":        position,
        "is_multiway":     multiway,
        "stack_range":     stack_range.value,
        "action_distribution": get_action_distribution(decision, stack_range),
    }


def _suggest_bet(pot_bb: float, equity_pct: float) -> float:
    """
    Sugestão de tamanho de bet baseado na equidade.
      equity > 70% → 75% do pote (bet grosso para proteção)
      equity 55-70% → 50% do pote
      equity < 55%  → 33% do pote (small bet ou semi-bluff)
    """
    if equity_pct >= 70:
        return round(pot_bb * 0.75, 1)
    if equity_pct >= 55:
        return round(pot_bb * 0.50, 1)
    return round(pot_bb * 0.33, 1)


# =============================================================================
# ENTRADA UNIFICADA
# =============================================================================

def evaluate_action(
    hand: str,
    position: str,
    open_size_bb: float,
    eff_stack_bb: float,
    is_multiway: bool,
    board: list[str] | None = None,
    bb_chips: float = 0.0,
    ante_chips: float = 0.0,
    custom_open_mult: float | None = None,
    limpers: int = 0,
    is_3bet_spot: bool = False,
) -> dict:
    """
    Ponto de entrada único — roteador principal.

    board vazio/None  → pré-flop
    board presente    → pós-flop (Monte Carlo)
    """
    canonical = normalize_hand(hand)

    if board:
        return {
            "decision": "WARNING",
            "action": "WARNING",
            "warning": "postflop_decision_moved_to_analysis_service",
            "abstained": True,
            "confidence": 0.0,
            "rationale": "postflop_ex_post_only",
            "source": "compatibility_wrapper",
            "hand": hand,
            "hand_canonical": canonical,
            "position": position,
            "has_board": True,
            "open_size_bb": open_size_bb,
            "eff_stack_bb": eff_stack_bb,
            "is_multiway": is_multiway,
            "decision_policy_version": "2026.04.ex-ante.v1",
        }

    from decision_service.inference import predict_ex_ante
    from decision_service.models import ExAnteDecisionRequest

    result = predict_ex_ante(
        ExAnteDecisionRequest(
            hand=canonical,
            position=position,
            stack_bb=float(eff_stack_bb),
            pot_bb_before=float(open_size_bb or 0.0),
            num_players=3 if is_multiway else 2,
            limpers=int(limpers or 0),
            open_size_bb=float(open_size_bb or 0.0),
            street="PREFLOP",
            ante_bb=float(ante_chips or 0.0) / float(bb_chips or 1.0) if bb_chips else 0.0,
            bb_chips=float(bb_chips or 100.0),
            is_3bet_spot=bool(is_3bet_spot),
        )
    )

    payload = result.__dict__.copy()
    payload.update(
        {
            "hand": hand,
            "hand_canonical": canonical,
            "position": position,
            "open_size_bb": open_size_bb,
            "eff_stack_bb": eff_stack_bb,
            "is_multiway": is_multiway,
            "has_board": False,
            "decision": payload.get("action", "WARNING"),
            "decision_detail": payload.get("rationale", ""),
            "decision_policy_version": payload.get("schema_version", "2026.04.ex-ante.v1"),
        }
    )
    return payload
    canonical = normalize_hand(hand)
    has_board = bool(board)

    if not has_board:
        result = evaluate_preflop(
            canonical, position, open_size_bb, eff_stack_bb,
            is_multiway,
            bb_chips=bb_chips,
            ante_chips=ante_chips,
            custom_open_mult=custom_open_mult,
            limpers=limpers,
            is_3bet_spot=is_3bet_spot,
        )
        result["has_board"] = False
        result["hand"] = hand
        result["hand_canonical"] = canonical
        result["position"] = position
        result["open_size_bb"] = open_size_bb
        result["eff_stack_bb"] = eff_stack_bb
        result["is_multiway"] = is_multiway
        tier_info = _get_tier_info(canonical)
        result["preflop_equity"] = round(tier_info["preflop_equity"] * 100, 1) if tier_info else None
        result = _apply_ml_adjustment(
            result,
            canonical,
            position,
            open_size_bb,
            eff_stack_bb,
            is_multiway,
            None,
            bb_chips,
        )
        result = _finalize_decision_payload(result)
        return MathValidator.validate_and_correct(result)

    result = evaluate_postflop(
        canonical, position, open_size_bb, eff_stack_bb,
        board, is_multiway,
    )
    result["has_board"] = True
    result["hand"] = hand
    result["hand_canonical"] = canonical
    result["position"] = position
    stack_range = classify_stack(eff_stack_bb)
    result["stack_range"] = stack_range.value
    result["action_distribution"] = get_action_distribution(result.get("decision", ""), stack_range)
    result["open_raise_chips"] = None
    result["open_raise_str"] = None
    result = _apply_ml_adjustment(
        result,
        canonical,
        position,
        open_size_bb,
        eff_stack_bb,
        is_multiway,
        board,
        bb_chips,
    )
    result = _finalize_decision_payload(result)
    return MathValidator.validate_and_correct(result)


# =============================================================================
# Helpers
# =============================================================================

def _detect_texture(board: list[str]) -> str:
    if not board:
        return "dry"
    cards = [DeckManager.parse_card(c) for c in board]
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]

    if len(set(suits)) == 1:
        return "monotone"

    rf: dict[int, int] = {}
    for r in ranks:
        rf[r] = rf.get(r, 0) + 1
    if max(rf.values()) >= 2:
        return "paired"

    sf: dict[str, int] = {}
    for s in suits:
        sf[s] = sf.get(s, 0) + 1
    if max(sf.values()) >= 2:
        return "wet"

    sr = sorted(set(ranks))
    for i in range(len(sr) - 2):
        if sr[i + 2] - sr[i] <= 3:
            return "wet"
    return "dry"


# =============================================================================
# Test suite
# =============================================================================

if __name__ == "__main__":
    SEP = "=" * 70
    print(SEP)
    print(f"{'decision_engine v5 — Test Suite':^70}")
    print(SEP)

    print("\n── Bug fixes ──\n")
    for h in ["AAs", "AAo", "KKs", "KKo", "QQs"]:
        n = normalize_hand(h)
        print(f"  normalize_hand('{h}') → '{n}'")

    print("\n── Pré-flop — decisões binárias ──\n")
    tests = [
        ("AA UTG  — RFI",        "AA",  "UTG", 0,   100, False, 0, False),
        ("AKs BTN — RFI",        "AKs", "BTN", 0,   100, False, 0, False),
        ("72o UTG — FOLD",        "72o", "UTG", 0,   100, False, 0, False),
        ("T9s CO  — RFI",        "T9s", "CO",  0,    80, False, 0, False),
        ("86o SB  — FOLD",        "86o", "SB",  0,    45, False, 0, False),
        ("JJ  BTN vs open 2.5",   "JJ",  "BTN", 2.5, 100, False, 0, False),
        ("AKs BB  vs open 2.5",   "AKs", "BB",  2.5, 100, False, 0, False),
        ("AA  BTN vs 3-bet 9bb",  "AA",  "BTN", 9.0,  80, False, 0, True),
        ("KK  CO  vs 3-bet 9bb",  "KK",  "CO",  9.0,  80, False, 0, True),
        ("TT  HJ  vs 3-bet 9bb",  "TT",  "HJ",  9.0,  80, False, 0, True),
        ("77  BTN stack 12bb",     "77",  "BTN", 0,    12, False, 0, False),
        ("AA  SB  stack 8bb",      "AA",  "SB",  0,     8, False, 0, False),
        ("RFI com 1 limp BTN",    "ATs", "BTN", 0,    80, False, 1, False),
    ]

    for desc, hand, pos, op, stk, mw, lmp, three in tests:
        res = evaluate_action(hand, pos, op, stk, mw,
                              limpers=lmp, is_3bet_spot=three)
        print(f"  [{pos}] {hand:<5} {desc:<35} → {res['decision']}")

    print(f"\n{SEP}")