"""
Utilities compartilhadas para o módulo de análise de leaks.

Objetivo:
- padronizar buckets de stack e fase
- classificar board texture e line type de forma consistente
- fornecer validações numéricas seguras
- integrar, quando houver contexto suficiente, EV esperado via engine/MC
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
from typing import Any, Optional, Dict, List

import pandas as pd

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - fallback when sklearn is unavailable
    KMeans = None

log = logging.getLogger(__name__)

STACK_BUCKETS_GRANULAR: list[tuple[float, float, str, str]] = [
    (0.0, 10.0, "0-10bb", "LOW"),
    (10.0, 15.0, "10-15bb", "LOW"),
    (15.0, 20.0, "15-20bb", "LOW"),
    (20.0, 25.0, "20-25bb", "MID"),
    (25.0, 35.0, "25-35bb", "MID"),
    (35.0, 50.0, "35-50bb", "MID"),
    (50.0, float("inf"), "50bb+", "HIGH"),
]

PHASE_ORDER = ("EARLY", "MIDDLE", "BUBBLE", "ITM", "FINAL_TABLE")
POSITION_ORDER = ("UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    denominator = safe_float(denominator, 0.0)
    if denominator == 0:
        return default
    return safe_float(numerator, 0.0) / denominator


def clamp(value: float, minimum: float, maximum: float) -> float:
    value = safe_float(value, minimum)
    return max(minimum, min(maximum, value))


def normalize_position(position: Any) -> str:
    pos = str(position or "UNKNOWN").strip().upper()
    if pos in POSITION_ORDER:
        return pos
    if pos in {"UTG+1", "UTG1", "EP"}:
        return "LJ"
    if pos in {"MP", "MIDDLE"}:
        return "HJ"
    return pos or "UNKNOWN"


def classify_stack_bucket(stack_bb: float) -> tuple[str, str]:
    """Retorna (stack_bucket_granular, stack_depth_bucket)."""
    stack_bb = safe_float(stack_bb, 0.0)
    for lo, hi, label, summary in STACK_BUCKETS_GRANULAR:
        if lo <= stack_bb < hi:
            return label, summary
    return "50bb+", "HIGH"


def classify_tournament_phase(row: Any) -> str:
    """Classifica fase do torneio usando fase explícita ou m-ratio."""
    getter = getattr(row, "get", lambda k, d=None: d)
    explicit = str(getter("tournament_phase", None) or "").strip().upper()
    if explicit in PHASE_ORDER:
        return explicit

    m_ratio = safe_float(getter("m_ratio", None), 0.0)
    if m_ratio <= 0:
        return "UNKNOWN"
    if m_ratio > 20:
        return "EARLY"
    if m_ratio > 10:
        return "MIDDLE"
    if m_ratio > 5:
        return "BUBBLE"
    if m_ratio > 2:
        return "ITM"
    return "FINAL_TABLE"


def _parse_board_cards(board: Any) -> list[str]:
    if board is None:
        return []
    if isinstance(board, (list, tuple)):
        return [str(c).strip() for c in board if str(c).strip()]
    text = str(board).strip()
    if not text:
        return []
    if " " in text:
        return [token.strip() for token in text.split() if token.strip()]
    cards = re.findall(r"[2-9TJQKA][hdcsHDCS]", text)
    if cards:
        return [c[:2] for c in cards]
    if len(text) in {6, 8, 10}:
        return [text[i:i + 2] for i in range(0, len(text), 2)]
    return []


def classify_board_texture(board: Any) -> str:
    cards = _parse_board_cards(board)
    if len(cards) < 3:
        return "NO_BOARD"

    ranks = [c[0].upper() for c in cards if len(c) >= 2]
    suits = [c[1].lower() for c in cards if len(c) >= 2]

    paired = len(set(ranks)) < len(ranks)
    unique_suits = set(suits)
    monotone = len(unique_suits) == 1
    two_tone = len(unique_suits) == 2

    rank_order = "23456789TJQKA"
    rank_idx = sorted(rank_order.index(r) for r in ranks if r in rank_order)
    connected = False
    if len(rank_idx) >= 3:
        span = rank_idx[-1] - rank_idx[0]
        connected = span <= 4

    if monotone:
        return "MONOTONE"
    if paired and connected:
        return "PAIRED_CONNECTED"
    if paired:
        return "PAIRED"
    if two_tone and connected:
        return "WET_CONNECTED"
    if two_tone:
        return "WET"
    if connected:
        return "CONNECTED"
    return "DRY"


def _normalize_action_text(action: Any) -> str:
    text = str(action or "").strip().upper()
    text = text.replace("→", ":")
    text = text.replace("BET/RAISE", "RAISE")
    text = text.replace("ALL IN", "ALL-IN")
    return text


def parse_action_sequence(row: Any) -> list[str]:
    actions: list[str] = []
    getter = getattr(row, "get", lambda k, d=None: d)
    for field in ("hero_action_preflop", "hero_action_flop", "hero_action_turn", "hero_action_river"):
        value = getter(field, None)
        if value is None:
            continue
        normalized = _normalize_action_text(value)
        if normalized:
            actions.append(normalized)
    return actions


def classify_line_type(row: Any) -> str:
    actions = parse_action_sequence(row)
    if not actions:
        return "UNKNOWN"

    preflop = actions[0] if len(actions) > 0 else ""
    postflop = actions[1:] if len(actions) > 1 else []

    if any("ALL-IN" in action for action in actions):
        return "JAM"
    if "3BET" in preflop or "4BET" in preflop:
        return "3BET_POT"
    if "RAISE" in preflop or "BET" in preflop:
        if postflop and any("BET" in a for a in postflop):
            return "AGGRESSIVE_POSTFLOP"
        return "AGGRESSIVE_PREFLOP"
    if "CALL" in preflop and postflop and any("BET" in a for a in postflop):
        return "CALL_LEAD"
    if all("CHECK" in a or "FOLD" in a for a in actions if a):
        return "PASSIVE"
    return "MIXED"


def infer_opponent_type(row: Any) -> str:
    """
    Infere tipo de vilão a partir de métricas disponíveis.

    Se não houver colunas suficientes, retorna UNKNOWN em vez de inventar.
    """
    getter = getattr(row, "get", lambda k, d=None: d)
    villain_vpip = safe_float(getter("villain_vpip", None), -1.0)
    villain_pfr = safe_float(getter("villain_pfr", None), -1.0)
    villain_agg = safe_float(getter("villain_aggression", None), -1.0)
    villain_f3b = safe_float(getter("villain_fold_to_3bet", None), -1.0)
    villain_wtsd = safe_float(getter("villain_wtsd", None), -1.0)

    stats_available = any(v >= 0 for v in (villain_vpip, villain_pfr, villain_agg, villain_f3b, villain_wtsd))
    if not stats_available:
        return "unknown"

    vpip = max(villain_vpip, 0.0)
    pfr = max(villain_pfr, 0.0)
    agg = max(villain_agg, 0.0)
    f3b = max(villain_f3b, 0.0)
    wtsd = max(villain_wtsd, 0.0)
    stack_bb = safe_div(safe_float(getter("hero_stack_start", None), 0.0), max(safe_float(getter("big_blind", None), 1.0), 1.0), 0.0)

    if vpip >= 0.45 and pfr <= 0.12:
        return "calling station"
    if vpip >= 0.40 and pfr >= 0.25 and agg >= 0.60:
        return "maniaco"
    if vpip <= 0.22 and pfr >= 0.18 and agg >= 0.50:
        return "reg agressivo"
    if vpip <= 0.20 and f3b >= 0.60:
        return "fit-or-fold"
    if vpip >= 0.35 and pfr <= 0.10 and wtsd >= 0.35:
        return "passive"
    if vpip <= 0.18 and pfr <= 0.08:
        return "overfolder"
    if stack_bb <= 20 and agg >= 0.50:
        return "short stack pressionador"
    if vpip >= 0.28 and pfr <= 0.16:
        return "recreativo"
    return "unknown"


def stack_bucket_multiplier(stack_depth_bucket: str) -> float:
    summary = str(stack_depth_bucket or "").upper()
    return {
        "LOW": 1.35,
        "MID": 1.0,
        "HIGH": 0.85,
    }.get(summary, 1.0)


def context_distance_penalty(distance_score: float) -> float:
    return clamp(1.0 - safe_float(distance_score, 0.0), 0.0, 1.0)


@dataclass(frozen=True)
class EVEstimate:
    expected_ev: Optional[float]
    real_ev: Optional[float]
    source: str
    confidence: float
    note: str


RANGE_EV_BASELINES: dict[tuple[str, str], float] = {
    ("calling station", "CALL_LINE"): 0.12,
    ("calling station", "PASSIVE"): 0.08,
    ("reg agressivo", "3BET_POT"): -0.06,
    ("reg agressivo", "AGGRESSIVE_POSTFLOP"): -0.03,
    ("recreativo", "CALL_LINE"): 0.06,
    ("short stack pressionador", "JAM"): -0.04,
    ("maniaco", "AGGRESSIVE_POSTFLOP"): -0.02,
    ("fit-or-fold", "AGGRESSIVE_PREFLOP"): 0.05,
    ("overfolder", "AGGRESSIVE_PREFLOP"): 0.07,
}


def estimate_incomplete_info_ev(row: Any) -> EVEstimate:
    """
    Estima EV esperado quando nao ha showdown (contexto tipico de ACR).

    A estimativa usa baselines por tipo de vilao e line_type. Quando faltarem
    colunas essenciais, retorna EV indefinido com nota explicita.
    """
    getter = getattr(row, "get", lambda k, d=None: d)
    opponent_type = str(getter("_opponent_type", getter("opponent_type", "unknown")) or "unknown").strip().lower()
    line_type = str(getter("_line_type", getter("line_type", "UNKNOWN")) or "UNKNOWN").strip().upper()
    stack_bb = safe_float(getter("_stack_bb", getter("stack_bb", 0.0)), 0.0)
    bb = max(safe_float(getter("big_blind", None), 1.0), 1.0)
    real_ev = safe_div(safe_float(getter("hero_amount_won", None), 0.0), bb, 0.0)

    if stack_bb <= 0:
        stack_bb = safe_div(safe_float(getter("hero_stack_start", 0.0), 0.0), bb, 0.0)

    baseline = RANGE_EV_BASELINES.get((opponent_type, line_type), 0.0)
    _, stack_depth = classify_stack_bucket(stack_bb)
    stack_adj = {"LOW": -0.03, "MID": 0.0, "HIGH": 0.02}.get(stack_depth, 0.0)

    expected_ev = baseline + stack_adj
    confidence = 0.55 if opponent_type != "unknown" else 0.40
    if line_type in {"3BET_POT", "JAM"}:
        confidence += 0.10
    confidence = clamp(confidence, 0.20, 0.80)

    return EVEstimate(
        expected_ev=expected_ev,
        real_ev=real_ev,
        source="range_bridge",
        confidence=confidence,
        note=f"range bridge: opp={opponent_type}, line={line_type}, stack={stack_depth}",
    )


def estimate_expected_ev_from_row(row: Any) -> EVEstimate:
    """
    Tenta estimar EV esperado com as funções já existentes.

    Regras:
    - só usa decision_engine/Monte Carlo se houver contexto suficiente
    - nunca inventa EV quando faltam pot size / call size / board
    - retorna um motivo claro quando não puder calcular
    """
    getter = getattr(row, "get", lambda k, d=None: d)
    hero_cards = getter("hero_cards", None)
    board_flop = getter("board_flop", None)
    board_turn = getter("board_turn", None)
    board_river = getter("board_river", None)
    open_size_bb = safe_float(getter("open_size_bb", None), 0.0)
    call_size_bb = safe_float(getter("call_size_bb", None), 0.0)
    pot_size_bb = safe_float(getter("pot_size_bb", None), 0.0)
    stack_bb = safe_float(getter("hero_stack_start", None), 0.0)
    bb = safe_float(getter("big_blind", None), 0.0)
    stack_bb = safe_div(stack_bb, bb, 0.0) if bb > 0 else stack_bb

    board = [c for c in (board_flop, board_turn, board_river) if c]
    went_to_showdown = int(safe_float(getter("went_to_showdown", 0), 0.0))

    real_ev_bb = safe_div(safe_float(getter("hero_amount_won", None), 0.0), bb, 0.0) if bb > 0 else safe_float(getter("hero_amount_won", None), 0.0)

    if went_to_showdown == 0:
        return estimate_incomplete_info_ev(row)

    if not hero_cards:
        return EVEstimate(None, real_ev_bb, "no_cards", 0.0, "Hero cards ausentes")

    try:
        from decision_engine import evaluate_action, evaluate_postflop, normalize_hand
    except Exception as exc:  # pragma: no cover - fallback seguro
        return EVEstimate(None, real_ev_bb, "engine_unavailable", 0.0, f"decision_engine indisponivel: {exc}")

    hand = normalize_hand(str(hero_cards)) if not isinstance(hero_cards, str) else normalize_hand(hero_cards)
    num_players = int(max(2, safe_float(getter("num_players", None), 2.0)))
    is_multiway = num_players > 2

    if not board:
        if open_size_bb > 0 or call_size_bb > 0:
            result = evaluate_action(
                hand=hand,
                position=str(getter("hero_position", "UNKNOWN")),
                open_size_bb=open_size_bb,
                eff_stack_bb=stack_bb,
                is_multiway=is_multiway,
                board=None,
                bb_chips=bb,
                ante_chips=safe_float(getter("ante", None), 0.0),
                limpers=int(safe_float(getter("limpers", None), 0.0)),
                is_3bet_spot=bool(getter("is_3bet_spot", None)),
            )
            ev_value = result.get("ev")
            return EVEstimate(
                expected_ev=safe_float(ev_value, None) if isinstance(ev_value, (int, float)) else None,
                real_ev=real_ev_bb,
                source="decision_engine",
                confidence=0.7,
                note=str(result.get("decision", "")),
            )

    if board and call_size_bb > 0 and pot_size_bb > 0:
        result = evaluate_postflop(
            hand=hand,
            position=str(getter("hero_position", "UNKNOWN")),
            call_size_bb=call_size_bb,
            pot_size_bb=pot_size_bb,
            board=board,
            is_multiway=is_multiway,
        )
        ev_value = result.get("ev_adjusted", result.get("ev", None))
        return EVEstimate(
            expected_ev=safe_float(ev_value, None) if isinstance(ev_value, (int, float)) else None,
            real_ev=real_ev_bb,
            source="monte_carlo",
            confidence=0.8 if len(board) >= 3 else 0.6,
            note=str(result.get("decision", "")),
        )

    return EVEstimate(None, real_ev_bb, "insufficient_context", 0.0, "Contexto insuficiente para EV esperado")


def deduplicate_hands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicatas por hand_id + site_name quando possivel.
    Fallback para hand_id quando site_name nao existir.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out["hand_id"] = out.get("hand_id", pd.Series([None] * len(out))).astype(str)
    site_series = out.get("site_name", pd.Series(["UNKNOWN"] * len(out)))
    out["site_name"] = site_series.fillna("UNKNOWN").astype(str).str.upper()

    sort_cols = []
    if "date_utc" in out.columns:
        sort_cols.append("date_utc")
    if "id" in out.columns:
        sort_cols.append("id")
    if sort_cols:
        out = out.sort_values(sort_cols)

    before = len(out)
    out = out.drop_duplicates(subset=["hand_id", "site_name"], keep="last")
    dropped = before - len(out)
    if dropped > 0:
        log.warning("Deduplicacao removeu %s linhas (hand_id + site_name)", dropped)
    return out.reset_index(drop=True)


def _action_rate(series: pd.Series, pattern: str) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(series.astype(str).str.upper().str.contains(pattern, regex=True, na=False).mean())


def build_opponent_clusters(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Clusteriza oponentes em perfis operacionais usando K-Means.

    Retorna DataFrame com colunas:
      opponent_key, aggression_rate, call_rate, fold_rate, cbet_rate,
      opponent_cluster_id, opponent_cluster
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    key_col = None
    for candidate in ("villain_id", "opponent_id", "villain_name", "opponent_name"):
        if candidate in out.columns:
            key_col = candidate
            break

    if key_col is None:
        out["_opponent_key"] = out.get("_opponent_type", out.get("opponent_type", "unknown")).fillna("unknown").astype(str)
    else:
        out["_opponent_key"] = out[key_col].fillna("unknown").astype(str)

    action_series = out.get("hero_action_preflop", pd.Series([""] * len(out), index=out.index)).fillna("")
    flop_series = out.get("hero_action_flop", pd.Series([""] * len(out), index=out.index)).fillna("")

    grouped = []
    for opp_key, g in out.groupby("_opponent_key", dropna=False):
        grouped.append(
            {
                "opponent_key": str(opp_key),
                "samples": int(len(g)),
                "aggression_rate": _action_rate(g.get("hero_action_preflop", action_series), r"RAISE|3BET|4BET|ALL-IN|BET"),
                "call_rate": _action_rate(g.get("hero_action_preflop", action_series), r"CALL|FLAT"),
                "fold_rate": _action_rate(g.get("hero_action_preflop", action_series), r"FOLD"),
                "cbet_rate": _action_rate(g.get("hero_action_flop", flop_series), r"BET"),
            }
        )

    features = pd.DataFrame(grouped)
    if features.empty:
        return features

    if KMeans is None or len(features) < max(3, n_clusters):
        features["opponent_cluster_id"] = 0
        features["opponent_cluster"] = features.apply(
            lambda r: "Aggro" if r["aggression_rate"] >= 0.45 else ("Station" if r["call_rate"] >= 0.35 else "Passive"),
            axis=1,
        )
        return features

    n_clusters = max(2, min(n_clusters, len(features)))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    X = features[["aggression_rate", "call_rate", "fold_rate", "cbet_rate"]].to_numpy(dtype=float)
    labels = km.fit_predict(X)
    features["opponent_cluster_id"] = labels

    centers = km.cluster_centers_
    cluster_map: Dict[int, str] = {}
    for idx, center in enumerate(centers):
        agg, call, fold, _ = center.tolist()
        if agg >= max(call, fold):
            cluster_map[idx] = "Aggro"
        elif call >= max(agg, fold):
            cluster_map[idx] = "Station"
        else:
            cluster_map[idx] = "Passive"
    features["opponent_cluster"] = features["opponent_cluster_id"].map(cluster_map).fillna("Passive")
    return features


def compute_vop_deviation(observed_freq: float, baseline_freq: float) -> float:
    """Value of Participation deviation: desvio percentual vs baseline."""
    observed = clamp(safe_float(observed_freq, 0.0), 0.0, 1.0)
    baseline = clamp(safe_float(baseline_freq, 0.0), 0.0, 1.0)
    if baseline <= 0:
        return 0.0
    return abs(observed - baseline) / baseline


def detect_cascade_effect(row: Any) -> float:
    """
    Heurística de efeito cascata street-by-street.

    Retorna score 0..1, maior quando há agressão cedo seguida de colapso de linha.
    """
    getter = getattr(row, "get", lambda k, d=None: d)
    flop = str(getter("hero_action_flop", "") or "").upper()
    turn = str(getter("hero_action_turn", "") or "").upper()
    river = str(getter("hero_action_river", "") or "").upper()
    won = safe_float(getter("hero_amount_won", 0.0), 0.0)

    aggressive_flop = any(k in flop for k in ("BET", "RAISE"))
    collapse_turn = any(k in turn for k in ("CHECK", "FOLD"))
    collapse_river = any(k in river for k in ("CHECK", "FOLD"))

    score = 0.0
    if aggressive_flop and collapse_turn:
        score += 0.45
    if aggressive_flop and collapse_river:
        score += 0.35
    if won < 0:
        score += 0.20
    return clamp(score, 0.0, 1.0)
