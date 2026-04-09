"""
performance_drift.py
====================
Metrica de evolucao semanal para comparar a semana atual com a anterior.

Objetivos:
- medir drift tecnico com base em dados reais
- comparar bb/100 por posicao
- medir reducao de ICM_SUICIDE
- acompanhar PFR no SB em 15-25bb
- gerar serie temporal de erros tecnicos para o dashboard
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from leak_analysis.modules import ContextAnalyzer, LeakDetector, SeverityScorer, StudyPlanner, execute_query
from leak_analysis.modules.analysis_utils import normalize_position, safe_float


@dataclass
class WeeklyWindow:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


def _network_filter_clause(network: str) -> str:
    network_up = str(network or "WPN").upper()
    if network_up in {"WPN", "ACR"}:
        return "AND UPPER(COALESCE(s.source_file, '')) LIKE '%ACR%'"
    if network_up in {"POKERSTARS", "PS"}:
        return "AND UPPER(COALESCE(s.source_file, '')) LIKE '%POKERSTARS%'"
    return ""


def load_recent_hands(hero_name: str = "AurelioDizzy", network: str = "WPN", limit_hands: int = 50000) -> pd.DataFrame:
    filter_clause = _network_filter_clause(network)
    query = f"""
    SELECT
        h.id, h.hand_id, h.session_id, h.date_utc,
        h.hero_name, h.hero_position, h.hero_stack_start, h.big_blind,
        h.m_ratio, h.hero_vpip, h.hero_pfr, h.hero_amount_won, h.hero_result,
        h.hero_action_preflop, h.hero_action_flop, h.hero_action_turn, h.hero_action_river,
        h.board_flop, h.board_turn, h.board_river,
        s.source_file, s.tournament_name
    FROM hands h
    JOIN sessions s ON s.session_id = h.session_id
    WHERE UPPER(COALESCE(h.hero_name, '')) = UPPER(%s)
    {filter_clause}
    ORDER BY h.date_utc DESC
    LIMIT %s
    """
    rows = execute_query(query, (hero_name, limit_hands), fetch="all") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _prepare(df_hands: pd.DataFrame) -> pd.DataFrame:
    if df_hands is None or df_hands.empty:
        return pd.DataFrame()

    df = df_hands.copy()
    df["date_utc"] = pd.to_datetime(df.get("date_utc"), errors="coerce")
    df = df.dropna(subset=["date_utc"]).sort_values("date_utc")
    if "hero_position" in df.columns:
        df["hero_position"] = df["hero_position"].map(normalize_position)
    else:
        df["hero_position"] = "UNKNOWN"

    df["big_blind"] = pd.to_numeric(df.get("big_blind", 0), errors="coerce")
    df["hero_amount_won"] = pd.to_numeric(df.get("hero_amount_won", 0), errors="coerce").fillna(0.0)
    df["hero_pfr"] = pd.to_numeric(df.get("hero_pfr", 0), errors="coerce").fillna(0.0)
    df["hero_vpip"] = pd.to_numeric(df.get("hero_vpip", 0), errors="coerce").fillna(0.0)
    df["hero_stack_start"] = pd.to_numeric(df.get("hero_stack_start", 0), errors="coerce").fillna(0.0)
    return df


def _week_windows(anchor: pd.Timestamp) -> tuple[WeeklyWindow, WeeklyWindow]:
    anchor = pd.Timestamp(anchor).normalize()
    current_start = anchor - timedelta(days=6)
    previous_start = current_start - timedelta(days=7)
    previous_end = current_start
    current_end = anchor + timedelta(days=1)
    return (
        WeeklyWindow("Semana Anterior", previous_start, previous_end),
        WeeklyWindow("Semana Atual", current_start, current_end),
    )


def _slice_window(df: pd.DataFrame, window: WeeklyWindow) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    mask = (df["date_utc"] >= window.start) & (df["date_utc"] < window.end)
    return df.loc[mask].copy()


def _bb100(df: pd.DataFrame) -> float:
    if df.empty or "big_blind" not in df.columns:
        return 0.0
    bb = pd.to_numeric(df["big_blind"], errors="coerce")
    bb = bb.replace(0, pd.NA)
    won = pd.to_numeric(df["hero_amount_won"], errors="coerce").fillna(0.0)
    bb_won = (won / bb).fillna(0.0)
    hands = max(1, len(df))
    return float((bb_won.sum() / hands) * 100.0)


def _position_bb100(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    out: Dict[str, float] = {}
    for pos, group in df.groupby("hero_position", dropna=False):
        out[str(pos)] = round(_bb100(group), 4)
    return out


def _sb_pfr_15_25bb(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    bb = pd.to_numeric(df["big_blind"], errors="coerce").replace(0, pd.NA)
    stack_bb = (pd.to_numeric(df["hero_stack_start"], errors="coerce") / bb).fillna(0.0)
    subset = df[(df["hero_position"] == "SB") & stack_bb.between(15, 25, inclusive="both")]
    if subset.empty:
        return 0.0
    return float(pd.to_numeric(subset["hero_pfr"], errors="coerce").fillna(0.0).mean())


def _detect_daily_technical_errors(df: pd.DataFrame, window_label: str) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    series: List[Dict[str, Any]] = []
    df = df.copy()
    df["day"] = pd.to_datetime(df["date_utc"], errors="coerce").dt.normalize()
    for day, group in df.groupby("day", dropna=False):
        if pd.isna(day):
            continue
        if len(group) < 12:
            errors = 0
            icm_errors = 0
        else:
            leaks = LeakDetector(group).detect_all()
            errors = len(leaks)
            icm_errors = sum(1 for leak in leaks if getattr(leak, "leak_code", "") in {"ICM_OVERFOLD", "ICM_OVERCALL", "ICM_SUICIDE", "ICM_PASSIVITY"})
        series.append(
            {
                "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                "week_label": window_label,
                "technical_errors": int(errors),
                "icm_errors": int(icm_errors),
            }
        )
    return sorted(series, key=lambda r: r["date"])


def _window_summary(df: pd.DataFrame, window: WeeklyWindow) -> Dict[str, Any]:
    slice_df = _slice_window(df, window)
    if slice_df.empty:
        return {
            "label": window.label,
            "hands": 0,
            "bb100": 0.0,
            "position_bb100": {},
            "icm_suicide": 0,
            "sb_pfr_15_25bb": 0.0,
            "technical_errors_series": [],
        }

    detector = LeakDetector(slice_df)
    leaks = detector.detect_all()
    ctx = ContextAnalyzer(slice_df)
    ctx.analyze(leaks)
    contexts = ctx.aggregate_by_context()
    scorer = SeverityScorer(contexts)
    scorer.K_SKEPTICISM = 70.0
    scores = scorer.score_all()
    planner = StudyPlanner(scores, leaks)
    planner.plan_study()

    icm_suicide = sum(1 for leak in leaks if getattr(leak, "leak_code", "") == "ICM_SUICIDE")

    return {
        "label": window.label,
        "hands": int(len(slice_df)),
        "bb100": round(_bb100(slice_df), 4),
        "position_bb100": _position_bb100(slice_df),
        "icm_suicide": int(icm_suicide),
        "sb_pfr_15_25bb": round(_sb_pfr_15_25bb(slice_df), 4),
        "study_top_1": planner.to_json()[0] if planner.recommendations else {},
    }


def build_performance_drift_report(
    df_hands: Optional[pd.DataFrame] = None,
    hero_name: str = "AurelioDizzy",
    network: str = "WPN",
    limit_hands: int = 50000,
) -> Dict[str, Any]:
    """Compara a semana atual com a anterior e monta a serie temporal de erros."""
    if df_hands is None or df_hands.empty:
        df_hands = load_recent_hands(hero_name=hero_name, network=network, limit_hands=limit_hands)

    if df_hands is None or df_hands.empty:
        return {"status": "no_data", "technical_error_series": [], "kpis": {}}

    df = _prepare(df_hands)
    if df.empty:
        return {"status": "no_data", "technical_error_series": [], "kpis": {}}

    anchor = pd.Timestamp(df["date_utc"].max())
    previous_window, current_window = _week_windows(anchor)

    current_slice = _slice_window(df, current_window)
    previous_slice = _slice_window(df, previous_window)

    current_summary = _window_summary(df, current_window)
    previous_summary = _window_summary(df, previous_window)

    current_errors = current_summary.get("icm_suicide", 0)
    previous_errors = previous_summary.get("icm_suicide", 0)
    icm_reduction_pct = 0.0
    if previous_errors > 0:
        icm_reduction_pct = ((previous_errors - current_errors) / previous_errors) * 100.0

    current_sb_pfr = safe_float(current_summary.get("sb_pfr_15_25bb", 0.0), 0.0)
    previous_sb_pfr = safe_float(previous_summary.get("sb_pfr_15_25bb", 0.0), 0.0)

    current_bb100 = safe_float(current_summary.get("bb100", 0.0), 0.0)
    previous_bb100 = safe_float(previous_summary.get("bb100", 0.0), 0.0)
    delta_bb100 = current_bb100 - previous_bb100

    current_positions = current_summary.get("position_bb100", {}) or {}
    previous_positions = previous_summary.get("position_bb100", {}) or {}
    position_delta: Dict[str, Dict[str, float]] = {}
    for pos in sorted(set(current_positions) | set(previous_positions)):
        cur = safe_float(current_positions.get(pos, 0.0), 0.0)
        prev = safe_float(previous_positions.get(pos, 0.0), 0.0)
        position_delta[pos] = {
            "current_bb100": round(cur, 4),
            "previous_bb100": round(prev, 4),
            "delta_bb100": round(cur - prev, 4),
        }

    current_series = _detect_daily_technical_errors(current_slice, current_window.label)
    previous_series = _detect_daily_technical_errors(previous_slice, previous_window.label)
    technical_error_series = previous_series + current_series

    summary_message = ""
    if icm_reduction_pct > 0:
        summary_message = f"Aurelio, voce reduziu seus erros de ICM em {icm_reduction_pct:.1f}% esta semana. Continue assim."
    elif icm_reduction_pct < 0:
        summary_message = f"Aurelio, seus erros de ICM subiram {abs(icm_reduction_pct):.1f}% nesta semana. Hora de ajustar ICM."
    else:
        summary_message = "Aurelio, o volume de ICM ficou estavel nesta semana. Mantenha a disciplina e continue comparando os buckets."

    return {
        "status": "ok",
        "anchor_date": anchor.strftime("%Y-%m-%d"),
        "previous_window": {
            "start": previous_window.start.strftime("%Y-%m-%d"),
            "end": (previous_window.end - timedelta(days=1)).strftime("%Y-%m-%d"),
            **previous_summary,
        },
        "current_window": {
            "start": current_window.start.strftime("%Y-%m-%d"),
            "end": (current_window.end - timedelta(days=1)).strftime("%Y-%m-%d"),
            **current_summary,
        },
        "kpis": {
            "delta_bb100": round(delta_bb100, 4),
            "current_bb100": round(current_bb100, 4),
            "previous_bb100": round(previous_bb100, 4),
            "icm_suicide_current": int(current_errors),
            "icm_suicide_previous": int(previous_errors),
            "icm_suicide_reduction_pct": round(icm_reduction_pct, 2),
            "sb_pfr_15_25bb_current": round(current_sb_pfr, 4),
            "sb_pfr_15_25bb_previous": round(previous_sb_pfr, 4),
            "sb_pfr_15_25bb_delta": round(current_sb_pfr - previous_sb_pfr, 4),
            "position_bb100_delta": position_delta,
            "summary_message": summary_message,
        },
        "technical_error_series": technical_error_series,
    }
