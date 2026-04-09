"""
leak_analysis/modules/leak_detector.py
Detector conservador e auditável de leaks a partir do histórico de mãos.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from shared_logic import calculate_bubble_factor

from .analysis_utils import (
    classify_board_texture,
    classify_stack_bucket,
    classify_tournament_phase,
    classify_line_type,
    estimate_expected_ev_from_row,
    infer_opponent_type,
    normalize_position,
    safe_div,
    safe_float,
    estimate_incomplete_info_ev,
    build_opponent_clusters,
    compute_vop_deviation,
    detect_cascade_effect,
)
from .validation import DataQualityValidator

log = logging.getLogger(__name__)


LEAK_TYPES: Dict[str, Dict[str, Any]] = {
    "VPIP_LOOSE": {"category": "PREFLOP", "weight": 1.20, "name": "VPIP alto"},
    "VPIP_TIGHT": {"category": "PREFLOP", "weight": 0.80, "name": "VPIP baixo"},
    "PFR_LOW": {"category": "PREFLOP", "weight": 1.00, "name": "PFR baixo"},
    "CALL_EXCESSIVE": {"category": "PREFLOP", "weight": 1.05, "name": "Call excessivo"},
    "FOLD_EXCESSIVE": {"category": "PREFLOP", "weight": 1.05, "name": "Fold excessivo"},
    "THREE_BET_WRONG": {"category": "PREFLOP", "weight": 1.10, "name": "3-bet incorreto"},
    "FOUR_BET_WRONG": {"category": "PREFLOP", "weight": 1.10, "name": "4-bet incorreto"},
    "FOLD_VS_3BET_HIGH": {"category": "PREFLOP", "weight": 1.10, "name": "Fold vs 3-bet alto"},
    "C_BET_WEAK": {"category": "POSTFLOP", "weight": 1.00, "name": "C-bet fraco"},
    "C_BET_STRONG": {"category": "POSTFLOP", "weight": 0.90, "name": "C-bet excessivo"},
    "OVERPLAY_MEDIUM": {"category": "POSTFLOP", "weight": 1.15, "name": "Overplay de mãos médias"},
    "BLUFF_BAD": {"category": "POSTFLOP", "weight": 1.20, "name": "Blefe ruim"},
    "VALUE_BET_LOST": {"category": "POSTFLOP", "weight": 1.10, "name": "Value bets perdidas"},
    "SIZING_POOR": {"category": "SIZING", "weight": 1.00, "name": "Sizing ruim"},
    "OVERBETTING": {"category": "SIZING", "weight": 1.05, "name": "Overbetting"},
    "UNDERBETTING": {"category": "SIZING", "weight": 0.90, "name": "Underbetting"},
    "ICM_OVERFOLD": {"category": "ICM", "weight": 1.20, "name": "Overfold na bolha"},
    "ICM_OVERCALL": {"category": "ICM", "weight": 1.20, "name": "Overcall de eliminação na bolha"},
    "ICM_SUICIDE": {"category": "ICM", "weight": 1.30, "name": "ICM suicide"},
    "ICM_PASSIVITY": {"category": "ICM", "weight": 1.18, "name": "ICM passivity"},
    "TIMING_IMPULSIVE": {"category": "DISCIPLINE", "weight": 1.10, "name": "Decisão rápida em spot complexo"},
    "TIMING_TELEGRAPH": {"category": "DISCIPLINE", "weight": 1.00, "name": "Delay suspeito em blefe"},
    "TILT_PATTERN": {"category": "DISCIPLINE", "weight": 1.35, "name": "Padrão de tilt"},
}

BENCHMARKS = {
    "LOW": {"vpip_min": 0.15, "vpip_max": 0.42, "pfr_min": 0.10, "pfr_max": 0.38, "call_max": 0.28, "fold_max": 0.62},
    "MID": {"vpip_min": 0.12, "vpip_max": 0.38, "pfr_min": 0.08, "pfr_max": 0.34, "call_max": 0.24, "fold_max": 0.58},
    "HIGH": {"vpip_min": 0.10, "vpip_max": 0.34, "pfr_min": 0.06, "pfr_max": 0.30, "call_max": 0.20, "fold_max": 0.54},
}

ACTION_RE = re.compile(r"(FOLD|CALL|CHECK|BET|RAISE|3BET|4BET|ALL-IN)", re.I)


@dataclass
class LeakDetection:
    hand_id: Any
    leak_code: str
    leak_name: str
    category: str
    confidence: float
    error_severity: float
    desvio_gto_pct: float
    context: Dict[str, Any]
    sample_size: int = 0
    ev_expected: Optional[float] = None
    ev_real: Optional[float] = None
    ev_delta: Optional[float] = None
    confidence_reason: str = ""
    evidence: List[str] = field(default_factory=list)


class LeakDetector:
    """Detecta leaks de forma conservadora e auditável."""

    MIN_SAMPLE_DEFAULT = 15
    MIN_SAMPLE_POSTFLOP = 20

    def __init__(self, df_hands: pd.DataFrame):
        self.df = df_hands.copy() if df_hands is not None else pd.DataFrame()
        self.detections: List[LeakDetection] = []
        self.validation = DataQualityValidator(min_rows=self.MIN_SAMPLE_DEFAULT).validate(self.df)

    def detect_all(self) -> List[LeakDetection]:
        log.info("Iniciando detecção de leaks com %s linhas", len(self.df))
        if not self.validation.ok:
            log.warning("DataFrame com problemas estruturais: %s", self.validation.to_dict())

        if self.df.empty:
            return []

        df = self._prepare_df(self.df)
        self._detect_vpip_leaks(df)
        self._detect_pfr_leaks(df)
        self._detect_fold_patterns(df)
        self._detect_call_patterns(df)
        self._detect_3bet_leaks(df)
        self._detect_postflop_leaks(df)
        self._detect_sizing_leaks(df)
        self._detect_icm_pressure(df)
        self._detect_timing_tells(df)
        self._detect_tilt_patterns(df)

        self.detections.sort(key=lambda x: (x.error_severity, x.confidence, x.sample_size), reverse=True)
        log.info("Detectados %s leaks", len(self.detections))
        return self.detections

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["_big_blind"] = pd.to_numeric(out.get("big_blind", 0), errors="coerce").replace([np.inf, -np.inf], np.nan)
        out["_hero_stack_start"] = pd.to_numeric(out.get("hero_stack_start", 0), errors="coerce").replace([np.inf, -np.inf], np.nan)
        out["_stack_bb"] = out.apply(lambda r: safe_div(safe_float(r.get("hero_stack_start"), 0.0), max(safe_float(r.get("big_blind"), 1.0), 1.0), 0.0), axis=1)
        out["_hero_amount_bb"] = out.apply(
            lambda r: safe_div(safe_float(r.get("hero_amount_won"), 0.0), max(safe_float(r.get("big_blind"), 1.0), 1.0), 0.0),
            axis=1,
        )
        out["_stack_depth_bucket"], out["_stack_bucket"] = zip(*out["_stack_bb"].map(classify_stack_bucket))
        out["_position"] = out.get("hero_position", pd.Series(["UNKNOWN"] * len(out))).map(normalize_position)
        out["_phase"] = out.apply(classify_tournament_phase, axis=1)
        out["_board_texture"] = out.get("board_flop", pd.Series([None] * len(out))).map(classify_board_texture)
        out["_line_type"] = out.apply(classify_line_type, axis=1)
        out["_opponent_type"] = out.apply(infer_opponent_type, axis=1)
        cluster_df = build_opponent_clusters(out)
        if not cluster_df.empty:
            cluster_map = dict(zip(cluster_df["opponent_key"], cluster_df["opponent_cluster"]))
            cluster_key = out.get("villain_id", out.get("opponent_id", out.get("villain_name", out.get("opponent_name", out["_opponent_type"]))))
            out["_opponent_cluster"] = cluster_key.fillna(out["_opponent_type"]).astype(str).map(cluster_map).fillna("Passive")
        else:
            out["_opponent_cluster"] = out["_opponent_type"].map(
                lambda v: "Aggro" if str(v).lower() in {"reg agressivo", "maniaco", "short stack pressionador"} else ("Station" if str(v).lower() in {"calling station", "recreativo"} else "Passive")
            )
        out["_action_pf"] = out.get("hero_action_preflop", pd.Series([""] * len(out))).fillna("").astype(str).str.upper()
        out["_action_flop"] = out.get("hero_action_flop", pd.Series([""] * len(out))).fillna("").astype(str).str.upper()
        out["_action_turn"] = out.get("hero_action_turn", pd.Series([""] * len(out))).fillna("").astype(str).str.upper()
        out["_action_river"] = out.get("hero_action_river", pd.Series([""] * len(out))).fillna("").astype(str).str.upper()
        out["_cascade_effect"] = out.apply(detect_cascade_effect, axis=1)
        return out

    def _series(self, df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(default)
        return pd.Series([default] * len(df), index=df.index)

    def _group_metric(self, df: pd.DataFrame) -> dict[str, Any]:
        sample_size = len(df)
        if sample_size == 0:
            return {}

        vpip = safe_float(self._series(df, "hero_vpip").mean(), 0.0)
        pfr = safe_float(self._series(df, "hero_pfr").mean(), 0.0)
        call_rate = safe_float(df["_action_pf"].str.contains("CALL|FLAT", regex=True, na=False).mean(), 0.0)
        fold_rate = safe_float(df["_action_pf"].str.contains("FOLD", regex=False, na=False).mean(), 0.0)
        raise_rate = safe_float(df["_action_pf"].str.contains("RAISE|3BET|4BET|ALL-IN", regex=True, na=False).mean(), 0.0)
        cbet_rate = safe_float(df["_action_flop"].str.contains("BET", regex=False, na=False).mean(), 0.0)
        river_bluff_rate = safe_float(((df["_action_river"].str.contains("BET|RAISE", regex=True, na=False)) & (self._series(df, "_hero_amount_bb") < 0)).mean(), 0.0)
        fold_to_3bet_rate = 0.0
        if "_line_type" in df.columns:
            spots_3bet = df[df["_line_type"].eq("3BET_POT")]
            if not spots_3bet.empty:
                fold_to_3bet_rate = safe_float(spots_3bet["_action_pf"].str.contains("FOLD", regex=False, na=False).mean(), 0.0)
        cascade_rate = safe_float(self._series(df, "_cascade_effect", 0.0).mean(), 0.0)
        total_ev_lost = float(max(0.0, -self._series(df, "_hero_amount_bb").sum()))
        avg_loss = float(total_ev_lost / sample_size) if sample_size else 0.0
        win_rate = float((self._series(df, "_hero_amount_bb") > 0).mean())

        return {
            "sample_size": sample_size,
            "vpip": vpip,
            "pfr": pfr,
            "call_rate": call_rate,
            "fold_rate": fold_rate,
            "raise_rate": raise_rate,
            "cbet_rate": cbet_rate,
            "river_bluff_rate": river_bluff_rate,
            "fold_to_3bet_rate": fold_to_3bet_rate,
            "cascade_rate": cascade_rate,
            "total_ev_lost": total_ev_lost,
            "avg_loss": avg_loss,
            "win_rate": win_rate,
        }

    def _emit_group_leak(self, df: pd.DataFrame, leak_code: str, metric_value: float, benchmark: float, context: dict[str, Any], severity_base: float, evidence: list[str], min_sample: int) -> None:
        sample_size = len(df)
        if sample_size < min_sample:
            return
        deviation = safe_div(abs(metric_value - benchmark), benchmark if benchmark else 1.0, 0.0) * 100.0
        showdown_rate = float(pd.to_numeric(df.get("went_to_showdown", 0), errors="coerce").fillna(0.0).mean()) if "went_to_showdown" in df.columns else 0.0
        confidence = min(1.0, 0.15 + (sample_size / (sample_size + 40.0)) + min(0.25, deviation / 200.0))
        # Em contexto ACR sem showdown, desvio de frequencia e mais confiavel que resultado financeiro.
        if showdown_rate < 0.35:
            confidence = min(1.0, confidence + min(0.20, deviation / 150.0))
            evidence.append(f"freq_priority_no_showdown={deviation:.1f}%")
        ev_real = float(pd.to_numeric(df.get("_hero_amount_bb", 0), errors="coerce").sum())
        ev_expected = None
        ev_delta = None

        rep_row = df.sort_values(by=["_hero_amount_bb"], ascending=True).iloc[0]
        estimate = estimate_expected_ev_from_row(rep_row)
        if estimate.expected_ev is None and int(safe_float(rep_row.get("went_to_showdown", 0), 0.0)) == 0:
            estimate = estimate_incomplete_info_ev(rep_row)
        if estimate.expected_ev is not None:
            ev_expected = estimate.expected_ev
            ev_delta = safe_float(ev_expected - safe_float(estimate.real_ev, 0.0), 0.0)

        detection = LeakDetection(
            hand_id=rep_row.get("id", rep_row.get("hand_id", None)),
            leak_code=leak_code,
            leak_name=LEAK_TYPES[leak_code]["name"],
            category=LEAK_TYPES[leak_code]["category"],
            confidence=confidence,
            error_severity=min(1.0, max(0.0, severity_base * (0.75 + confidence / 2.0))),
            desvio_gto_pct=deviation,
            context=context,
            sample_size=sample_size,
            ev_expected=ev_expected,
            ev_real=ev_real,
            ev_delta=ev_delta,
            confidence_reason=estimate.note if estimate.note else f"sample={sample_size}",
            evidence=evidence + [f"showdown_rate={showdown_rate:.2f}", f"ev_source={estimate.source}"],
        )
        self.detections.append(detection)

    def _position_adjustment(self, position: str) -> float:
        position = normalize_position(position)
        if position in {"UTG", "LJ", "HJ"}:
            return 0.90
        if position in {"CO", "BTN"}:
            return 1.05
        if position in {"SB", "BB"}:
            return 1.00
        return 1.0

    def _phase_adjustment(self, phase: str) -> float:
        return {"EARLY": 0.95, "MIDDLE": 1.0, "BUBBLE": 1.05, "ITM": 1.05, "FINAL_TABLE": 1.10}.get(phase, 1.0)

    def _detect_vpip_leaks(self, df: pd.DataFrame) -> None:
        for (stack_bucket, position, phase), group in df.groupby(["_stack_bucket", "_position", "_phase"], dropna=False):
            metric = self._group_metric(group)
            if not metric:
                continue
            benchmark = BENCHMARKS.get(stack_bucket, BENCHMARKS["MID"])
            adj = self._position_adjustment(position) * self._phase_adjustment(phase)
            vpip_max = benchmark["vpip_max"] * adj
            vpip_min = benchmark["vpip_min"] * min(1.0, adj + 0.05)
            context = {
                "stack_depth_bucket": group["_stack_depth_bucket"].mode().iat[0] if not group["_stack_depth_bucket"].mode().empty else "UNKNOWN",
                "stack_bucket": stack_bucket,
                "position": position,
                "phase": phase,
                "opponent_type": group["_opponent_type"].mode().iat[0] if not group["_opponent_type"].mode().empty else "unknown",
                "opponent_cluster": group["_opponent_cluster"].mode().iat[0] if "_opponent_cluster" in group.columns and not group["_opponent_cluster"].mode().empty else "Passive",
                "board_texture": group["_board_texture"].mode().iat[0] if not group["_board_texture"].mode().empty else "NO_BOARD",
                "line_type": group["_line_type"].mode().iat[0] if not group["_line_type"].mode().empty else "UNKNOWN",
            }
            if metric["vpip"] > vpip_max:
                self._emit_group_leak(group, "VPIP_LOOSE", metric["vpip"], vpip_max, context, 0.70, [f"vpip={metric['vpip']:.2f}", f"max={vpip_max:.2f}"], self.MIN_SAMPLE_DEFAULT)
            elif metric["vpip"] < vpip_min:
                self._emit_group_leak(group, "VPIP_TIGHT", metric["vpip"], vpip_min, context, 0.55, [f"vpip={metric['vpip']:.2f}", f"min={vpip_min:.2f}"], self.MIN_SAMPLE_DEFAULT)

    def _detect_pfr_leaks(self, df: pd.DataFrame) -> None:
        for (stack_bucket, position, phase), group in df.groupby(["_stack_bucket", "_position", "_phase"], dropna=False):
            metric = self._group_metric(group)
            if not metric:
                continue
            benchmark = BENCHMARKS.get(stack_bucket, BENCHMARKS["MID"])
            adj = self._position_adjustment(position) * self._phase_adjustment(phase)
            pfr_min = benchmark["pfr_min"] * adj
            pfr_max = benchmark["pfr_max"] * (adj + 0.05)
            context = {
                "stack_depth_bucket": group["_stack_depth_bucket"].mode().iat[0] if not group["_stack_depth_bucket"].mode().empty else "UNKNOWN",
                "stack_bucket": stack_bucket,
                "position": position,
                "phase": phase,
                "opponent_type": group["_opponent_type"].mode().iat[0] if not group["_opponent_type"].mode().empty else "unknown",
                "opponent_cluster": group["_opponent_cluster"].mode().iat[0] if "_opponent_cluster" in group.columns and not group["_opponent_cluster"].mode().empty else "Passive",
                "board_texture": group["_board_texture"].mode().iat[0] if not group["_board_texture"].mode().empty else "NO_BOARD",
                "line_type": group["_line_type"].mode().iat[0] if not group["_line_type"].mode().empty else "UNKNOWN",
            }
            if metric["pfr"] < pfr_min:
                self._emit_group_leak(group, "PFR_LOW", metric["pfr"], pfr_min, context, 0.65, [f"pfr={metric['pfr']:.2f}", f"min={pfr_min:.2f}"], self.MIN_SAMPLE_DEFAULT)
            elif metric["pfr"] > pfr_max:
                self._emit_group_leak(group, "C_BET_STRONG", metric["pfr"], pfr_max, context, 0.55, [f"pfr={metric['pfr']:.2f}", f"max={pfr_max:.2f}"], self.MIN_SAMPLE_DEFAULT)

    def _detect_fold_patterns(self, df: pd.DataFrame) -> None:
        for (stack_bucket, position, phase), group in df.groupby(["_stack_bucket", "_position", "_phase"], dropna=False):
            metric = self._group_metric(group)
            if not metric:
                continue
            benchmark = BENCHMARKS.get(stack_bucket, BENCHMARKS["MID"])
            fold_max = benchmark["fold_max"]
            if position in {"CO", "BTN", "SB", "BB"} and metric["fold_rate"] > fold_max:
                vop_dev = compute_vop_deviation(metric["fold_rate"], fold_max)
                self._emit_group_leak(
                    group,
                    "FOLD_EXCESSIVE",
                    metric["fold_rate"],
                    fold_max,
                    {"stack_depth_bucket": group["_stack_depth_bucket"].mode().iat[0] if not group["_stack_depth_bucket"].mode().empty else "UNKNOWN", "stack_bucket": stack_bucket, "position": position, "phase": phase, "opponent_type": group["_opponent_type"].mode().iat[0] if not group["_opponent_type"].mode().empty else "unknown", "board_texture": group["_board_texture"].mode().iat[0] if not group["_board_texture"].mode().empty else "NO_BOARD", "line_type": group["_line_type"].mode().iat[0] if not group["_line_type"].mode().empty else "UNKNOWN"},
                    0.60,
                    [f"fold_rate={metric['fold_rate']:.2f}", f"max={fold_max:.2f}", f"vop_dev={vop_dev:.2f}"],
                    self.MIN_SAMPLE_DEFAULT,
                )

    def _detect_call_patterns(self, df: pd.DataFrame) -> None:
        for (stack_bucket, position, phase), group in df.groupby(["_stack_bucket", "_position", "_phase"], dropna=False):
            metric = self._group_metric(group)
            if not metric:
                continue
            benchmark = BENCHMARKS.get(stack_bucket, BENCHMARKS["MID"])
            call_max = benchmark["call_max"]
            if metric["call_rate"] > call_max and position in {"UTG", "LJ", "HJ", "CO", "BTN"}:
                vop_dev = compute_vop_deviation(metric["call_rate"], call_max)
                self._emit_group_leak(
                    group,
                    "CALL_EXCESSIVE",
                    metric["call_rate"],
                    call_max,
                    {"stack_depth_bucket": group["_stack_depth_bucket"].mode().iat[0] if not group["_stack_depth_bucket"].mode().empty else "UNKNOWN", "stack_bucket": stack_bucket, "position": position, "phase": phase, "opponent_type": group["_opponent_type"].mode().iat[0] if not group["_opponent_type"].mode().empty else "unknown", "board_texture": group["_board_texture"].mode().iat[0] if not group["_board_texture"].mode().empty else "NO_BOARD", "line_type": group["_line_type"].mode().iat[0] if not group["_line_type"].mode().empty else "UNKNOWN"},
                    0.55,
                    [f"call_rate={metric['call_rate']:.2f}", f"max={call_max:.2f}", f"vop_dev={vop_dev:.2f}"],
                    self.MIN_SAMPLE_DEFAULT,
                )

    def _detect_3bet_leaks(self, df: pd.DataFrame) -> None:
        for (stack_bucket, position, phase), group in df.groupby(["_stack_bucket", "_position", "_phase"], dropna=False):
            metric = self._group_metric(group)
            if not metric:
                continue
            action_pf = group["_action_pf"]
            three_bet_rate = action_pf.str.contains("3BET", regex=False, na=False).mean()
            four_bet_rate = action_pf.str.contains("4BET", regex=False, na=False).mean()
            if three_bet_rate >= 0.08 and metric["sample_size"] >= self.MIN_SAMPLE_DEFAULT:
                self._emit_group_leak(
                    group,
                    "THREE_BET_WRONG",
                    three_bet_rate,
                    0.05,
                    {"stack_depth_bucket": group["_stack_depth_bucket"].mode().iat[0] if not group["_stack_depth_bucket"].mode().empty else "UNKNOWN", "stack_bucket": stack_bucket, "position": position, "phase": phase, "opponent_type": group["_opponent_type"].mode().iat[0] if not group["_opponent_type"].mode().empty else "unknown", "board_texture": group["_board_texture"].mode().iat[0] if not group["_board_texture"].mode().empty else "NO_BOARD", "line_type": group["_line_type"].mode().iat[0] if not group["_line_type"].mode().empty else "UNKNOWN"},
                    0.50,
                    [f"3bet_rate={three_bet_rate:.2f}"],
                    self.MIN_SAMPLE_DEFAULT,
                )
            if metric.get("fold_to_3bet_rate", 0.0) > 0.55 and metric["sample_size"] >= self.MIN_SAMPLE_DEFAULT:
                vop_dev = compute_vop_deviation(metric["fold_to_3bet_rate"], 0.55)
                self._emit_group_leak(
                    group,
                    "FOLD_VS_3BET_HIGH",
                    metric["fold_to_3bet_rate"],
                    0.55,
                    {"stack_depth_bucket": stack_bucket, "position": position, "phase": phase, "opponent_type": group["_opponent_type"].mode().iat[0] if not group["_opponent_type"].mode().empty else "unknown", "opponent_cluster": group["_opponent_cluster"].mode().iat[0] if "_opponent_cluster" in group.columns and not group["_opponent_cluster"].mode().empty else "Passive", "board_texture": group["_board_texture"].mode().iat[0] if not group["_board_texture"].mode().empty else "NO_BOARD", "line_type": "3BET_POT"},
                    0.62,
                    [f"fold_to_3bet={metric['fold_to_3bet_rate']:.2f}", f"baseline=0.55", f"vop_dev={vop_dev:.2f}"],
                    self.MIN_SAMPLE_DEFAULT,
                )
            if four_bet_rate >= 0.03 and metric["sample_size"] >= self.MIN_SAMPLE_DEFAULT:
                self._emit_group_leak(
                    group,
                    "FOUR_BET_WRONG",
                    four_bet_rate,
                    0.02,
                    {"stack_depth_bucket": stack_bucket, "position": position, "phase": phase, "opponent_type": group["_opponent_type"].mode().iat[0] if not group["_opponent_type"].mode().empty else "unknown", "board_texture": group["_board_texture"].mode().iat[0] if not group["_board_texture"].mode().empty else "NO_BOARD", "line_type": group["_line_type"].mode().iat[0] if not group["_line_type"].mode().empty else "UNKNOWN"},
                    0.55,
                    [f"4bet_rate={four_bet_rate:.2f}"],
                    self.MIN_SAMPLE_DEFAULT,
                )

    def _detect_postflop_leaks(self, df: pd.DataFrame) -> None:
        postflop_df = df[df[["_action_flop", "_action_turn", "_action_river"]].any(axis=1)]
        if postflop_df.empty:
            return

        for (stack_bucket, position, phase, board_texture), group in postflop_df.groupby(["_stack_bucket", "_position", "_phase", "_board_texture"], dropna=False):
            metric = self._group_metric(group)
            if not metric or metric["sample_size"] < self.MIN_SAMPLE_POSTFLOP:
                continue
            context = {
                "stack_depth_bucket": group["_stack_depth_bucket"].mode().iat[0] if not group["_stack_depth_bucket"].mode().empty else "UNKNOWN",
                "stack_bucket": stack_bucket,
                "position": position,
                "phase": phase,
                "board_texture": board_texture,
                "opponent_type": group["_opponent_type"].mode().iat[0] if not group["_opponent_type"].mode().empty else "unknown",
                "opponent_cluster": group["_opponent_cluster"].mode().iat[0] if "_opponent_cluster" in group.columns and not group["_opponent_cluster"].mode().empty else "Passive",
                "line_type": group["_line_type"].mode().iat[0] if not group["_line_type"].mode().empty else "UNKNOWN",
            }

            if metric["cbet_rate"] < 0.18 and position in {"BTN", "CO", "HJ"} and phase != "FINAL_TABLE":
                self._emit_group_leak(group, "C_BET_WEAK", metric["cbet_rate"], 0.25, context, 0.50, [f"cbet_rate={metric['cbet_rate']:.2f}"], self.MIN_SAMPLE_POSTFLOP)
            if metric["cbet_rate"] > 0.70 and stack_bucket in {"LOW", "MID"}:
                self._emit_group_leak(group, "C_BET_STRONG", metric["cbet_rate"], 0.65, context, 0.45, [f"cbet_rate={metric['cbet_rate']:.2f}"], self.MIN_SAMPLE_POSTFLOP)
            if metric["river_bluff_rate"] > 0.22 and metric["win_rate"] < 0.45:
                self._emit_group_leak(group, "BLUFF_BAD", metric["river_bluff_rate"], 0.15, context, 0.75, [f"river_bluff_rate={metric['river_bluff_rate']:.2f}", f"win_rate={metric['win_rate']:.2f}"], self.MIN_SAMPLE_POSTFLOP)
            if metric["avg_loss"] > 0 and metric["vpip"] > 0.30 and metric["pfr"] < 0.18:
                self._emit_group_leak(group, "OVERPLAY_MEDIUM", metric["avg_loss"], 0.0, context, 0.70, [f"avg_loss={metric['avg_loss']:.2f}", f"vpip={metric['vpip']:.2f}", f"pfr={metric['pfr']:.2f}"], self.MIN_SAMPLE_POSTFLOP)
            if metric.get("cascade_rate", 0.0) > 0.45:
                self._emit_group_leak(
                    group,
                    "BLUFF_BAD",
                    metric["cascade_rate"],
                    0.30,
                    context,
                    0.66,
                    [f"cascade_rate={metric['cascade_rate']:.2f}", "cascade_effect=flop->turn/river"],
                    self.MIN_SAMPLE_POSTFLOP,
                )

            # Value bets perdidas: vitória baixa em showdowns + agressão reduzida em turn/river
            went_sd = float(self._series(group, "went_to_showdown").mean()) if "went_to_showdown" in group.columns else 0.0
            if went_sd > 0.25 and metric["win_rate"] < 0.52 and metric["cbet_rate"] < 0.45:
                self._emit_group_leak(group, "VALUE_BET_LOST", went_sd, 0.25, context, 0.60, [f"wtsd={went_sd:.2f}", f"win_rate={metric['win_rate']:.2f}"], self.MIN_SAMPLE_POSTFLOP)

    def _detect_sizing_leaks(self, df: pd.DataFrame) -> None:
        size_cols = [c for c in df.columns if any(k in c.lower() for k in ("bet_size", "raise_size", "sizing", "amount_bet"))]
        if not size_cols:
            return
        # Apenas reporta quando há colunas explícitas de sizing.
        for col in size_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.dropna().empty:
                continue
            ratio = float((values > values.quantile(0.90)).mean())
            if ratio > 0.10:
                self._emit_group_leak(
                    df.dropna(subset=[col]),
                    "OVERBETTING",
                    ratio,
                    0.05,
                    {"stack_depth_bucket": "UNKNOWN", "stack_bucket": "UNKNOWN", "position": "UNKNOWN", "phase": "UNKNOWN", "opponent_type": "unknown", "board_texture": "UNKNOWN", "line_type": "UNKNOWN"},
                    0.45,
                    [f"{col} high frequency={ratio:.2f}"],
                    self.MIN_SAMPLE_DEFAULT,
                )

    def _detect_tilt_patterns(self, df: pd.DataFrame) -> None:
        if "date_utc" not in df.columns:
            return
        ordered = df.sort_values("date_utc").copy()
        if len(ordered) < 30:
            return

        amounts = self._series(ordered, "_hero_amount_bb")
        vpip_series = pd.to_numeric(ordered.get("hero_vpip", 0), errors="coerce").fillna(0.0)
        ordered["_loss_hand"] = amounts < 0

        # Detecta clusters de perda sequencial (>=3) e observa mudança de VPIP nas mãos subsequentes.
        streaks: list[tuple[int, int]] = []
        start = None
        for idx, is_loss in enumerate(ordered["_loss_hand"].tolist()):
            if is_loss and start is None:
                start = idx
            elif (not is_loss) and start is not None:
                if idx - start >= 3:
                    streaks.append((start, idx - 1))
                start = None
        if start is not None and len(ordered) - start >= 3:
            streaks.append((start, len(ordered) - 1))

        if not streaks:
            return

        baseline_window = max(20, min(80, len(ordered) // 2))
        baseline_vpip = float(vpip_series.head(baseline_window).mean())
        baseline_ev = float(amounts.head(baseline_window).mean())

        max_streak = 0
        worst_streak_score = 0.0
        representative = None
        vpip_deltas: list[float] = []
        ev_deltas: list[float] = []

        for (s, e) in streaks:
            streak_len = e - s + 1
            max_streak = max(max_streak, streak_len)
            follow_start = e + 1
            follow_end = min(len(ordered), follow_start + 6)
            if follow_start >= follow_end:
                continue
            follow_vpip = float(vpip_series.iloc[follow_start:follow_end].mean())
            follow_ev = float(amounts.iloc[follow_start:follow_end].mean())
            vpip_delta = follow_vpip - baseline_vpip
            ev_delta = baseline_ev - follow_ev
            vpip_deltas.append(vpip_delta)
            ev_deltas.append(ev_delta)

            score = max(0.0, vpip_delta) * 100.0 + max(0.0, ev_delta) * 15.0 + streak_len * 6.0
            if score > worst_streak_score:
                worst_streak_score = score
                representative = ordered.iloc[s]

        if representative is None:
            return

        avg_vpip_delta = float(np.mean(vpip_deltas)) if vpip_deltas else 0.0
        avg_ev_delta = float(np.mean(ev_deltas)) if ev_deltas else 0.0

        if avg_vpip_delta < 0.08 and max_streak < 4:
            return

        confidence = min(0.95, 0.45 + (len(streaks) / 20.0) + min(0.25, avg_vpip_delta))
        severity = min(1.0, 0.55 + min(0.25, max_streak / 10.0) + min(0.2, max(0.0, avg_vpip_delta)))
        desvio_pct = min(100.0, max(0.0, avg_vpip_delta) * 100.0)

        self.detections.append(
            LeakDetection(
                hand_id=representative.get("id", representative.get("hand_id", None)),
                leak_code="TILT_PATTERN",
                leak_name=LEAK_TYPES["TILT_PATTERN"]["name"],
                category=LEAK_TYPES["TILT_PATTERN"]["category"],
                confidence=confidence,
                error_severity=severity,
                desvio_gto_pct=desvio_pct,
                context={
                    "stack_depth_bucket": "UNKNOWN",
                    "stack_bucket": "UNKNOWN",
                    "position": normalize_position(representative.get("hero_position", "UNKNOWN")),
                    "phase": classify_tournament_phase(representative),
                    "opponent_type": infer_opponent_type(representative),
                    "board_texture": classify_board_texture(representative.get("board_flop")),
                    "line_type": classify_line_type(representative),
                },
                sample_size=len(ordered),
                ev_expected=None,
                ev_real=float(pd.to_numeric(df.get("_hero_amount_bb", 0), errors="coerce").sum()),
                ev_delta=None,
                confidence_reason="loss_streak + vpip_shift",
                evidence=[
                    f"streak_clusters={len(streaks)}",
                    f"max_streak={max_streak}",
                    f"vpip_delta_post_loss={avg_vpip_delta:.3f}",
                    f"ev_decay_post_loss={avg_ev_delta:.3f}",
                ],
            )
        )

    def _detect_icm_pressure(self, df: pd.DataFrame) -> None:
        bubble_df = df[df["_phase"].isin(["BUBBLE", "ITM", "FINAL_TABLE"])].copy()
        if bubble_df.empty:
            return

        for (stack_bucket, position, phase), group in bubble_df.groupby(["_stack_bucket", "_position", "_phase"], dropna=False):
            metric = self._group_metric(group)
            if not metric or metric["sample_size"] < self.MIN_SAMPLE_DEFAULT:
                continue

            stack_bb = float(pd.to_numeric(group.get("_stack_bb", 0), errors="coerce").fillna(0.0).median())
            bubble_factor = calculate_bubble_factor(stack_bb, phase)
            line_type = group["_line_type"].mode().iat[0] if "_line_type" in group.columns and not group["_line_type"].mode().empty else "UNKNOWN"
            context = {
                "stack_depth_bucket": group["_stack_depth_bucket"].mode().iat[0] if not group["_stack_depth_bucket"].mode().empty else "UNKNOWN",
                "stack_bucket": stack_bucket,
                "position": position,
                "phase": phase,
                "bubble_factor": round(float(bubble_factor), 3),
                "opponent_type": group["_opponent_type"].mode().iat[0] if not group["_opponent_type"].mode().empty else "unknown",
                "opponent_cluster": group["_opponent_cluster"].mode().iat[0] if "_opponent_cluster" in group.columns and not group["_opponent_cluster"].mode().empty else "Passive",
                "board_texture": group["_board_texture"].mode().iat[0] if not group["_board_texture"].mode().empty else "NO_BOARD",
                "line_type": line_type,
            }

            if metric["fold_rate"] > 0.70 and bubble_factor <= 1.40 and position in {"SB", "BB", "BTN", "CO"}:
                self._emit_group_leak(
                    group,
                    "ICM_PASSIVITY",
                    metric["fold_rate"],
                    0.55,
                    context,
                    0.78,
                    [f"fold_rate={metric['fold_rate']:.2f}", f"bubble_factor={bubble_factor:.2f}", "icm_passivity=true"],
                    self.MIN_SAMPLE_DEFAULT,
                )

            jam_spots = group[group["_line_type"].isin(["JAM", "3BET_POT"])].copy()
            if not jam_spots.empty:
                call_rate = float(jam_spots["_action_pf"].str.contains("CALL", regex=False, na=False).mean())
                if call_rate > 0.24 and bubble_factor >= 1.25:
                    self._emit_group_leak(
                        jam_spots,
                        "ICM_SUICIDE",
                        call_rate,
                        0.20,
                        context,
                        0.84,
                        [f"call_rate_jam={call_rate:.2f}", f"bubble_factor={bubble_factor:.2f}", "icm_suicide=true"],
                        self.MIN_SAMPLE_DEFAULT,
                    )

            if bubble_factor >= 1.30 and metric["raise_rate"] < 0.14 and position in {"CO", "BTN", "SB"} and phase in {"BUBBLE", "FINAL_TABLE", "ITM"}:
                self._emit_group_leak(
                    group,
                    "ICM_PASSIVITY",
                    metric["raise_rate"],
                    0.18,
                    context,
                    0.76,
                    [f"raise_rate={metric['raise_rate']:.2f}", f"bubble_factor={bubble_factor:.2f}", "icm_passivity=true"],
                    self.MIN_SAMPLE_DEFAULT,
                )

    def _detect_timing_tells(self, df: pd.DataFrame) -> None:
        time_col = None
        for c in ("decision_time_ms", "hero_decision_ms", "action_time_ms"):
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            return

        x = df.copy()
        x["_decision_ms"] = pd.to_numeric(x[time_col], errors="coerce").fillna(0.0)
        x = x[x["_decision_ms"] > 0]
        if x.empty or len(x) < self.MIN_SAMPLE_DEFAULT:
            return

        for (stack_bucket, position, phase, line_type), group in x.groupby(["_stack_bucket", "_position", "_phase", "_line_type"], dropna=False):
            if len(group) < self.MIN_SAMPLE_DEFAULT:
                continue
            t_med = float(group["_decision_ms"].median())
            t_q1 = float(group["_decision_ms"].quantile(0.25))
            t_q3 = float(group["_decision_ms"].quantile(0.75))

            context = {
                "stack_depth_bucket": group["_stack_depth_bucket"].mode().iat[0] if not group["_stack_depth_bucket"].mode().empty else "UNKNOWN",
                "stack_bucket": stack_bucket,
                "position": position,
                "phase": phase,
                "opponent_type": group["_opponent_type"].mode().iat[0] if not group["_opponent_type"].mode().empty else "unknown",
                "opponent_cluster": group["_opponent_cluster"].mode().iat[0] if "_opponent_cluster" in group.columns and not group["_opponent_cluster"].mode().empty else "Passive",
                "board_texture": group["_board_texture"].mode().iat[0] if not group["_board_texture"].mode().empty else "NO_BOARD",
                "line_type": line_type,
            }

            complex_spot = line_type in {"3BET_POT", "JAM", "AGGRESSIVE_POSTFLOP"}
            if complex_spot and t_med < max(1200.0, t_q1):
                self._emit_group_leak(
                    group,
                    "TIMING_IMPULSIVE",
                    t_med,
                    max(1800.0, t_q3),
                    context,
                    0.58,
                    [f"decision_ms_median={t_med:.0f}", f"complex_spot={line_type}"],
                    self.MIN_SAMPLE_DEFAULT,
                )

            bluff_spot = line_type in {"AGGRESSIVE_POSTFLOP", "MIXED"}
            river_agg = group["_action_river"].str.contains("BET|RAISE", regex=True, na=False)
            if bluff_spot and river_agg.mean() > 0.25 and t_med > max(8000.0, t_q3 * 1.5):
                self._emit_group_leak(
                    group,
                    "TIMING_TELEGRAPH",
                    t_med,
                    max(3500.0, t_q3),
                    context,
                    0.50,
                    [f"decision_ms_median={t_med:.0f}", "slow_bluff_pattern"],
                    self.MIN_SAMPLE_DEFAULT,
                )

    def _filter_by_stack_bucket(self, bucket: str) -> pd.DataFrame:
        if self.df.empty:
            return self.df
        bucket = str(bucket).upper()
        if bucket == "LOW":
            return self.df[self.df["hero_stack_start"] <= 20 * self.df["big_blind"]]
        if bucket == "MID":
            return self.df[(self.df["hero_stack_start"] > 20 * self.df["big_blind"]) & (self.df["hero_stack_start"] <= 50 * self.df["big_blind"])]
        if bucket == "HIGH":
            return self.df[self.df["hero_stack_start"] > 50 * self.df["big_blind"]]
        return self.df

    def to_dataframe(self) -> pd.DataFrame:
        if not self.detections:
            return pd.DataFrame()
        return pd.DataFrame([asdict(d) for d in self.detections])
