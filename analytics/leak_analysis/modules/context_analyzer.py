"""
leak_analysis/modules/context_analyzer.py
Agrega o contexto real de cada leak sem buckets artificiais ou duplicação.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .analysis_utils import (
    classify_board_texture,
    classify_stack_bucket,
    classify_tournament_phase,
    infer_opponent_type,
    normalize_position,
    safe_div,
    safe_float,
)

log = logging.getLogger(__name__)


@dataclass
class LeakContext:
    leak_code: str
    hero_name: str
    network: str
    position: str
    stack_bb: float
    stack_bucket: str
    stack_depth_bucket: str
    opponent_type: str
    opponent_cluster: str
    tournament_phase: str
    board_texture: str
    street: str
    line_type: str
    frequency: int
    total_ev_lost: float
    average_loss_per_instance: float
    sample_size: int
    confidence: float
    ev_expected: Optional[float] = None
    ev_real: Optional[float] = None
    ev_delta: Optional[float] = None
    vop_deviation: float = 0.0
    cascade_rate: float = 0.0
    confidence_reason: str = ""
    evidence: List[str] = field(default_factory=list)


class ContextAnalyzer:
    """Agrega o contexto de leaks por chave auditável."""

    def __init__(self, df_hands: pd.DataFrame):
        self.df = df_hands.copy() if df_hands is not None else pd.DataFrame()
        self.contexts: List[LeakContext] = []

    def analyze(self, leak_detections: List) -> List[LeakContext]:
        log.info("Analisando contexto de %s leaks...", len(leak_detections))
        self.contexts = []
        for detection in leak_detections:
            ctx = self._build_context(detection)
            if ctx is not None:
                self.contexts.append(ctx)
        return self.contexts

    def _build_context(self, detection) -> Optional[LeakContext]:
        if detection is None:
            return None

        context = dict(getattr(detection, "context", {}) or {})
        position = normalize_position(context.get("position") or context.get("hero_position") or "UNKNOWN")
        stack_depth_bucket = str(context.get("stack_depth_bucket") or context.get("stack_bucket") or "UNKNOWN").upper()
        stack_bucket = str(context.get("stack_bucket") or stack_depth_bucket).upper()
        opponent_type = str(context.get("opponent_type") or "unknown").strip().lower()
        opponent_cluster = str(context.get("opponent_cluster") or "Passive").strip().title()
        tournament_phase = str(context.get("phase") or context.get("tournament_phase") or "UNKNOWN").upper()
        board_texture = str(context.get("board_texture") or "NO_BOARD").upper()
        line_type = str(context.get("line_type") or "UNKNOWN").upper()

        sample_size = int(getattr(detection, "sample_size", context.get("sample_size", 0)) or 0)
        total_ev_lost = safe_float(context.get("total_ev_lost", getattr(detection, "ev_real", 0.0)), 0.0)
        ev_real = safe_float(getattr(detection, "ev_real", None), None)
        ev_expected = safe_float(getattr(detection, "ev_expected", None), None)
        ev_delta = safe_float(getattr(detection, "ev_delta", None), None)
        confidence = safe_float(getattr(detection, "confidence", context.get("confidence", 0.0)), 0.0)
        desvio = safe_float(getattr(detection, "desvio_gto_pct", 0.0), 0.0)

        total_ev_lost = safe_float(total_ev_lost, 0.0)

        stack_bb = safe_float(context.get("stack_bb", context.get("stack_bb_raw", 0.0)), 0.0)
        if stack_bb <= 0 and self.df is not None and not self.df.empty and "hero_stack_start" in self.df.columns and "big_blind" in self.df.columns:
            hand_row = self._lookup_hand_row(detection.hand_id)
            if hand_row is not None:
                stack_bb = safe_div(safe_float(hand_row.get("hero_stack_start", 0.0), 0.0), max(safe_float(hand_row.get("big_blind", 1.0), 1.0), 1.0), 0.0)
        else:
            hand_row = self._lookup_hand_row(detection.hand_id)

        stack_bucket_calc, stack_depth_bucket_calc = classify_stack_bucket(stack_bb)

        if not tournament_phase or tournament_phase == "UNKNOWN":
            hand_row = self._lookup_hand_row(detection.hand_id)
            if hand_row is not None:
                tournament_phase = classify_tournament_phase(hand_row)
                if opponent_type == "unknown":
                    opponent_type = infer_opponent_type(hand_row)
                if board_texture == "NO_BOARD":
                    board_texture = classify_board_texture(hand_row.get("board_flop"))
                if line_type == "UNKNOWN":
                    line_type = self._infer_line_type_from_row(hand_row)

        if stack_bucket == "UNKNOWN":
            stack_bucket = stack_bucket_calc
        if stack_depth_bucket == "UNKNOWN":
            stack_depth_bucket = stack_depth_bucket_calc

        hero_name = "UNKNOWN"
        network = "UNKNOWN"
        if hand_row is not None:
            hero_name = str(hand_row.get("hero_name", "UNKNOWN") or "UNKNOWN")
            hero_up = hero_name.upper()
            if hero_up == "AURELIODIZZY":
                network = "WPN"
            elif hero_up == "PIODIZZY":
                network = "POKERSTARS"

        frequency = sample_size if sample_size > 0 else self._context_frequency(detection.leak_code, position, stack_depth_bucket)
        avg_loss = safe_div(total_ev_lost, frequency, 0.0) if frequency else 0.0

        street = str(context.get("street") or ("PREFLOP" if "PREFLOP" in detection.category.upper() else "POSTFLOP")).upper()

        return LeakContext(
            leak_code=detection.leak_code,
            hero_name=hero_name,
            network=network,
            position=position,
            stack_bb=stack_bb,
            stack_bucket=stack_bucket,
            stack_depth_bucket=stack_depth_bucket,
            opponent_type=opponent_type,
            opponent_cluster=opponent_cluster,
            tournament_phase=tournament_phase,
            board_texture=board_texture,
            street=street,
            line_type=line_type,
            frequency=frequency,
            total_ev_lost=total_ev_lost,
            average_loss_per_instance=avg_loss,
            sample_size=sample_size,
            confidence=confidence,
            ev_expected=ev_expected,
            ev_real=ev_real,
            ev_delta=ev_delta,
            vop_deviation=desvio / 100.0,
            cascade_rate=safe_float(context.get("cascade_rate", 0.0), 0.0),
            confidence_reason=str(getattr(detection, "confidence_reason", "")),
            evidence=list(getattr(detection, "evidence", [])),
        )

    def _lookup_hand_row(self, hand_id: Any) -> Optional[pd.Series]:
        if self.df is None or self.df.empty or hand_id is None:
            return None
        if "id" in self.df.columns:
            hand_row = self.df[self.df["id"] == hand_id]
            if not hand_row.empty:
                return hand_row.iloc[0]
        if "hand_id" in self.df.columns:
            hand_row = self.df[self.df["hand_id"] == hand_id]
            if not hand_row.empty:
                return hand_row.iloc[0]
        return None

    # Compatibilidade com testes antigos e com chamadas externas legadas.
    def _classify_stack_bucket(self, stack_bb: float) -> str:
        return classify_stack_bucket(stack_bb)[1]

    def _classify_tournament_phase(self, m_ratio: float) -> str:
        return classify_tournament_phase(pd.Series({"m_ratio": m_ratio}))

    def _classify_board_texture(self, board: Optional[str]) -> str:
        return classify_board_texture(board)

    def _infer_line_type_from_row(self, row: pd.Series) -> str:
        actions = [str(row.get(col, "") or "").upper() for col in ("hero_action_preflop", "hero_action_flop", "hero_action_turn", "hero_action_river")]
        if not any(actions):
            return "UNKNOWN"
        if any("ALL-IN" in a for a in actions):
            return "JAM"
        if "3BET" in actions[0] or "4BET" in actions[0]:
            return "3BET_POT"
        if any("BET" in a for a in actions[1:]):
            return "AGGRESSIVE_POSTFLOP"
        if any("CALL" in a for a in actions):
            return "CALL_LINE"
        if all("CHECK" in a or "FOLD" in a or not a for a in actions):
            return "PASSIVE"
        return "MIXED"

    def _context_frequency(self, leak_code: str, position: str, stack_depth_bucket: str) -> int:
        if self.df is None or self.df.empty:
            return 0
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        if "hero_position" in self.df.columns:
            mask &= self.df["hero_position"].astype(str).str.upper().eq(position)
        return int(mask.sum())

    def aggregate_by_context(self) -> pd.DataFrame:
        if not self.contexts:
            return pd.DataFrame()

        rows = []
        for ctx in self.contexts:
            rows.append({
                "leak_code": ctx.leak_code,
                "hero_name": ctx.hero_name,
                "network": ctx.network,
                "position": ctx.position,
                "stack_bb": ctx.stack_bb,
                "stack_bucket": ctx.stack_bucket,
                "stack_depth_bucket": ctx.stack_depth_bucket,
                "opponent_type": ctx.opponent_type,
                "opponent_cluster": ctx.opponent_cluster,
                "phase": ctx.tournament_phase,
                "board_texture": ctx.board_texture,
                "street": ctx.street,
                "line_type": ctx.line_type,
                "frequency": ctx.frequency,
                "sample_size": ctx.sample_size,
                "confidence": ctx.confidence,
                "total_ev_lost": ctx.total_ev_lost,
                "avg_loss_per_hand": ctx.average_loss_per_instance,
                "ev_expected": ctx.ev_expected,
                "ev_real": ctx.ev_real,
                "ev_delta": ctx.ev_delta,
                "vop_deviation": ctx.vop_deviation,
                "cascade_rate": ctx.cascade_rate,
                "confidence_reason": ctx.confidence_reason,
                "evidence": " | ".join(ctx.evidence),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Normaliza colunas numéricas para evitar NaN/strings em agregação.
        for col in ["frequency", "sample_size", "confidence", "total_ev_lost", "ev_expected", "ev_real", "ev_delta"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["frequency"] = df["frequency"].fillna(0.0)
        df["sample_size"] = df["sample_size"].fillna(0.0)
        df["confidence"] = df["confidence"].fillna(0.0)
        df["total_ev_lost"] = df["total_ev_lost"].fillna(0.0)

        group_cols = [
            "leak_code",
            "hero_name",
            "network",
            "position",
            "stack_bucket",
            "stack_depth_bucket",
            "opponent_type",
            "opponent_cluster",
            "phase",
            "board_texture",
            "street",
            "line_type",
        ]
        def _aggregate_group(g: pd.DataFrame) -> pd.Series:
            freq = float(pd.to_numeric(g["frequency"], errors="coerce").fillna(0.0).sum())
            sample = float(pd.to_numeric(g["sample_size"], errors="coerce").fillna(0.0).sum())
            total_ev_lost = float(pd.to_numeric(g["total_ev_lost"], errors="coerce").fillna(0.0).sum())

            conf_num = float((pd.to_numeric(g["confidence"], errors="coerce").fillna(0.0) * pd.to_numeric(g["sample_size"], errors="coerce").fillna(0.0)).sum())
            confidence = safe_div(conf_num, sample, 0.0)

            avg_loss_per_hand = safe_div(total_ev_lost, freq, 0.0)

            sample_weights = pd.to_numeric(g["sample_size"], errors="coerce").fillna(0.0)

            def _weighted_mean(col: str) -> Optional[float]:
                if col not in g.columns:
                    return None
                values = pd.to_numeric(g[col], errors="coerce")
                valid = values.notna() & sample_weights.gt(0)
                if not valid.any():
                    return None
                num = float((values[valid] * sample_weights[valid]).sum())
                den = float(sample_weights[valid].sum())
                return safe_div(num, den, None)

            ev_expected = _weighted_mean("ev_expected")
            ev_real = _weighted_mean("ev_real")
            ev_delta = _weighted_mean("ev_delta")
            vop_deviation = _weighted_mean("vop_deviation") or 0.0
            cascade_rate = _weighted_mean("cascade_rate") or 0.0

            confidence_reason = "; ".join(sorted(set(map(str, g["confidence_reason"].dropna().astype(str)))))
            evidence = " || ".join([v for v in g["evidence"].dropna().astype(str) if v])

            return pd.Series(
                {
                    "frequency": int(round(freq)),
                    "sample_size": int(round(sample)),
                    "confidence": confidence,
                    "total_ev_lost": total_ev_lost,
                    "avg_loss_per_hand": avg_loss_per_hand,
                    "ev_expected": ev_expected,
                    "ev_real": ev_real,
                    "ev_delta": ev_delta,
                    "vop_deviation": vop_deviation,
                    "cascade_rate": cascade_rate,
                    "confidence_reason": confidence_reason,
                    "evidence": evidence,
                }
            )

        grouped = df.groupby(group_cols, dropna=False).apply(_aggregate_group).reset_index()
        grouped = grouped.sort_values(["total_ev_lost", "frequency"], ascending=False).reset_index(drop=True)
        return grouped

