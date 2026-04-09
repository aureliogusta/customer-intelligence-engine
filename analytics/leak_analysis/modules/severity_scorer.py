"""
leak_analysis/modules/severity_scorer.py
Score auditável de severidade com penalização por amostra e confiança.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from .analysis_utils import clamp, context_distance_penalty, safe_float, stack_bucket_multiplier

log = logging.getLogger(__name__)


@dataclass
class SeverityScore:
    leak_code: str
    position: str
    stack_bucket: str
    stack_depth_bucket: str
    opponent_type: str
    phase: str
    board_texture: str
    frequency_score: float
    impact_score: float
    recurrence_score: float
    context_importance: float
    repair_difficulty: float
    distance_from_optimal: float
    cascade_score: float
    severity_score: float
    severity_label: str
    priority_rank: int
    potential_roi_gain: float
    sample_size: int
    confidence: float
    bayesian_ev_loss: float
    bayesian_bb100: float
    why_i_lost: str
    shap_like_values: Dict[str, float]
    justification: str = ""
    evidence: List[str] = field(default_factory=list)


class SeverityScorer:
    """Calcula severidade com base em impacto, recorrência, contexto e confiança."""

    W_FREQUENCY = 0.16
    W_IMPACT = 0.28
    W_RECURRENCE = 0.16
    W_CONTEXT = 0.16
    W_DIFFICULTY = 0.12
    W_DISTANCE = 0.08
    W_CASCADE = 0.04
    K_SKEPTICISM = 70.0

    SHARKSCOPE_CRISIS = {
        "AURELIODIZZY": {
            "network": "WPN",
            "roi": -88.9,
            "itm": 6.7,
            "early_finish_rate": 1.0,
            "small_ball_focus": True,
        },
        "PIODIZZY": {
            "network": "POKERSTARS",
            "roi": -91.9,
            "itm": 14.3,
            "early_finish_rate": 0.0,
            "small_ball_focus": True,
        },
    }

    REPAIR_DIFFICULTY = {
        "VPIP_LOOSE": 0.35,
        "VPIP_TIGHT": 0.40,
        "PFR_LOW": 0.40,
        "CALL_EXCESSIVE": 0.50,
        "FOLD_EXCESSIVE": 0.50,
        "THREE_BET_WRONG": 0.55,
        "FOUR_BET_WRONG": 0.60,
        "FOLD_VS_3BET_HIGH": 0.55,
        "C_BET_WEAK": 0.60,
        "C_BET_STRONG": 0.55,
        "OVERPLAY_MEDIUM": 0.70,
        "BLUFF_BAD": 0.75,
        "VALUE_BET_LOST": 0.68,
        "SIZING_POOR": 0.55,
        "OVERBETTING": 0.60,
        "UNDERBETTING": 0.55,
        "ICM_OVERFOLD": 0.78,
        "ICM_OVERCALL": 0.80,
        "ICM_SUICIDE": 0.90,
        "ICM_PASSIVITY": 0.76,
        "TILT_PATTERN": 0.90,
    }

    POSITION_MULTIPLIER = {
        "UTG": 0.95,
        "LJ": 0.98,
        "HJ": 1.00,
        "CO": 1.08,
        "BTN": 1.12,
        "SB": 1.10,
        "BB": 1.15,
    }

    PHASE_MULTIPLIER = {
        "EARLY": 0.92,
        "MIDDLE": 1.00,
        "BUBBLE": 1.08,
        "ITM": 1.10,
        "FINAL_TABLE": 1.18,
    }

    def __init__(self, df_contexts: pd.DataFrame):
        self.df = df_contexts.copy() if df_contexts is not None else pd.DataFrame()
        self.scores: List[SeverityScore] = []
        self._impact_scale = self._compute_impact_scale()

    def _compute_impact_scale(self) -> float:
        if self.df.empty or "total_ev_lost" not in self.df.columns:
            return 1.0
        series = pd.to_numeric(self.df["total_ev_lost"], errors="coerce").fillna(0.0)
        q90 = float(series.quantile(0.90)) if len(series) else 1.0
        return max(q90, 1.0)

    def _crisis_profile(self, hero_name: str, network: str) -> dict[str, float] | None:
        hero = str(hero_name or "").upper()
        profile = self.SHARKSCOPE_CRISIS.get(hero)
        if not profile:
            return None
        net = str(network or "").upper()
        if net and net != "UNKNOWN" and str(profile.get("network", "")).upper() not in {"", net}:
            return None
        return profile

    def _small_ball_risk(self, leak_code: str, line_type: str, position: str, stack_depth_bucket: str) -> float:
        risk = 0.0
        if leak_code in {"PFR_LOW", "VPIP_TIGHT", "FOLD_EXCESSIVE", "C_BET_WEAK", "CALL_EXCESSIVE"}:
            risk += 0.35
        if str(line_type).upper() in {"SINGLE_RAISED_POT", "UNKNOWN", "CALL_LINE"}:
            risk += 0.10
        if str(position).upper() in {"BTN", "CO", "SB", "BB"}:
            risk += 0.10
        if str(stack_depth_bucket).upper() in {"15-20BB", "20-25BB", "25-35BB"}:
            risk += 0.15
        return clamp(risk, 0.0, 1.0)

    def _baseline_ev_loss(self, leak_code: str) -> float:
        if self.df.empty:
            return 0.0
        scoped = self.df[self.df.get("leak_code", "") == leak_code] if "leak_code" in self.df.columns else pd.DataFrame()
        source = scoped if not scoped.empty else self.df
        col = pd.to_numeric(source.get("avg_loss_per_hand", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        if col.empty:
            return 0.0
        return float(max(0.0, col.median()))

    def _bayesian_shrinkage(self, observed_ev_loss: float, sample_size: int, baseline_loss: float) -> float:
        n = max(0.0, float(sample_size))
        k = float(self.K_SKEPTICISM)
        return (n / (n + k)) * observed_ev_loss + (k / (n + k)) * baseline_loss

    def score_all(self) -> List[SeverityScore]:
        log.info("Calculando severidade de %s contextos...", len(self.df))
        self.scores = []
        if self.df.empty:
            return []

        for _, row in self.df.iterrows():
            self.scores.append(self._calculate_score(row))

        self.scores.sort(key=lambda s: (s.severity_score, s.confidence, s.sample_size), reverse=True)
        for rank, score in enumerate(self.scores, 1):
            score.priority_rank = rank

        if self.scores:
            log.info("Top leak: %s (score %.1f)", self.scores[0].leak_code, self.scores[0].severity_score)
        return self.scores

    def _calculate_score(self, row: pd.Series) -> SeverityScore:
        leak_code = str(row.get("leak_code", "UNKNOWN"))
        position = str(row.get("position", "UNKNOWN")).upper()
        stack_bucket = str(row.get("stack_bucket", "UNKNOWN")).upper()
        stack_depth_bucket = str(row.get("stack_depth_bucket", stack_bucket)).upper()
        opponent_type = str(row.get("opponent_type", "unknown")).lower()
        phase = str(row.get("phase", "UNKNOWN")).upper()
        board_texture = str(row.get("board_texture", "NO_BOARD")).upper()
        line_type = str(row.get("line_type", "UNKNOWN")).upper()
        hero_name = str(row.get("hero_name", "UNKNOWN") or "UNKNOWN")
        network = str(row.get("network", "UNKNOWN") or "UNKNOWN").upper()

        frequency = max(0.0, safe_float(row.get("frequency", 0), 0.0))
        sample_size = int(max(0, safe_float(row.get("sample_size", frequency), 0.0)))
        total_loss = max(0.0, safe_float(row.get("total_ev_lost", 0), 0.0))
        avg_loss = max(0.0, safe_float(row.get("avg_loss_per_hand", 0), 0.0))
        confidence = clamp(safe_float(row.get("confidence", 0.0), 0.0), 0.0, 1.0)
        ev_expected = row.get("ev_expected", None)
        ev_delta = safe_float(row.get("ev_delta", 0.0), 0.0)
        vop_deviation = clamp(safe_float(row.get("vop_deviation", 0.0), 0.0), 0.0, 2.0)
        cascade_rate = clamp(safe_float(row.get("cascade_rate", 0.0), 0.0), 0.0, 1.0)
        evidence = str(row.get("evidence", "")).split("||") if row.get("evidence") else []

        baseline_loss = self._baseline_ev_loss(leak_code)
        bayesian_ev_loss = self._bayesian_shrinkage(avg_loss, sample_size, baseline_loss)
        bayesian_bb100 = bayesian_ev_loss * 100.0

        frequency_score = clamp((math.log1p(frequency) / math.log1p(60.0)) * 100.0, 0.0, 100.0)
        # Usa perda suavizada para reduzir falso positivo em amostra pequena.
        impact_score = clamp((bayesian_ev_loss / self._impact_scale) * 100.0, 0.0, 100.0)

        unique_context_count = len({
            str(row.get("position", "UNKNOWN")),
            str(row.get("stack_bucket", "UNKNOWN")),
            str(row.get("stack_depth_bucket", stack_bucket)),
            str(row.get("opponent_type", "unknown")),
            str(row.get("phase", "UNKNOWN")),
            str(row.get("board_texture", "NO_BOARD")),
            str(row.get("line_type", "UNKNOWN")),
        })
        recurrence_score = clamp((unique_context_count / 7.0) * 100.0, 0.0, 100.0)

        context_multiplier = stack_bucket_multiplier(stack_depth_bucket) * self.POSITION_MULTIPLIER.get(position, 1.0) * self.PHASE_MULTIPLIER.get(phase, 1.0)
        if phase == "FINAL_TABLE" or stack_depth_bucket in {"20-25BB", "15-20BB"}:
            context_multiplier *= 2.0
        context_importance = clamp((context_multiplier - 0.8) * 100.0, 0.0, 100.0)

        repair_difficulty = self.REPAIR_DIFFICULTY.get(leak_code, 0.60)
        correction_ease = clamp(1.0 - repair_difficulty, 0.0, 1.0)
        if leak_code in {"VPIP_LOOSE", "VPIP_TIGHT", "PFR_LOW", "CALL_EXCESSIVE", "FOLD_EXCESSIVE", "C_BET_WEAK", "C_BET_STRONG", "ICM_OVERFOLD", "ICM_OVERCALL", "ICM_SUICIDE", "ICM_PASSIVITY"}:
            correction_ease = clamp(correction_ease * 1.2, 0.0, 1.0)
        distance_from_optimal = 0.0
        if ev_expected is not None and safe_float(ev_expected, None) is not None:
            distance_from_optimal = clamp(abs(ev_delta) / max(abs(total_loss), 1.0), 0.0, 1.0)
        elif row.get("desvio_gto_pct") is not None:
            distance_from_optimal = clamp(safe_float(row.get("desvio_gto_pct"), 0.0) / 100.0, 0.0, 1.0)
        else:
            distance_from_optimal = clamp((abs(bayesian_ev_loss) / max(self._impact_scale, 1.0)), 0.0, 1.0)

        cascade_score = clamp((((context_multiplier - 1.0) * 30.0) + correction_ease * 40.0 + cascade_rate * 30.0), 0.0, 100.0)

        recurrence_score = clamp(recurrence_score * (1.0 + min(0.30, vop_deviation / 2.0)), 0.0, 100.0)
        distance_from_optimal = clamp(max(distance_from_optimal, min(1.0, vop_deviation)), 0.0, 1.0)

        sample_penalty = min(1.0, sample_size / 40.0) if sample_size > 0 else 0.0
        confidence_penalty = confidence if confidence > 0 else 0.0

        raw_score = (
            frequency_score * self.W_FREQUENCY +
            impact_score * self.W_IMPACT +
            recurrence_score * self.W_RECURRENCE +
            context_importance * self.W_CONTEXT +
            correction_ease * 100.0 * self.W_DIFFICULTY +
            distance_from_optimal * 100.0 * self.W_DISTANCE +
            cascade_score * self.W_CASCADE
        )

        severity_score = raw_score * (0.55 + 0.45 * sample_penalty) * (0.60 + 0.40 * confidence_penalty)
        severity_score = clamp(severity_score, 0.0, 100.0)

        crisis_profile = self._crisis_profile(hero_name, network)
        crisis_note = ""
        if crisis_profile:
            roi = float(crisis_profile.get("roi", 0.0))
            itm = float(crisis_profile.get("itm", 0.0))
            early_finish_rate = float(crisis_profile.get("early_finish_rate", 0.0))
            crisis_boost = 1.0
            if roi <= -80.0:
                crisis_boost += 0.12
            if itm < 10.0:
                crisis_boost += 0.10
            if early_finish_rate >= 0.95:
                crisis_boost += 0.08
            small_ball = self._small_ball_risk(leak_code, line_type, position, stack_depth_bucket)
            crisis_boost += 0.18 * small_ball
            if stack_depth_bucket in {"15-20BB", "20-25BB"}:
                crisis_boost += 0.14
            if leak_code.startswith("ICM_"):
                crisis_boost += 0.10
            severity_score = clamp(severity_score * crisis_boost, 0.0, 100.0)
            crisis_note = f"sharkscope_roi={roi:.1f}; itm={itm:.1f}; early_finish={early_finish_rate:.2f}; small_ball={small_ball:.2f}"

        if sample_size < 12 or confidence < 0.45:
            severity_score = min(severity_score, 59.0)

        if severity_score >= 80:
            severity_label = "CRÍTICO"
        elif severity_score >= 60:
            severity_label = "ALTO"
        elif severity_score >= 40:
            severity_label = "MÉDIO"
        else:
            severity_label = "BAIXO"

        if sample_size < 12:
            severity_label = "BAIXO"
        elif sample_size < 20 and severity_label == "CRÍTICO":
            severity_label = "ALTO"

        potential_roi_gain = bayesian_ev_loss * frequency if severity_label in {"CRÍTICO", "ALTO"} else bayesian_ev_loss * frequency * 0.5

        components = {
            "impact_ev": impact_score * self.W_IMPACT,
            "recurrence_volume": recurrence_score * self.W_RECURRENCE,
            "frequency": frequency_score * self.W_FREQUENCY,
            "context_stack_phase": context_importance * self.W_CONTEXT,
            "ease_of_fix": correction_ease * 100.0 * self.W_DIFFICULTY,
            "distance_from_optimal": distance_from_optimal * 100.0 * self.W_DISTANCE,
            "cascade_effect": cascade_score * self.W_CASCADE,
            "vop_deviation": min(100.0, vop_deviation * 100.0) * 0.05,
        }
        top_components = sorted(components.items(), key=lambda kv: kv[1], reverse=True)[:3]
        why_i_lost = ", ".join(f"{name}={value:.1f}" for name, value in top_components)
        justification = (
            f"freq={frequency_score:.1f}; impact={impact_score:.1f}; recurrence={recurrence_score:.1f}; "
            f"context={context_importance:.1f}; repair={repair_difficulty:.2f}; ease={correction_ease:.2f}; distance={distance_from_optimal:.2f}; "
            f"cascade={cascade_score:.1f}; vop={vop_deviation:.2f}; line={line_type}; sample={sample_size}; confidence={confidence:.2f}; "
            f"ev_obs={avg_loss:.3f}; ev_base={baseline_loss:.3f}; ev_bayes={bayesian_ev_loss:.3f}; K={self.K_SKEPTICISM:.0f}; "
            f"hero={hero_name}; network={network}; {crisis_note}"
        )

        return SeverityScore(
            leak_code=leak_code,
            position=position,
            stack_bucket=stack_bucket,
            stack_depth_bucket=stack_depth_bucket,
            opponent_type=opponent_type,
            phase=phase,
            board_texture=board_texture,
            frequency_score=frequency_score,
            impact_score=impact_score,
            recurrence_score=recurrence_score,
            context_importance=context_importance,
            repair_difficulty=repair_difficulty,
            distance_from_optimal=distance_from_optimal,
            cascade_score=cascade_score,
            severity_score=severity_score,
            severity_label=severity_label,
            priority_rank=0,
            potential_roi_gain=potential_roi_gain,
            sample_size=sample_size,
            confidence=confidence,
            bayesian_ev_loss=bayesian_ev_loss,
            bayesian_bb100=bayesian_bb100,
            why_i_lost=why_i_lost,
            shap_like_values={k: round(v, 4) for k, v in components.items()},
            justification=justification,
            evidence=[e.strip() for e in evidence if e.strip()],
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self.scores:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "leak_code": s.leak_code,
                "position": s.position,
                "stack_bucket": s.stack_bucket,
                "stack_depth_bucket": s.stack_depth_bucket,
                "opponent_type": s.opponent_type,
                "phase": s.phase,
                "board_texture": s.board_texture,
                "frequency_score": s.frequency_score,
                "impact_score": s.impact_score,
                "recurrence_score": s.recurrence_score,
                "context_importance": s.context_importance,
                "repair_difficulty": s.repair_difficulty,
                "distance_from_optimal": s.distance_from_optimal,
                "cascade_score": s.cascade_score,
                "severity_score": s.severity_score,
                "severity_label": s.severity_label,
                "priority_rank": s.priority_rank,
                "potential_roi_gain": s.potential_roi_gain,
                "sample_size": s.sample_size,
                "confidence": s.confidence,
                "bayesian_ev_loss": s.bayesian_ev_loss,
                "bayesian_bb100": s.bayesian_bb100,
                "why_i_lost": s.why_i_lost,
                "shap_like_values": s.shap_like_values,
                "justification": s.justification,
                "evidence": " | ".join(s.evidence),
            }
            for s in self.scores
        ])
