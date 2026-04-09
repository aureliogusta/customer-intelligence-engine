"""
leak_analysis/modules/study_planner.py
Gera um plano de estudo personalizado por contexto real e prioridade auditável.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import pandas as pd

from .analysis_utils import clamp

log = logging.getLogger(__name__)


@dataclass
class StudyRecommendation:
    priority: int
    leak_code: str
    severity_label: str
    recommendation_text: str
    study_focus: str
    estimated_hours: float
    resources: List[str]
    success_criteria: str
    expected_roi_gain: float
    sample_size: int
    confidence: float
    evidence: List[str]
    representative_hand_ids: List[Any]
    spot: str
    bayesian_bb100: float
    hand_references: List[str]
    evs_score: float


class StudyPlanner:
    """Gera plano de estudo priorizado e orientado ao EV perdido."""

    STUDY_RECOMMENDATIONS = {
        "VPIP_LOOSE": {
            "focus": "Apertar range pré-flop por posição e stack",
            "resources": ["Push/Fold charts", "RFI charts", "Range construction"],
            "criteria": "Reduzir VPIP mantendo EV em spots +EV",
        },
        "VPIP_TIGHT": {
            "focus": "Expandir ranges lucrativos por posição",
            "resources": ["RFI charts", "Blind defense charts"],
            "criteria": "Aumentar VPIP sem ampliar perdas em spots marginais",
        },
        "PFR_LOW": {
            "focus": "Aumentar agressão pré-flop com seleção correta",
            "resources": ["3-bet strategy articles", "RFI charts"],
            "criteria": "Subir PFR em spots de valor e fold equity",
        },
        "CALL_EXCESSIVE": {
            "focus": "Eliminar calls sem pot odds ou playability",
            "resources": ["Pot odds drills", "Range-vs-range exercises"],
            "criteria": "Reduzir calls negativos por posição",
        },
        "FOLD_EXCESSIVE": {
            "focus": "Defender corretamente blinds e late position",
            "resources": ["Blind defense charts", "ICM awareness"] ,
            "criteria": "Reduzir folds de excesso em spots com EV positivo",
        },
        "THREE_BET_WRONG": {
            "focus": "Separar 3-bet value de bluff por posição/stack",
            "resources": ["3-bet vs open charts", "Blocker logic"],
            "criteria": "Melhorar taxa de 3-bet lucrativa",
        },
        "FOUR_BET_WRONG": {
            "focus": "Selecionar 4-bets com stack depth coerente",
            "resources": ["4-bet trees", "Stack depth heuristics"],
            "criteria": "Evitar 4-bets fora de faixa ótima",
        },
        "C_BET_WEAK": {
            "focus": "C-bet com lógica de range e board texture",
            "resources": ["Board texture drills", "Range advantage review"],
            "criteria": "Aumentar c-bets lucrativas em boards favoráveis",
        },
        "OVERPLAY_MEDIUM": {
            "focus": "Desacelerar com mãos médias e respeitar texture",
            "resources": ["Hand strength by street", "Population tendencies"],
            "criteria": "Reduzir perdas em showdown e overcalls",
        },
        "BLUFF_BAD": {
            "focus": "Selecionar blefes com blockers e runout favorável",
            "resources": ["Bluff selection matrix", "Runout planning"],
            "criteria": "Melhorar EV de blefes em river/turn",
        },
        "VALUE_BET_LOST": {
            "focus": "Extrair mais value com sizing e linha corretos",
            "resources": ["Thin value spots", "Sizing calibration"],
            "criteria": "Aumentar captura de value em streets finais",
        },
        "SIZING_POOR": {
            "focus": "Padronizar sizing por street, board e stack",
            "resources": ["Sizing grids", "Pot geometry"],
            "criteria": "Diminuir leaks de sizing por rua",
        },
        "OVERBETTING": {
            "focus": "Controlar overbets para spots realmente +EV",
            "resources": ["Overbet theory", "Polarized ranges"],
            "criteria": "Reduzir overbets sem base de range",
        },
        "UNDERBETTING": {
            "focus": "Aumentar pressão quando o board pede sizing maior",
            "resources": ["Value extraction", "Protection bets"],
            "criteria": "Corrigir sizing pequeno em spots de valor",
        },
        "TILT_PATTERN": {
            "focus": "Gestão de sessão, pausa e stop-loss cognitivo",
            "resources": ["Mental game", "Session review"],
            "criteria": "Reduzir mudança comportamental pós-downswing",
        },
        "ICM_OVERFOLD": {
            "focus": "Pressão de bolha e defesa de stack médio",
            "resources": ["ICM charts", "Bubble factor drills", "Push/fold review"],
            "criteria": "Reduzir folds indevidos quando o call/defesa é +EV monetário",
        },
        "ICM_OVERCALL": {
            "focus": "Disciplina de call em spots de eliminação",
            "resources": ["ICM pressure spots", "Call vs shove training"],
            "criteria": "Evitar calls marginais que destroem $EV na bolha",
        },
        "ICM_SUICIDE": {
            "focus": "Evitar calls suicidas na bolha",
            "resources": ["ICM risk premium", "Bubble factor review"],
            "criteria": "Bloquear calls onde o risco de eliminação supera o ganho de fichas",
        },
        "ICM_PASSIVITY": {
            "focus": "Aplicar pressão quando a mesa está travada pela bolha",
            "resources": ["Open-push charts", "Pressure node review"],
            "criteria": "Aumentar open-shove e steals lucrativos na bolha",
        },
    }

    def __init__(self, severity_scores: List, leak_detections: Optional[List] = None):
        self.severity_scores = severity_scores or []
        self.leak_detections = leak_detections or []
        self.recommendations: List[StudyRecommendation] = []

    def plan_study(self) -> List[StudyRecommendation]:
        log.info("Gerando plano de estudo para %s leaks...", len(self.severity_scores))
        self.recommendations = []

        for score in self.severity_scores:
            if getattr(score, "sample_size", 0) < 12:
                continue
            if getattr(score, "confidence", 0.0) < 0.45:
                continue
            if score.severity_label not in {"CRÍTICO", "ALTO", "MÉDIO"}:
                continue
            self.recommendations.append(self._create_recommendation(score))

        self.recommendations.sort(key=lambda r: (r.evs_score, r.expected_roi_gain, r.confidence), reverse=True)
        for idx, rec in enumerate(self.recommendations, start=1):
            rec.priority = idx
        log.info("Plano gerado: %s itens de estudo", len(self.recommendations))
        return self.recommendations

    def _create_recommendation(self, score) -> StudyRecommendation:
        leak_code = score.leak_code
        study_info = self.STUDY_RECOMMENDATIONS.get(leak_code, {})
        rec_text = self._customize_recommendation(score, study_info)
        estimated_hours = self._estimate_hours(score, study_info)

        representative_ids = self._representative_hand_ids(leak_code)
        spot = f"{getattr(score, 'position', 'UNKNOWN')} vs field {getattr(score, 'stack_depth_bucket', getattr(score, 'stack_bucket', 'UNKNOWN'))}"
        hand_refs = [f"hand_id:{hid}" for hid in representative_ids]
        correction_weight = self._correction_weight(leak_code)
        ev_lost = float(max(0.0, getattr(score, "bayesian_ev_loss", 0.0)))
        recurrence = float(max(1.0, getattr(score, "sample_size", 0)))
        evs_score = (ev_lost * recurrence) * correction_weight
        evs_score *= max(0.5, float(getattr(score, "confidence", 0.0)))

        return StudyRecommendation(
            priority=score.priority_rank,
            leak_code=leak_code,
            severity_label=score.severity_label,
            recommendation_text=rec_text,
            study_focus=study_info.get("focus", f"Revisar {leak_code}"),
            estimated_hours=estimated_hours,
            resources=study_info.get("resources", ["Review solver output", "Session hand review"]),
            success_criteria=study_info.get("criteria", "Medir antes/depois em mãos comparáveis"),
            expected_roi_gain=score.potential_roi_gain,
            sample_size=getattr(score, "sample_size", 0),
            confidence=getattr(score, "confidence", 0.0),
            evidence=list(getattr(score, "evidence", [])),
            representative_hand_ids=representative_ids,
            spot=spot,
            bayesian_bb100=float(getattr(score, "bayesian_bb100", 0.0)),
            hand_references=hand_refs,
            evs_score=float(evs_score),
        )

    def _correction_weight(self, leak_code: str) -> float:
        easy = {"VPIP_LOOSE", "VPIP_TIGHT", "PFR_LOW", "CALL_EXCESSIVE", "FOLD_EXCESSIVE", "C_BET_WEAK", "C_BET_STRONG", "ICM_OVERFOLD", "ICM_OVERCALL", "ICM_SUICIDE", "ICM_PASSIVITY"}
        complex_spots = {"BLUFF_BAD", "OVERPLAY_MEDIUM", "VALUE_BET_LOST"}
        if leak_code in easy:
            return 1.30
        if leak_code in complex_spots:
            return 0.85
        return 1.0

    def _representative_hand_ids(self, leak_code: str, top_n: int = 5) -> List[Any]:
        if not self.leak_detections:
            return []

        ranked = []
        for det in self.leak_detections:
            if getattr(det, "leak_code", None) != leak_code:
                continue
            hand_id = getattr(det, "hand_id", None)
            if hand_id is None:
                continue
            sev = float(getattr(det, "error_severity", 0.0) or 0.0)
            conf = float(getattr(det, "confidence", 0.0) or 0.0)
            ev_delta = getattr(det, "ev_delta", 0.0)
            ev_delta_abs = abs(float(ev_delta)) if ev_delta is not None else 0.0
            ranked.append((sev, conf, ev_delta_abs, hand_id))

        ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        unique_ids = []
        for _, _, _, hand_id in ranked:
            if hand_id in unique_ids:
                continue
            unique_ids.append(hand_id)
            if len(unique_ids) >= top_n:
                break
        return unique_ids

    def _estimate_hours(self, score, study_info: Dict) -> float:
        base = float(study_info.get("hours", 2.0))
        severity_factor = {"CRÍTICO": 1.5, "ALTO": 1.2, "MÉDIO": 1.0, "BAIXO": 0.8}.get(score.severity_label, 1.0)
        sample_factor = 1.0 + min(0.6, getattr(score, "sample_size", 0) / 100.0)
        confidence_factor = 1.0 + max(0.0, 1.0 - getattr(score, "confidence", 0.0))
        return round(clamp(base * severity_factor * sample_factor * confidence_factor, 0.5, 12.0), 1)

    def _customize_recommendation(self, score, study_info: Dict) -> str:
        position = getattr(score, "position", "UNKNOWN")
        stack_bucket = getattr(score, "stack_depth_bucket", getattr(score, "stack_bucket", "UNKNOWN"))
        evidence = "; ".join(getattr(score, "evidence", [])[:3])
        if str(getattr(score, "leak_code", "")).startswith("ICM_"):
            default_focus = "ICM, bubble factor e sobrevivência de stack"
        else:
            default_focus = "Revisão técnica"
        text = [
            f"PRIORIDADE {score.priority_rank}: {score.severity_label}",
            f"Leak: {score.leak_code}",
            f"Contexto: {position} / {stack_bucket}",
            f"Foco: {study_info.get('focus', default_focus)}",
            f"EV perdido estimado: {score.potential_roi_gain:.2f} bb",
            f"Perda Bayesiana: {getattr(score, 'bayesian_bb100', 0.0):.2f} bb/100",
            f"Amostra: {getattr(score, 'sample_size', 0)} | Confiança: {getattr(score, 'confidence', 0.0):.2f}",
        ]
        if evidence:
            text.append(f"Evidência: {evidence}")
        return "\n".join(text)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.recommendations:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "priority": r.priority,
                "leak_code": r.leak_code,
                "severity": r.severity_label,
                "focus": r.study_focus,
                "hours": r.estimated_hours,
                "expected_gain": r.expected_roi_gain,
                "spot": r.spot,
                "bayesian_bb100": r.bayesian_bb100,
                "sample_size": r.sample_size,
                "confidence": r.confidence,
                "evs_score": r.evs_score,
                "evidence": " | ".join(r.evidence),
                "representative_hand_ids": r.representative_hand_ids,
                "hand_references": r.hand_references,
                "recommendation": r.recommendation_text,
            }
            for r in self.recommendations
        ])

    def to_json(self) -> List[Dict]:
        return [
            {
                "priority": r.priority,
                "leak_code": r.leak_code,
                "severity": r.severity_label,
                "recommendation": r.recommendation_text,
                "study_focus": r.study_focus,
                "estimated_hours": r.estimated_hours,
                "resources": r.resources,
                "success_criteria": r.success_criteria,
                "expected_roi_gain": r.expected_roi_gain,
                "spot": r.spot,
                "bayesian_bb100": r.bayesian_bb100,
                "sample_size": r.sample_size,
                "confidence": r.confidence,
                "evs_score": r.evs_score,
                "evidence": r.evidence,
                "representative_hand_ids": r.representative_hand_ids,
                "hand_references": r.hand_references,
            }
            for r in self.recommendations
        ]
