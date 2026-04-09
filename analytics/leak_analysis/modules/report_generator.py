"""
leak_analysis/modules/report_generator.py
Gera relatórios auditáveis com contexto, EV e evidência.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd

from .analysis_utils import safe_float

log = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, leak_detections: List, contexts: pd.DataFrame, severity_scores: List, study_recommendations: List, df_hands: pd.DataFrame):
        self.leak_detections = leak_detections or []
        self.contexts = contexts if contexts is not None else pd.DataFrame()
        self.severity_scores = severity_scores or []
        self.study_recommendations = study_recommendations or []
        self.df_hands = df_hands if df_hands is not None else pd.DataFrame()

    def _ensure_context_data(self) -> None:
        if self.contexts is None or self.contexts.empty:
            raise ValueError("Relatório não pode ser gerado sem contextos agregados.")

    def _warnings(self) -> List[str]:
        warnings = []
        if self.df_hands.empty:
            warnings.append("Nenhuma mão disponível para análise.")
        elif len(self.df_hands) < 20:
            warnings.append("Amostra pequena: conclusões devem ser tratadas como indicativas.")
        if self.contexts.empty:
            warnings.append("Sem contexto agregado suficiente para comparar leaks por bucket.")
        if not self.severity_scores:
            warnings.append("Sem scores de severidade gerados.")
        return warnings

    def generate_executive_summary(self) -> Dict:
        total_hands = int(len(self.df_hands))
        total_ev_lost = 0.0
        if not self.df_hands.empty and "hero_amount_won" in self.df_hands.columns:
            total_ev_lost = float(max(0.0, -pd.to_numeric(self.df_hands["hero_amount_won"], errors="coerce").fillna(0.0).sum()))
        win_rate = float((self.df_hands.get("hero_result", pd.Series(dtype=str)) == "win").mean()) if not self.df_hands.empty and "hero_result" in self.df_hands.columns else 0.0
        top_leaks = self.severity_scores[:3] if self.severity_scores else []

        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_hands_analyzed": total_hands,
            "total_ev_lost": total_ev_lost,
            "win_rate": win_rate,
            "leaks_detected": len(self.leak_detections),
            "analysis_warnings": self._warnings(),
            "top_3_leaks": [
                {
                    "rank": i + 1,
                    "leak": s.leak_code,
                    "position": s.position,
                    "stack_bucket": s.stack_depth_bucket,
                    "severity": s.severity_label,
                    "score": float(s.severity_score),
                    "potential_gain": float(s.potential_roi_gain),
                    "sample_size": int(getattr(s, "sample_size", 0)),
                    "confidence": float(getattr(s, "confidence", 0.0)),
                    "justification": getattr(s, "justification", ""),
                }
                for i, s in enumerate(top_leaks)
            ],
            "total_potential_gain": float(sum(s.potential_roi_gain for s in self.severity_scores[:10])) if self.severity_scores else 0.0,
            "recommended_study_hours": float(sum(r.estimated_hours for r in self.study_recommendations[:5])) if self.study_recommendations else 0.0,
            "key_message": self._generate_key_message(top_leaks),
        }
        return summary

    def _generate_key_message(self, top_leaks: List) -> str:
        if not top_leaks:
            return "Sem leaks confirmados com amostra e contexto suficientes."

        primary = top_leaks[0]
        context = f"{getattr(primary, 'position', 'UNKNOWN')} / {getattr(primary, 'stack_depth_bucket', getattr(primary, 'stack_bucket', 'UNKNOWN'))}"
        base = {
            "VPIP_LOOSE": "Você está entrando em mãos demais para o contexto.",
            "VPIP_TIGHT": "Você está abrindo ou defendendo pouco demais no contexto.",
            "PFR_LOW": "Sua agressão pré-flop está abaixo do esperado.",
            "CALL_EXCESSIVE": "Você está pagando demais em spots de baixo EV.",
            "FOLD_EXCESSIVE": "Você está desistindo mais do que o contexto permite.",
            "THREE_BET_WRONG": "Sua estrutura de 3-bet merece revisão.",
            "FOUR_BET_WRONG": "Seu uso de 4-bet está fora do ótimo em parte dos spots.",
            "OVERPLAY_MEDIUM": "Você está overplayando mãos médias.",
            "BLUFF_BAD": "Seu processo de seleção de blefes está custando EV.",
            "VALUE_BET_LOST": "Você está perdendo value em spots que pediam captura de EV.",
            "TILT_PATTERN": "Há evidência de mudança comportamental após sequência ruim.",
            "ICM_OVERFOLD": "Você está cedendo valor demais na bolha e comprimindo seu $EV.",
            "ICM_OVERCALL": "Você está pagando shoves onde o risco de eliminação supera o ganho.",
            "ICM_SUICIDE": "Você está tomando calls suicidas na bolha/ITM.",
            "ICM_PASSIVITY": "Você está deixando de pressionar stacks médios quando a mesa pede agressão.",
        }.get(primary.leak_code, f"Leak principal identificado: {primary.leak_code}.")

        return (
            f"{base} Contexto principal: {context}. "
            f"EV perdido estimado: {getattr(primary, 'potential_roi_gain', 0.0):.2f} bb. "
            f"Amostra: {getattr(primary, 'sample_size', 0)} | Confiança: {getattr(primary, 'confidence', 0.0):.2f}."
        )

    def generate_full_report(self) -> Dict:
        self._ensure_context_data()
        critical_hands = []
        for rec in self.study_recommendations:
            if getattr(rec, "severity_label", "") == "CRÍTICO":
                for hid in getattr(rec, "representative_hand_ids", [])[:5]:
                    if hid not in critical_hands:
                        critical_hands.append(hid)
        why_lost = [
            {
                "leak": s.leak_code,
                "position": s.position,
                "stack_bucket": s.stack_depth_bucket,
                "why_i_lost": getattr(s, "why_i_lost", ""),
                "shap_like_values": getattr(s, "shap_like_values", {}),
            }
            for s in self.severity_scores[:10]
        ]
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "analysis_version": "2.0-auditable",
                "rows": len(self.df_hands),
            },
            "executive_summary": self.generate_executive_summary(),
            "breakdowns": {
                "by_position": self._breakdown_by_position(),
                "by_stack": self._breakdown_by_stack(),
                "by_phase": self._breakdown_by_phase(),
                "by_opponent_type": self._breakdown_by_opponent_type(),
                "by_severity": self._breakdown_by_severity(),
            },
            "top_10_leaks": self._top_10_leaks(),
            "top_10_costly_spots": self._top_10_costly_spots(),
            "study_plan": self._study_plan(),
            "why_i_lost": why_lost,
            "critical_hand_ids": critical_hands,
            "metrics": self._calculate_metrics(),
        }

    def _breakdown(self, column: str) -> Dict:
        if self.contexts.empty or column not in self.contexts.columns:
            return {}
        out = {}
        for value, group in self.contexts.groupby(column, dropna=False):
            key = str(value)
            sample_series = pd.to_numeric(group.get("sample_size", pd.Series([0] * len(group))), errors="coerce").fillna(0.0)
            confidence_series = pd.to_numeric(group.get("confidence", pd.Series([0] * len(group))), errors="coerce").fillna(0.0)
            avg_loss_series = pd.to_numeric(group.get("avg_loss_per_hand", pd.Series([0] * len(group))), errors="coerce").fillna(0.0)
            total_sample = float(sample_series.sum())
            weighted_conf = float((confidence_series * sample_series).sum() / total_sample) if total_sample > 0 else float(confidence_series.mean() if len(confidence_series) else 0.0)
            if total_sample > 0:
                avg_ev = float((avg_loss_series * sample_series).sum() / total_sample)
            else:
                avg_ev = float(avg_loss_series.mean() if len(avg_loss_series) else 0.0)
            out[key] = {
                "count": int(len(group)),
                "total_ev_lost": float(group["total_ev_lost"].sum()) if "total_ev_lost" in group.columns else 0.0,
                "avg_confidence": weighted_conf,
                "sample_size": int(total_sample),
                "avg_loss_per_hand": avg_ev,
            }
        return out

    def _breakdown_by_position(self) -> Dict:
        return self._breakdown("position")

    def _breakdown_by_stack(self) -> Dict:
        return self._breakdown("stack_depth_bucket") or self._breakdown("stack_bucket")

    def _breakdown_by_phase(self) -> Dict:
        return self._breakdown("phase")

    def _breakdown_by_opponent_type(self) -> Dict:
        return self._breakdown("opponent_type")

    def _breakdown_by_severity(self) -> Dict:
        breakdown = {"CRÍTICO": 0, "ALTO": 0, "MÉDIO": 0, "BAIXO": 0}
        for score in self.severity_scores:
            breakdown[score.severity_label] = breakdown.get(score.severity_label, 0) + 1
        return breakdown

    def _top_10_leaks(self) -> List[Dict]:
        return [
            {
                "rank": i + 1,
                "leak": s.leak_code,
                "position": s.position,
                "stack": s.stack_bucket,
                "stack_depth_bucket": s.stack_depth_bucket,
                "opponent_type": s.opponent_type,
                "phase": getattr(s, "phase", "UNKNOWN"),
                "severity": s.severity_label,
                "score": float(s.severity_score),
                "potential_gain": float(s.potential_roi_gain),
                "bayesian_ev_loss": float(getattr(s, "bayesian_ev_loss", 0.0)),
                "bayesian_bb100": float(getattr(s, "bayesian_bb100", 0.0)),
                "sample_size": int(getattr(s, "sample_size", 0)),
                "confidence": float(getattr(s, "confidence", 0.0)),
                "why_i_lost": getattr(s, "why_i_lost", ""),
                "shap_like_values": getattr(s, "shap_like_values", {}),
                "justification": getattr(s, "justification", ""),
                "evidence": getattr(s, "evidence", []),
            }
            for i, s in enumerate(self.severity_scores[:10])
        ]

    def _top_10_costly_spots(self) -> List[Dict]:
        if self.contexts.empty:
            return []
        top_costly = self.contexts.nlargest(min(10, len(self.contexts)), "total_ev_lost") if "total_ev_lost" in self.contexts.columns else self.contexts.head(10)
        return [
            {
                "rank": i + 1,
                "leak_code": row.get("leak_code"),
                "position": row.get("position"),
                "stack": row.get("stack_depth_bucket", row.get("stack_bucket")),
                "opponent": row.get("opponent_type"),
                "phase": row.get("phase"),
                "board_texture": row.get("board_texture"),
                "line_type": row.get("line_type"),
                "frequency": int(row.get("frequency", 0)),
                "sample_size": int(row.get("sample_size", 0)),
                "confidence": float(row.get("confidence", 0.0)),
                "total_cost": float(row.get("total_ev_lost", 0.0)),
                "avg_per_hand": float(row.get("avg_loss_per_hand", 0.0)),
                "ev_expected": row.get("ev_expected"),
                "ev_real": row.get("ev_real"),
                "ev_delta": row.get("ev_delta"),
                "evidence": row.get("evidence", ""),
            }
            for i, (_, row) in enumerate(top_costly.iterrows())
        ]

    def _study_plan(self) -> List[Dict]:
        return [
            {
                "priority": r.priority,
                "leak": r.leak_code,
                "recommendation": r.recommendation_text,
                "hours": r.estimated_hours,
                "expected_gain": r.expected_roi_gain,
                "spot": getattr(r, "spot", "UNKNOWN"),
                "bayesian_bb100": float(getattr(r, "bayesian_bb100", 0.0)),
                "sample_size": r.sample_size,
                "confidence": r.confidence,
                "evs_score": getattr(r, "evs_score", 0.0),
                "evidence": r.evidence,
                "representative_hand_ids": getattr(r, "representative_hand_ids", []),
                "hand_references": getattr(r, "hand_references", []),
            }
            for r in self.study_recommendations[:10]
        ]

    def _calculate_metrics(self) -> Dict:
        total_hands = len(self.df_hands)
        win_rate = float((self.df_hands["hero_result"] == "win").mean()) if total_hands and "hero_result" in self.df_hands.columns else 0.0
        return {
            "total_hands": total_hands,
            "win_rate": win_rate,
            "leaks_detected": len(self.leak_detections),
            "contexts_analyzed": len(self.contexts),
            "critical_leaks": sum(1 for s in self.severity_scores if s.severity_label == "CRÍTICO"),
            "high_leaks": sum(1 for s in self.severity_scores if s.severity_label == "ALTO"),
            "total_potential_gain": float(sum(s.potential_roi_gain for s in self.severity_scores)),
            "estimated_study_hours": float(sum(r.estimated_hours for r in self.study_recommendations)),
            "analysis_warnings": self._warnings(),
        }

    def to_json(self, filepath: str) -> None:
        report = self.generate_full_report()
        def _json_default(obj):
            if hasattr(obj, "item"):
                return obj.item()
            return str(obj)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)
        log.info("Relatório salvo em %s", filepath)

    def to_html(self, filepath: str) -> None:
        report = self.generate_full_report()
        warnings_html = "".join(f"<li>{w}</li>" for w in report["executive_summary"].get("analysis_warnings", []))
        top_leaks_html = "".join(
            f"<tr><td>{leak['rank']}</td><td>{leak['leak']}</td><td>{leak['position']}</td><td>{leak['stack']}</td><td>{leak['opponent_type']}</td><td>{leak['severity']}</td><td>{leak['score']:.1f}</td><td>{leak['sample_size']}</td><td>{leak['confidence']:.2f}</td><td>{leak['potential_gain']:.2f}</td></tr>"
            for leak in report["top_10_leaks"]
        )
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Relatório de Leaks</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; color: #222; }}
                .container {{ max-width: 1320px; margin: 0 auto; background: #fff; padding: 24px; border-radius: 12px; }}
                .warn {{ background: #fff7e6; border-left: 4px solid #ffb703; padding: 12px 16px; margin-bottom: 16px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
                th, td {{ padding: 10px; border-bottom: 1px solid #e5e5e5; text-align: left; vertical-align: top; }}
                th {{ background: #f7f7f7; position: sticky; top: 0; }}
                .small {{ color: #666; font-size: 0.92em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Relatório Auditável de Leaks</h1>
                <p class="small">Gerado em {report['metadata']['generated_at']} | versão {report['metadata']['analysis_version']}</p>
                <h2>Resumo Executivo</h2>
                <p><strong>{report['executive_summary']['key_message']}</strong></p>
                <div class="warn"><strong>Avisos</strong><ul>{warnings_html}</ul></div>
                <p>Mãos analisadas: {report['executive_summary']['total_hands_analyzed']}</p>
                <p>EV perdido estimado: {report['executive_summary']['total_ev_lost']:.2f} bb</p>
                <p>Leaks detectados: {report['executive_summary']['leaks_detected']}</p>
                <p>Ganho potencial: {report['executive_summary']['total_potential_gain']:.2f} bb</p>
                <h2>Top Leaks</h2>
                <table>
                    <tr><th>#</th><th>Leak</th><th>Posição</th><th>Stack</th><th>Vilão</th><th>Severidade</th><th>Score</th><th>Amostra</th><th>Confiança</th><th>Potencial</th></tr>
                    {top_leaks_html}
                </table>
            </div>
        </body>
        </html>
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        log.info("Relatório HTML salvo em %s", filepath)
