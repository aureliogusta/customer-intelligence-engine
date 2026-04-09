"""
revenue_automation/reports/builder.py
=======================================
ReportBuilder — agrega dados de predições, intervenções e feedback num ReportData.

Aceita:
  - Lista de InterventionContext (predições do período)
  - Lista de InterventionRecord (intervenções executadas)
  - FeedbackStore (para contar outcomes)
  - DriftReport (opcional)

Produz:
  - ReportData pronto para renderização
"""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..schemas.models import (
    AccountRiskSummary,
    InterventionContext,
    InterventionRecord,
    ReportData,
    ReportPeriod,
)


class ReportBuilder:
    """
    Constrói ReportData a partir dos dados coletados no período.

    Uso:
        builder = ReportBuilder(period=ReportPeriod.WEEKLY, ...)
        data    = builder.build()
    """

    def __init__(
        self,
        period:          ReportPeriod,
        period_start:    str,
        period_end:      str,
        contexts:        List[InterventionContext],
        records:         List[InterventionRecord],
        feedback_counts: Dict[str, int] | None = None,
        drift_report:    Any | None = None,
        critical_mrr_threshold: float = 5000.0,
    ):
        self.period           = period
        self.period_start     = period_start
        self.period_end       = period_end
        self.contexts         = contexts
        self.records          = records
        self.feedback_counts  = feedback_counts or {}
        self.drift_report     = drift_report
        self.critical_mrr_threshold = critical_mrr_threshold

    def build(self) -> ReportData:
        ctxs = self.contexts
        recs = self.records

        total          = len(ctxs)
        at_risk        = sum(1 for c in ctxs if c.risk_level == "HIGH")
        critical       = sum(1 for c in ctxs if c.risk_level == "HIGH" and c.mrr >= self.critical_mrr_threshold)
        recoverable    = sum(1 for c in ctxs if c.risk_level == "MEDIUM")
        safe           = sum(1 for c in ctxs if c.risk_level == "LOW")

        total_mrr      = sum(c.mrr for c in ctxs)
        mrr_at_risk    = sum(c.mrr_at_risk for c in ctxs)

        # MRR preservado = mrr_at_risk de contas onde ação foi tomada (não MONITOR_ONLY)
        acted_accounts = {r.account_id for r in recs if r.actions_taken != ["MONITOR_ONLY"]}
        mrr_preserved  = sum(
            c.mrr_at_risk for c in ctxs
            if c.account_id in acted_accounts and c.risk_level in ("HIGH", "MEDIUM")
        )

        top_risk = sorted(
            [
                AccountRiskSummary(
                    account_id   = c.account_id,
                    account_name = c.account_name or c.account_id,
                    segment      = c.segment,
                    mrr          = c.mrr,
                    churn_risk   = c.churn_risk,
                    risk_level   = c.risk_level,
                    mrr_at_risk  = c.mrr_at_risk,
                    top_action   = self._top_action_for(c.account_id, recs),
                )
                for c in ctxs
                if c.risk_level == "HIGH"
            ],
            key=lambda x: x.mrr_at_risk,
            reverse=True,
        )[:10]

        segment_breakdown = self._segment_breakdown(ctxs)
        top_actions       = self._top_actions(recs)
        channels_used     = self._channels_used(recs)
        drift_summary, drift_detected = self._drift_info()

        executive = self._executive_summary(
            total, at_risk, critical, total_mrr, mrr_at_risk, mrr_preserved,
            top_actions, drift_detected,
        )

        return ReportData(
            period                  = self.period.value,
            period_start            = self.period_start,
            period_end              = self.period_end,
            generated_at            = datetime.now(timezone.utc).isoformat(),
            total_accounts          = total,
            accounts_at_risk        = at_risk,
            critical_accounts       = critical,
            recoverable_accounts    = recoverable,
            safe_accounts           = safe,
            total_mrr_analyzed      = round(total_mrr, 2),
            total_mrr_at_risk       = round(mrr_at_risk, 2),
            estimated_mrr_preserved = round(mrr_preserved, 2),
            top_risk_accounts       = top_risk,
            segment_breakdown       = segment_breakdown,
            top_actions_taken       = top_actions,
            channels_used           = channels_used,
            feedback_outcomes       = self.feedback_counts,
            drift_summary           = drift_summary,
            drift_detected          = drift_detected,
            executive_summary       = executive,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _top_action_for(self, account_id: str, records: List[InterventionRecord]) -> str:
        for r in records:
            if r.account_id == account_id and r.actions_taken:
                return r.actions_taken[0]
        return "MONITOR_ONLY"

    def _segment_breakdown(self, ctxs: List[InterventionContext]) -> Dict[str, Dict]:
        breakdown: Dict[str, Dict] = {}
        for c in ctxs:
            seg = c.segment or "UNKNOWN"
            if seg not in breakdown:
                breakdown[seg] = {"total": 0, "high_risk": 0, "mrr": 0.0, "mrr_at_risk": 0.0}
            breakdown[seg]["total"]      += 1
            breakdown[seg]["mrr"]        += c.mrr
            breakdown[seg]["mrr_at_risk"] += c.mrr_at_risk
            if c.risk_level == "HIGH":
                breakdown[seg]["high_risk"] += 1
        # Round MRR values
        for seg in breakdown:
            breakdown[seg]["mrr"]         = round(breakdown[seg]["mrr"], 2)
            breakdown[seg]["mrr_at_risk"] = round(breakdown[seg]["mrr_at_risk"], 2)
        return breakdown

    def _top_actions(self, records: List[InterventionRecord]) -> Dict[str, int]:
        counter: Counter = Counter()
        for r in records:
            for a in r.actions_taken:
                counter[a] += 1
        return dict(counter.most_common(10))

    def _channels_used(self, records: List[InterventionRecord]) -> Dict[str, int]:
        counter: Counter = Counter()
        for r in records:
            for ch in r.channels_used:
                counter[ch] += 1
        return dict(counter.most_common())

    def _drift_info(self):
        if self.drift_report is None:
            return None, False
        summary = getattr(self.drift_report, "summary_line", lambda: None)()
        detected = getattr(self.drift_report, "should_retrain", False)
        return summary, detected

    def _executive_summary(
        self,
        total: int,
        at_risk: int,
        critical: int,
        total_mrr: float,
        mrr_at_risk: float,
        mrr_preserved: float,
        top_actions: Dict[str, int],
        drift_detected: bool,
    ) -> str:
        pct_risk     = (at_risk / total * 100) if total else 0
        pct_mrr_risk = (mrr_at_risk / total_mrr * 100) if total_mrr else 0
        top_action   = next(iter(top_actions), "nenhuma")

        lines = [
            f"No período analisado, {total} contas foram avaliadas.",
            f"{at_risk} ({pct_risk:.1f}%) estão em risco alto de churn, "
            f"sendo {critical} classificadas como críticas (MRR > R$ 5.000).",
            f"O MRR total em risco é R$ {mrr_at_risk:,.2f} "
            f"({pct_mrr_risk:.1f}% do MRR analisado).",
        ]
        if mrr_preserved > 0:
            lines.append(
                f"Intervenções foram disparadas para contas representando "
                f"R$ {mrr_preserved:,.2f} em MRR potencialmente preservado."
            )
        if top_action != "nenhuma":
            lines.append(f"A ação mais frequente foi: {top_action}.")
        if drift_detected:
            lines.append(
                "ATENCAO: drift estatístico detectado nas features de entrada — "
                "retreino do modelo é recomendado."
            )
        return " ".join(lines)
