"""
study_service/recommendations.py
==================================
Gera recomendações de ação priorizadas baseadas no risco de churn e probabilidade de upsell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Action:
    code:        str
    label:       str
    priority:    int    # 1 = mais urgente
    description: str


# Catálogo de ações disponíveis
_ACTIONS: Dict[str, Action] = {
    "ESCALATE": Action(
        code="ESCALATE",
        label="Escalar para liderança",
        priority=1,
        description="Envolver C-level / VP de CS — MRR em risco alto.",
    ),
    "SCHEDULE_CALL": Action(
        code="SCHEDULE_CALL",
        label="Agendar call de emergência",
        priority=2,
        description="Contato direto com decision-maker da conta.",
    ),
    "SEND_DISCOUNT": Action(
        code="SEND_DISCOUNT",
        label="Oferecer desconto de retenção",
        priority=3,
        description="Enviar proposta comercial com incentivo para renovação.",
    ),
    "SCHEDULE_QBR": Action(
        code="SCHEDULE_QBR",
        label="Agendar QBR",
        priority=3,
        description="Business Review para realinhar valor entregue.",
    ),
    "SEND_SURVEY": Action(
        code="SEND_SURVEY",
        label="Enviar pesquisa de satisfação",
        priority=4,
        description="NPS/CSAT para entender causas de insatisfação.",
    ),
    "SEND_CONTENT": Action(
        code="SEND_CONTENT",
        label="Enviar conteúdo de nurture",
        priority=5,
        description="Cases de sucesso, webinars ou documentação relevante.",
    ),
    "TRACK_ENGAGEMENT": Action(
        code="TRACK_ENGAGEMENT",
        label="Monitorar engajamento",
        priority=6,
        description="Conta saudável — manter acompanhamento regular.",
    ),
    "PROPOSE_UPSELL": Action(
        code="PROPOSE_UPSELL",
        label="Propor upsell / expansão",
        priority=2,
        description="Conta com alto engajamento e NPS — oportunidade de expansão.",
    ),
    "SHARE_ROADMAP": Action(
        code="SHARE_ROADMAP",
        label="Compartilhar roadmap de produto",
        priority=4,
        description="Mostrar novas features para estimular adoção e expansão.",
    ),
}


def gerar_recomendacoes(
    account_id: str,
    churn_risk: float,
    upsell_probability: float = 0.0,
    mrr: float = 0.0,
) -> List[Dict]:
    """
    Retorna lista de ações recomendadas ordenadas por prioridade.

    Args:
        account_id:         ID da conta
        churn_risk:         Probabilidade de churn (0.0–1.0)
        upsell_probability: Probabilidade de upsell (0.0–1.0)
        mrr:                MRR da conta (para contextualizar impacto)

    Returns:
        Lista de dicts com code, label, priority, description, mrr_at_risk
    """
    codes: List[str] = []

    # Lógica de risco
    if churn_risk > 0.70:
        codes += ["ESCALATE", "SCHEDULE_CALL", "SEND_DISCOUNT"]
        if mrr > 5000:
            codes.insert(0, "ESCALATE")  # reforça prioridade para contas grandes
    elif churn_risk > 0.40:
        codes += ["SCHEDULE_CALL", "SEND_SURVEY", "SCHEDULE_QBR"]
    else:
        codes += ["TRACK_ENGAGEMENT", "SEND_CONTENT"]
        if upsell_probability > 0.50:
            codes = ["PROPOSE_UPSELL", "SHARE_ROADMAP"] + codes

    # Dedup preservando ordem
    seen: set[str] = set()
    unique_codes = [c for c in codes if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

    actions = []
    for code in unique_codes:
        a = _ACTIONS.get(code)
        if a:
            actions.append({
                "code":        a.code,
                "label":       a.label,
                "priority":    a.priority,
                "description": a.description,
                "mrr_at_risk": round(mrr * churn_risk, 2),
            })

    return sorted(actions, key=lambda x: x["priority"])
