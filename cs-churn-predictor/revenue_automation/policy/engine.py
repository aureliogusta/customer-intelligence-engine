"""
revenue_automation/policy/engine.py
=====================================
PolicyEngine — avalia regras YAML contra o InterventionContext e decide ações.

Design:
  - Regras carregadas de YAML (configurável, não hardcoded)
  - Cada regra tem condições compostas (AND implícito entre campos)
  - Operadores: gt, lt, gte, lte, eq, ne, in, not_in
  - Regras avaliadas em ordem de prioridade
  - Flag `stop=True` interrompe avaliação após match (short-circuit)
  - Resultado: PolicyDecision com lista ordenada de SelectedAction

Uso:
    engine = PolicyEngine.from_default()
    decision = engine.evaluate(context)
    print(decision.action_codes())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..schemas.models import (
    ActionCode,
    DispatchChannel,
    InterventionContext,
    PolicyDecision,
    SelectedAction,
)

log = logging.getLogger(__name__)

DEFAULT_RULES_PATH = Path(__file__).parent / "default_rules.yaml"

# ── Rule dataclass ────────────────────────────────────────────────────────────

@dataclass
class RuleAction:
    code:     ActionCode
    channels: List[DispatchChannel]


@dataclass
class Rule:
    name:       str
    priority:   int
    conditions: Dict[str, Any]   # field → {operator: value}
    actions:    List[RuleAction]
    reason:     str
    stop:       bool = False


# ── Condition evaluator ───────────────────────────────────────────────────────

_OPS = {
    "gt":      lambda v, t: v >  t,
    "lt":      lambda v, t: v <  t,
    "gte":     lambda v, t: v >= t,
    "lte":     lambda v, t: v <= t,
    "eq":      lambda v, t: str(v).upper() == str(t).upper(),
    "ne":      lambda v, t: str(v).upper() != str(t).upper(),
    "in":      lambda v, t: v in t,
    "not_in":  lambda v, t: v not in t,
}


def _eval_condition(value: Any, spec: Dict[str, Any]) -> bool:
    """Avalia um campo contra todos os operadores definidos (AND entre operadores)."""
    for op, threshold in spec.items():
        fn = _OPS.get(op)
        if fn is None:
            log.warning("Operador desconhecido na regra: %s", op)
            continue
        try:
            if not fn(value, threshold):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _match_rule(rule: Rule, ctx: InterventionContext) -> bool:
    """Retorna True se TODAS as condições da regra forem satisfeitas."""
    ctx_dict = ctx.as_dict()
    for field_name, spec in rule.conditions.items():
        value = ctx_dict.get(field_name)
        if value is None:
            return False
        if not _eval_condition(value, spec):
            return False
    return True


# ── YAML loader ───────────────────────────────────────────────────────────────

def _load_rules(path: Path) -> List[Rule]:
    try:
        import yaml  # type: ignore
    except ImportError:
        # Fallback: parse manual mínimo se PyYAML não estiver instalado
        log.warning("PyYAML não disponível — usando regras embutidas mínimas.")
        return _builtin_rules()

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules: List[Rule] = []
    for item in data.get("rules", []):
        rule_actions: List[RuleAction] = []
        for a in item.get("actions", []):
            code = ActionCode(a["code"])
            channels = [DispatchChannel(c) for c in a.get("channels", ["console"])]
            rule_actions.append(RuleAction(code=code, channels=channels))

        rules.append(Rule(
            name       = item["name"],
            priority   = int(item.get("priority", 99)),
            conditions = item.get("conditions", {}),
            actions    = rule_actions,
            reason     = item.get("reason", item["name"]),
            stop       = bool(item.get("stop", False)),
        ))

    return sorted(rules, key=lambda r: r.priority)


def _builtin_rules() -> List[Rule]:
    """Regras mínimas embutidas — fallback quando PyYAML não disponível."""
    return [
        Rule(
            name       = "critical_high_mrr",
            priority   = 1,
            conditions = {"churn_risk": {"gt": 0.80}, "mrr": {"gt": 5000}},
            actions    = [
                RuleAction(ActionCode.ESCALATE_TO_MANAGER, [DispatchChannel.CONSOLE, DispatchChannel.FILE]),
                RuleAction(ActionCode.SEND_SLACK_ALERT,    [DispatchChannel.SLACK,   DispatchChannel.CONSOLE]),
                RuleAction(ActionCode.SEND_EMAIL,          [DispatchChannel.EMAIL,   DispatchChannel.CONSOLE]),
            ],
            reason = "Conta crítica: churn > 80% com MRR alto",
            stop   = True,
        ),
        Rule(
            name       = "high_risk_action",
            priority   = 3,
            conditions = {"churn_risk": {"gt": 0.60}},
            actions    = [
                RuleAction(ActionCode.CREATE_CRM_TASK, [DispatchChannel.CONSOLE, DispatchChannel.FILE]),
                RuleAction(ActionCode.SCHEDULE_CALL,   [DispatchChannel.CONSOLE, DispatchChannel.FILE]),
            ],
            reason = "Churn 60-80%",
        ),
        Rule(
            name       = "upsell_opportunity",
            priority   = 8,
            conditions = {"upsell_probability": {"gt": 0.50}, "churn_risk": {"lt": 0.30}},
            actions    = [
                RuleAction(ActionCode.PROPOSE_UPSELL, [DispatchChannel.EMAIL, DispatchChannel.CONSOLE]),
            ],
            reason = "Alta prob. de upsell + baixo risco",
        ),
        Rule(
            name       = "healthy_monitor",
            priority   = 10,
            conditions = {"churn_risk": {"lt": 0.40}},
            actions    = [
                RuleAction(ActionCode.MONITOR_ONLY, [DispatchChannel.CONSOLE]),
            ],
            reason = "Conta saudável",
        ),
    ]


# ── Engine ────────────────────────────────────────────────────────────────────

class PolicyEngine:
    """
    Avalia regras de política e retorna um PolicyDecision.

    Regras são carregadas de YAML e podem ser recarregadas em runtime
    chamando `.reload()`.
    """

    def __init__(self, rules_path: Path = DEFAULT_RULES_PATH):
        self.rules_path = rules_path
        self.rules: List[Rule] = _load_rules(rules_path) if rules_path.exists() else _builtin_rules()
        log.info("PolicyEngine: %d regras carregadas de %s", len(self.rules), rules_path)

    @classmethod
    def from_default(cls) -> "PolicyEngine":
        return cls(rules_path=DEFAULT_RULES_PATH)

    @classmethod
    def from_file(cls, path: str | Path) -> "PolicyEngine":
        return cls(rules_path=Path(path))

    def reload(self) -> None:
        """Recarrega regras do YAML sem reiniciar o processo."""
        self.rules = _load_rules(self.rules_path)
        log.info("PolicyEngine: regras recarregadas (%d)", len(self.rules))

    def evaluate(self, ctx: InterventionContext) -> PolicyDecision:
        """
        Avalia o contexto contra todas as regras em ordem de prioridade.
        Retorna PolicyDecision com a lista de ações selecionadas.
        """
        matched_actions: List[SelectedAction] = []
        triggered_rules: List[str] = []

        for rule in self.rules:
            if not _match_rule(rule, ctx):
                continue

            triggered_rules.append(rule.name)
            for ra in rule.actions:
                matched_actions.append(SelectedAction(
                    code     = ra.code,
                    priority = rule.priority,
                    reason   = rule.reason,
                    channels = ra.channels,
                    payload  = _build_payload(ctx, ra.code),
                ))

            if rule.stop:
                log.debug("Regra '%s' com stop=True — avaliação encerrada", rule.name)
                break

        # Dedup por code, mantendo prioridade mais alta
        seen: dict[ActionCode, SelectedAction] = {}
        for a in matched_actions:
            if a.code not in seen or a.priority < seen[a.code].priority:
                seen[a.code] = a

        final_actions = sorted(seen.values(), key=lambda a: a.priority)
        monitor_only  = (
            len(final_actions) == 1 and
            final_actions[0].code == ActionCode.MONITOR_ONLY
        )

        return PolicyDecision(
            account_id      = ctx.account_id,
            session_id      = ctx.session_id,
            timestamp       = datetime.now(timezone.utc).isoformat(),
            actions         = final_actions,
            triggered_rules = triggered_rules,
            monitor_only    = monitor_only,
        )


# ── Payload builder ───────────────────────────────────────────────────────────

def _build_payload(ctx: InterventionContext, code: ActionCode) -> Dict[str, Any]:
    """Monta payload contextualizado para cada tipo de ação."""
    base = {
        "account_id":   ctx.account_id,
        "account_name": ctx.account_name or ctx.account_id,
        "segment":      ctx.segment,
        "mrr":          ctx.mrr,
        "churn_risk":   f"{ctx.churn_risk:.1%}",
        "mrr_at_risk":  f"R$ {ctx.mrr_at_risk:,.2f}",
        "csm":          ctx.csm_name or "N/A",
    }

    if code in (ActionCode.ESCALATE_TO_MANAGER, ActionCode.SEND_SLACK_ALERT):
        base.update({
            "urgency":          "CRITICAL" if ctx.churn_risk > 0.80 else "HIGH",
            "nps_score":        ctx.nps_score,
            "engajamento_pct":  f"{ctx.engajamento_pct:.1f}%",
            "dias_sem_interacao": ctx.dias_sem_interacao,
        })

    elif code == ActionCode.PROPOSE_UPSELL:
        base.update({
            "upsell_probability": f"{ctx.upsell_probability:.1%}",
            "retention_prob":     f"{ctx.retention_prob:.1%}",
        })

    elif code == ActionCode.TRIGGER_QBR:
        base.update({
            "dias_no_contrato": ctx.dias_no_contrato,
            "engagement_trend": f"{ctx.engagement_trend:+.0%}",
        })

    return base
