"""
revenue_automation/schemas/models.py
======================================
Tipos centrais do subsistema de automação de intervenção.
Todos os módulos importam daqui — zero acoplamento circular.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums ─────────────────────────────────────────────────────────────────────

class ActionCode(str, Enum):
    SEND_EMAIL            = "SEND_EMAIL"
    CREATE_CRM_TASK       = "CREATE_CRM_TASK"
    SEND_SLACK_ALERT      = "SEND_SLACK_ALERT"
    ESCALATE_TO_MANAGER   = "ESCALATE_TO_MANAGER"
    SCHEDULE_CALL         = "SCHEDULE_CALL"
    TRIGGER_QBR           = "TRIGGER_QBR"
    SEND_INTERNAL_REPORT  = "SEND_INTERNAL_REPORT"
    SEND_CUSTOMER_CHECKIN = "SEND_CUSTOMER_CHECKIN"
    MONITOR_ONLY          = "MONITOR_ONLY"
    PROPOSE_UPSELL        = "PROPOSE_UPSELL"
    SHARE_ROADMAP         = "SHARE_ROADMAP"


class DispatchChannel(str, Enum):
    EMAIL    = "email"
    SLACK    = "slack"
    CONSOLE  = "console"
    FILE     = "file"
    API_HOOK = "api_hook"


class FeedbackOutcome(str, Enum):
    CUSTOMER_REPLIED  = "customer_replied"
    TICKET_CREATED    = "ticket_created"
    MEETING_BOOKED    = "meeting_booked"
    RISK_DECREASED    = "risk_decreased"
    RENEWAL_HAPPENED  = "renewal_happened"
    UPSELL_HAPPENED   = "upsell_happened"
    NO_RESPONSE       = "no_response"
    UNKNOWN           = "unknown"


class ReportPeriod(str, Enum):
    DAILY      = "daily"
    WEEKLY     = "weekly"
    BIWEEKLY   = "biweekly"
    MONTHLY    = "monthly"


# ── Input context ─────────────────────────────────────────────────────────────

@dataclass
class InterventionContext:
    """
    Contexto completo de uma conta no momento da avaliação.
    Alimenta o PolicyEngine e o Dispatcher.
    """
    account_id:              str
    session_id:              str

    # ML scores
    churn_risk:              float
    retention_prob:          float
    risk_level:              str          # HIGH / MEDIUM / LOW
    upsell_probability:      float
    mrr_at_risk:             float

    # Account metadata
    mrr:                     float
    segment:                 str          # ENTERPRISE / MID_MARKET / SMB
    dias_sem_interacao:      int
    tickets_abertos:         int
    engajamento_pct:         float
    nps_score:               float
    dias_no_contrato:        int

    # Trend
    engagement_trend:        float
    nps_trend:               float
    tickets_trend:           float

    # Health score (optional, from dashboard)
    overall_score:           Optional[float] = None
    rag_status:              Optional[str]   = None   # GREEN / AMBER / RED
    trend_delta:             Optional[float] = None   # score drop last 7 days

    # Extra
    account_name:            str = ""
    csm_name:                str = ""
    timestamp:               str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_prediction_turn(cls, turn: Any, df_row: Dict[str, Any] | None = None) -> "InterventionContext":
        """Constrói a partir de um PredictionTurn + metadados opcionais do CSV."""
        meta = df_row or {}
        return cls(
            account_id         = turn.account_id,
            session_id         = turn.session_id,
            churn_risk         = turn.churn_risk,
            retention_prob     = turn.retention_prob,
            risk_level         = turn.risk_level,
            upsell_probability = turn.upsell_probability or 0.0,
            mrr_at_risk        = turn.mrr_at_risk,
            mrr                = turn.mrr,
            segment            = str(meta.get("segment", "UNKNOWN")),
            dias_sem_interacao = int(meta.get("dias_sem_interacao", 0)),
            tickets_abertos    = int(meta.get("tickets_abertos", 0)),
            engajamento_pct    = float(meta.get("engajamento_pct", 0.0)),
            nps_score          = float(meta.get("nps_score", 5.0)),
            dias_no_contrato   = int(meta.get("dias_no_contrato", 0)),
            engagement_trend   = float(meta.get("engagement_trend", 0.0)),
            nps_trend          = float(meta.get("nps_trend", 0.0)),
            tickets_trend      = float(meta.get("tickets_trend", 0.0)),
            account_name       = str(meta.get("name", "")),
            csm_name           = str(meta.get("csm_name", "")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Policy output ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SelectedAction:
    code:           ActionCode
    priority:       int
    reason:         str          # qual regra disparou
    channels:       List[DispatchChannel]
    payload:        Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    account_id:      str
    session_id:      str
    timestamp:       str
    actions:         List[SelectedAction]
    triggered_rules: List[str]
    monitor_only:    bool        # True se só MONITOR_ONLY foi selecionado

    def top_action(self) -> Optional[SelectedAction]:
        return self.actions[0] if self.actions else None

    def action_codes(self) -> List[str]:
        return [a.code.value for a in self.actions]


# ── Dispatch ──────────────────────────────────────────────────────────────────

@dataclass
class DispatchResult:
    channel:        DispatchChannel
    action_code:    str
    success:        bool
    message:        str
    payload_sent:   Dict[str, Any] = field(default_factory=dict)
    error:          Optional[str]  = None


# ── Intervention record (full audit row) ──────────────────────────────────────

@dataclass
class InterventionRecord:
    """Registro completo de uma intervenção — persiste no audit trail."""
    correlation_id:    str
    account_id:        str
    session_id:        str
    timestamp:         str

    previous_score:    Optional[float]
    current_score:     Optional[float]
    churn_risk:        float
    risk_level:        str

    actions_taken:     List[str]
    channels_used:     List[str]
    reasons:           List[str]

    dispatch_results:  List[Dict[str, Any]]
    payload_summary:   Dict[str, Any]

    acknowledged:      bool = False
    ack_timestamp:     Optional[str] = None

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


# ── Feedback ──────────────────────────────────────────────────────────────────

@dataclass
class FeedbackEntry:
    feedback_id:       str
    correlation_id:    str      # links back to InterventionRecord
    account_id:        str
    outcome:           str      # FeedbackOutcome value
    notes:             str
    recorded_at:       str
    recorded_by:       str = "system"
    churn_risk_before: Optional[float] = None
    churn_risk_after:  Optional[float] = None

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class AccountRiskSummary:
    account_id:    str
    account_name:  str
    segment:       str
    mrr:           float
    churn_risk:    float
    risk_level:    str
    mrr_at_risk:   float
    top_action:    str


@dataclass
class ReportData:
    period:                 str
    period_start:           str
    period_end:             str
    generated_at:           str

    total_accounts:         int
    accounts_at_risk:       int      # HIGH
    critical_accounts:      int      # HIGH + mrr > threshold
    recoverable_accounts:   int      # MEDIUM
    safe_accounts:          int      # LOW

    total_mrr_analyzed:     float
    total_mrr_at_risk:      float
    estimated_mrr_preserved: float   # mrr_at_risk where action was taken

    top_risk_accounts:      List[AccountRiskSummary]
    segment_breakdown:      Dict[str, Dict[str, Any]]  # segment → counts + mrr

    top_actions_taken:      Dict[str, int]   # action_code → count
    channels_used:          Dict[str, int]   # channel → count
    feedback_outcomes:      Dict[str, int]   # outcome → count

    drift_summary:          Optional[str]
    drift_detected:         bool

    executive_summary:      str      # plain language

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
