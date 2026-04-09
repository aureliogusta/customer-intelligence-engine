"""Canal console — sempre disponível, nunca falha."""
from __future__ import annotations
import logging
from typing import Any, Dict
from ...schemas.models import DispatchChannel, DispatchResult

log = logging.getLogger("dispatch.console")


def send(action_code: str, payload: Dict[str, Any]) -> DispatchResult:
    account = payload.get("account_name") or payload.get("account_id", "?")
    churn   = payload.get("churn_risk", "?")
    mrr     = payload.get("mrr_at_risk", "?")
    log.info("[CONSOLE] %s | conta=%s churn=%s mrr_at_risk=%s", action_code, account, churn, mrr)
    return DispatchResult(
        channel      = DispatchChannel.CONSOLE,
        action_code  = action_code,
        success      = True,
        message      = f"Logged to console: {action_code} for {account}",
        payload_sent = payload,
    )
