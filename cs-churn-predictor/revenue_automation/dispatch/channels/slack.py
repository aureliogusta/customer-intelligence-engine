"""Canal Slack — HTTP POST para webhook URL. Fallback para console se URL ausente."""
from __future__ import annotations
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict
from ...schemas.models import DispatchChannel, DispatchResult

log = logging.getLogger("dispatch.slack")

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
TIMEOUT_SEC       = int(os.getenv("SLACK_TIMEOUT", "10"))


def _format_message(action_code: str, payload: Dict[str, Any]) -> str:
    account  = payload.get("account_name") or payload.get("account_id", "?")
    churn    = payload.get("churn_risk", "?")
    mrr      = payload.get("mrr_at_risk", "R$ ?")
    urgency  = payload.get("urgency", "")
    segment  = payload.get("segment", "")
    csm      = payload.get("csm", "")
    return (
        f"*[{action_code}]* {urgency} | Conta: *{account}* ({segment})\n"
        f"> Churn risk: *{churn}* | MRR em risco: *{mrr}*\n"
        f"> CSM: {csm}"
    )


def send(action_code: str, payload: Dict[str, Any], webhook_url: str = "") -> DispatchResult:
    url = webhook_url or SLACK_WEBHOOK_URL
    if not url:
        log.debug("SLACK_WEBHOOK_URL não configurado — skip Slack dispatch")
        return DispatchResult(
            channel      = DispatchChannel.SLACK,
            action_code  = action_code,
            success      = False,
            message      = "SLACK_WEBHOOK_URL not configured",
            payload_sent = payload,
            error        = "missing_webhook_url",
        )

    body = json.dumps({"text": _format_message(action_code, payload)}).encode("utf-8")
    req  = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            ok = resp.status == 200
            return DispatchResult(
                channel      = DispatchChannel.SLACK,
                action_code  = action_code,
                success      = ok,
                message      = f"Slack HTTP {resp.status}",
                payload_sent = payload,
            )
    except Exception as e:
        log.error("dispatch.slack failed: %s", e)
        return DispatchResult(
            channel      = DispatchChannel.SLACK,
            action_code  = action_code,
            success      = False,
            message      = "Slack request failed",
            payload_sent = payload,
            error        = str(e),
        )
