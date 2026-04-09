"""Canal email — SMTP via variáveis de ambiente. Fallback para console se não configurado."""
from __future__ import annotations
import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Any, Dict
from ...schemas.models import DispatchChannel, DispatchResult

log = logging.getLogger("dispatch.email")

SMTP_HOST     = os.getenv("SMTP_HOST", "")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM    = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO      = os.getenv("EMAIL_TO", "")       # comma-separated
TIMEOUT_SEC   = int(os.getenv("SMTP_TIMEOUT", "10"))


def _build_body(action_code: str, payload: Dict[str, Any]) -> str:
    lines = [
        f"Ação automática disparada: {action_code}",
        "",
        f"Conta       : {payload.get('account_name') or payload.get('account_id', '?')}",
        f"Segmento    : {payload.get('segment', '?')}",
        f"MRR         : {payload.get('mrr', '?')}",
        f"Churn Risk  : {payload.get('churn_risk', '?')}",
        f"MRR em risco: {payload.get('mrr_at_risk', '?')}",
        f"CSM         : {payload.get('csm', 'N/A')}",
        "",
        "-- CS Intelligence Platform (automated) --",
    ]
    return "\n".join(lines)


def send(action_code: str, payload: Dict[str, Any]) -> DispatchResult:
    if not SMTP_HOST or not EMAIL_TO:
        log.debug("SMTP não configurado — skip email dispatch")
        return DispatchResult(
            channel      = DispatchChannel.EMAIL,
            action_code  = action_code,
            success      = False,
            message      = "SMTP not configured (SMTP_HOST or EMAIL_TO missing)",
            payload_sent = payload,
            error        = "missing_smtp_config",
        )

    account = payload.get("account_name") or payload.get("account_id", "?")
    subject = f"[CS Alert] {action_code} — {account}"
    body    = _build_body(action_code, payload)

    msg           = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=TIMEOUT_SEC) as server:
            server.ehlo()
            if SMTP_PASSWORD:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO.split(","), msg.as_string())
        return DispatchResult(
            channel      = DispatchChannel.EMAIL,
            action_code  = action_code,
            success      = True,
            message      = f"Email sent to {EMAIL_TO}",
            payload_sent = payload,
        )
    except Exception as e:
        log.error("dispatch.email failed: %s", e)
        return DispatchResult(
            channel      = DispatchChannel.EMAIL,
            action_code  = action_code,
            success      = False,
            message      = "Email send failed",
            payload_sent = payload,
            error        = str(e),
        )
