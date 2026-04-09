"""Canal api_hook — POST para endpoint interno configurável."""
from __future__ import annotations
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict
from ...schemas.models import DispatchChannel, DispatchResult

log = logging.getLogger("dispatch.api_hook")

API_HOOK_URL  = os.getenv("API_HOOK_URL", "")
TIMEOUT_SEC   = int(os.getenv("API_HOOK_TIMEOUT", "10"))


def send(action_code: str, payload: Dict[str, Any], hook_url: str = "") -> DispatchResult:
    url = hook_url or API_HOOK_URL
    if not url:
        return DispatchResult(
            channel      = DispatchChannel.API_HOOK,
            action_code  = action_code,
            success      = False,
            message      = "API_HOOK_URL not configured",
            payload_sent = payload,
            error        = "missing_hook_url",
        )

    body = json.dumps({"action": action_code, **payload}, ensure_ascii=False, default=str).encode("utf-8")
    req  = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            return DispatchResult(
                channel      = DispatchChannel.API_HOOK,
                action_code  = action_code,
                success      = resp.status < 400,
                message      = f"API hook HTTP {resp.status}",
                payload_sent = payload,
            )
    except Exception as e:
        log.error("dispatch.api_hook failed: %s", e)
        return DispatchResult(
            channel      = DispatchChannel.API_HOOK,
            action_code  = action_code,
            success      = False,
            message      = "API hook request failed",
            payload_sent = payload,
            error        = str(e),
        )
