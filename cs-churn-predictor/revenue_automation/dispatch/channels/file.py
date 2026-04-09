"""Canal file — escreve em JSONL local."""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from ...schemas.models import DispatchChannel, DispatchResult

log = logging.getLogger("dispatch.file")

DEFAULT_PATH = Path(os.getenv("DISPATCH_FILE_PATH", "logs/dispatched_actions.jsonl"))


def send(action_code: str, payload: Dict[str, Any], path: Path = DEFAULT_PATH) -> DispatchResult:
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "action_code": action_code,
            **payload,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return DispatchResult(
            channel      = DispatchChannel.FILE,
            action_code  = action_code,
            success      = True,
            message      = f"Written to {path}",
            payload_sent = payload,
        )
    except Exception as e:
        log.error("dispatch.file failed: %s", e)
        return DispatchResult(
            channel      = DispatchChannel.FILE,
            action_code  = action_code,
            success      = False,
            message      = "File write failed",
            payload_sent = payload,
            error        = str(e),
        )
