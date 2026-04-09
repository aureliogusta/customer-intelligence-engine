from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def calculate_bubble_factor(stack_bb: float, phase: str) -> float:
    """Centralized ICM pressure (bubble factor) used across the system."""
    phase_norm = str(phase or "").upper()
    if phase_norm not in {"BUBBLE", "ITM", "FINAL_TABLE"}:
        return 1.0

    stack = float(stack_bb or 0.0)
    if stack <= 0:
        return 1.0

    if stack < 10:
        risk_premium = 0.18
    elif stack < 15:
        risk_premium = 0.25
    elif stack < 25:
        risk_premium = 0.34
    elif stack < 35:
        risk_premium = 0.24
    else:
        risk_premium = 0.18

    if phase_norm == "FINAL_TABLE":
        risk_premium += 0.08
    return 1.0 + risk_premium


def atomic_write_json(path: str | Path, payload: dict[str, Any], default: Callable[[Any], Any] | None = None) -> None:
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default), encoding="utf-8")
    tmp.replace(target)


def atomic_write_dataframe_csv(df: Any, path: str | Path, **to_csv_kwargs: Any) -> None:
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_csv(tmp, **to_csv_kwargs)
    tmp.replace(target)