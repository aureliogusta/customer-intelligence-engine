from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExAnteDecisionRequest:
    hand: str
    position: str
    stack_bb: float
    pot_bb_before: float
    num_players: int
    limpers: int = 0
    open_size_bb: float = 0.0
    street: str = "PREFLOP"
    board_cards: tuple[str, ...] = field(default_factory=tuple)
    ante_bb: float = 0.0
    bb_chips: float = 100.0
    is_3bet_spot: bool = False
    session_id: str | None = None
    source_file: str | None = None


@dataclass(frozen=True)
class DecisionResult:
    action: str
    confidence: float
    warning: str = ""
    abstained: bool = False
    rationale: str = ""
    source: str = "heuristic"
    schema_version: str = "2026.04.ex-ante.v1"
    model_version: str = "none"
    feature_version: str = "2026.04.ex-ante.v1"
    probabilities: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
