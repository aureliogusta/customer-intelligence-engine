"""
decision_service/models.py
==========================
Dataclasses de configuração para os modelos de churn e expansão.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChurnModel:
    """Configuração do modelo de predição de churn (30 dias antes)."""
    model_type:   str  = "gradient_boosting"
    target:       str  = "renovado"           # bool — True = renovou, False = churnou
    features: List[str] = field(default_factory=lambda: [
        "engajamento_pct",
        "nps_score",
        "tickets_abertos",
        "dias_no_contrato",
        "engagement_trend",
        "tickets_trend",
        "nps_trend",
        "dias_sem_interacao",
        "mrr",
        "max_users",
        "segment_encoded",
    ])
    test_size:    float = 0.20
    random_state: int   = 42
    n_estimators: int   = 100
    max_depth:    int   = 5
    learning_rate: float = 0.10


@dataclass
class ExpansionModel:
    """Configuração do modelo de predição de expansão/upsell."""
    model_type:   str  = "random_forest"
    target:       str  = "fez_upsell"         # bool — True = fez upsell
    features: List[str] = field(default_factory=lambda: [
        "engajamento_pct",
        "nps_score",
        "dias_no_contrato",
        "engagement_trend",
        "nps_trend",
        "mrr",
        "max_users",
        "segment_encoded",
    ])
    test_size:    float = 0.20
    random_state: int   = 42
    n_estimators: int   = 200
    max_depth:    int   = 6
