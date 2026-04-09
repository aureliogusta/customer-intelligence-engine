"""
Validação estrutural e estatística do histórico de mãos.

O objetivo é separar:
- dado incompleto
- variância
- leak provável

Sem validação, qualquer score posterior vira ruído com aparência de evidência.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .analysis_utils import safe_float, safe_div

log = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    level: str
    code: str
    message: str
    column: str | None = None
    sample_size: int | None = None


@dataclass
class ValidationReport:
    ok: bool
    rows: int
    columns: list[str]
    issues: list[ValidationIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rows": self.rows,
            "columns": self.columns,
            "issues": [asdict(issue) for issue in self.issues],
        }


class DataQualityValidator:
    """Valida integridade mínima do histórico de mãos."""

    REQUIRED_COLUMNS = (
        "id",
        "hand_id",
        "hero_position",
        "hero_stack_start",
        "big_blind",
        "hero_vpip",
        "hero_pfr",
        "hero_action_preflop",
        "hero_amount_won",
        "hero_result",
    )

    NUMERIC_COLUMNS = (
        "hero_stack_start",
        "big_blind",
        "small_blind",
        "ante",
        "hero_vpip",
        "hero_pfr",
        "hero_amount_won",
        "pot_final",
        "m_ratio",
    )

    def __init__(self, min_rows: int = 20) -> None:
        self.min_rows = min_rows

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        issues: list[ValidationIssue] = []

        if df is None or df.empty:
            return ValidationReport(False, 0, [], [ValidationIssue("error", "empty_dataset", "DataFrame vazio ou inexistente")])

        rows = len(df)
        columns = list(df.columns)

        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            issues.append(ValidationIssue("error", "missing_columns", f"Colunas obrigatórias ausentes: {', '.join(missing)}"))

        duplicates = 0
        if "hand_id" in df.columns:
            duplicates = int(df["hand_id"].duplicated().sum())
            if duplicates:
                issues.append(ValidationIssue("error", "duplicate_hand_id", f"Foram encontradas {duplicates} mãos duplicadas", "hand_id", duplicates))

        if rows < self.min_rows:
            issues.append(ValidationIssue("warning", "small_sample", f"Amostra pequena: {rows} linhas (mínimo recomendado {self.min_rows})", sample_size=rows))

        for col in self.NUMERIC_COLUMNS:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            nulls = int(series.isna().sum())
            infs = int(np.isinf(series.fillna(0.0)).sum())
            if nulls:
                issues.append(ValidationIssue("warning", "numeric_nulls", f"{nulls} valores nulos/invalidos em {col}", col, nulls))
            if infs:
                issues.append(ValidationIssue("error", "numeric_infinite", f"{infs} valores infinitos em {col}", col, infs))

        if "hero_stack_start" in df.columns and "big_blind" in df.columns:
            stacks = pd.to_numeric(df["hero_stack_start"], errors="coerce")
            bbs = pd.to_numeric(df["big_blind"], errors="coerce")
            invalid_ratio = int(((stacks <= 0) | (bbs <= 0) | stacks.isna() | bbs.isna()).sum())
            if invalid_ratio:
                issues.append(ValidationIssue("error", "invalid_stack_bb", f"{invalid_ratio} mãos com stack/bb inválidos", "hero_stack_start", invalid_ratio))

        if "hero_vpip" in df.columns and "hero_pfr" in df.columns:
            vpip = pd.to_numeric(df["hero_vpip"], errors="coerce")
            pfr = pd.to_numeric(df["hero_pfr"], errors="coerce")
            out_of_bounds = int((((vpip < 0) | (vpip > 1)) | ((pfr < 0) | (pfr > 1))).sum())
            if out_of_bounds:
                issues.append(ValidationIssue("warning", "rate_out_of_bounds", f"{out_of_bounds} taxas fora de [0,1]", "hero_vpip", out_of_bounds))

        if "hero_amount_won" in df.columns:
            amount = pd.to_numeric(df["hero_amount_won"], errors="coerce")
            if amount.isna().any():
                issues.append(ValidationIssue("warning", "amount_nan", "hero_amount_won contém valores ausentes/invalidos", "hero_amount_won", int(amount.isna().sum())))

        ok = not any(issue.level == "error" for issue in issues)
        return ValidationReport(ok=ok, rows=rows, columns=columns, issues=issues)

    def log_report(self, report: ValidationReport) -> None:
        for issue in report.issues:
            level = logging.ERROR if issue.level == "error" else logging.WARNING
            log.log(level, "%s: %s (%s)", issue.code, issue.message, issue.column or "-")
