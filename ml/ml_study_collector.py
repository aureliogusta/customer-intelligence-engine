"""
ml_study_collector.py
=====================
Camada de coleta/ML baseada no módulo de estudos.

Objetivo
--------
1. Resgatar dados do PostgreSQL (hands + contexto útil)
2. Executar pipeline de estudos (LeakDetector -> ContextAnalyzer -> SeverityScorer -> StudyPlanner)
3. Enriquecer as mãos com sinais do módulo de estudos
4. Gerar recomendações de estudo heurísticas e exportar um dataset rastreável

Uso
---
python ml_study_collector.py --limit 50000 --output-csv hands_study_enriched.csv --output-json study_ml_summary.json --train
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from shared_logic import atomic_write_json

from leak_analysis.modules import (
    init_db,
    execute_query,
    DataQualityValidator,
    LeakDetector,
    ContextAnalyzer,
    SeverityScorer,
    StudyPlanner,
)
from leak_analysis.modules.analysis_utils import deduplicate_hands
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [STUDY-ML]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("study_ml_collector")

DEFAULT_OUTPUT_CSV = Path("hands_study_enriched.csv")
DEFAULT_OUTPUT_JSON = Path("study_ml_summary.json")


def load_hands_from_postgres(limit: int = 50000) -> pd.DataFrame:
    """Carrega mãos do PostgreSQL para o pipeline de estudos + ML."""
    query = """
    SELECT
        h.*,
        s.tournament_name
    FROM hands h
    LEFT JOIN sessions s ON h.session_id = s.session_id
    ORDER BY h.date_utc DESC
    LIMIT %s
    """
    rows = execute_query(query, (limit,), fetch="all")
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    before = len(df)
    df = deduplicate_hands(df)
    dropped = before - len(df)
    if dropped > 0:
        log.warning("Deduplicação removeu %s linhas", dropped)
    return df


def _build_hand_study_labels(leak_detections: list[Any], severity_scores: list[Any]) -> pd.DataFrame:
    """Consolida sinais do módulo de estudos por hand_id para treino supervisionado."""
    if not leak_detections:
        return pd.DataFrame(columns=[
            "hand_id", "study_leak_code", "study_severity_score", "study_confidence",
            "study_severity_label", "study_priority_rank",
        ])

    severity_by_code = {getattr(s, "leak_code", ""): s for s in severity_scores}

    records = []
    for det in leak_detections:
        hand_id = getattr(det, "hand_id", None)
        leak_code = getattr(det, "leak_code", "UNKNOWN")
        sev = severity_by_code.get(leak_code)
        records.append(
            {
                "hand_id": hand_id,
                "study_leak_code": leak_code,
                "study_severity_score": float(getattr(sev, "severity_score", 0.0) if sev else 0.0),
                "study_confidence": float(getattr(det, "confidence", 0.0) or 0.0),
                "study_severity_label": str(getattr(sev, "severity_label", "BAIXO") if sev else "BAIXO"),
                "study_priority_rank": int(getattr(sev, "priority_rank", 999) if sev else 999),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Se uma mão recebeu múltiplos leaks, mantém o mais severo.
    df = (
        df.sort_values(["study_severity_score", "study_confidence"], ascending=[False, False])
          .drop_duplicates(subset=["hand_id"], keep="first")
          .reset_index(drop=True)
    )
    return df


def build_study_enriched_dataset(df_hands: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Roda módulo de estudos e devolve dataset enriquecido para ML."""
    if df_hands.empty:
        return pd.DataFrame(), {"error": "no_hands"}

    validator = DataQualityValidator(min_rows=20)
    validation = validator.validate(df_hands)
    if not validation.ok:
        return pd.DataFrame(), {"error": "validation_failed", "validation": validation.to_dict()}

    detector = LeakDetector(df_hands)
    detections = detector.detect_all()

    ctx = ContextAnalyzer(df_hands)
    ctx.analyze(detections)
    df_contexts = ctx.aggregate_by_context()

    scorer = SeverityScorer(df_contexts)
    scores = scorer.score_all()

    planner = StudyPlanner(scores, detections)
    recommendations = planner.plan_study()

    df_labels = _build_hand_study_labels(detections, scores)

    enriched = df_hands.copy()
    join_key = "id" if "id" in enriched.columns else "hand_id"
    if not df_labels.empty:
        enriched = enriched.merge(df_labels, how="left", left_on=join_key, right_on="hand_id")
    else:
        enriched["study_leak_code"] = "NONE"
        enriched["study_severity_score"] = 0.0
        enriched["study_confidence"] = 0.0
        enriched["study_severity_label"] = "BAIXO"
        enriched["study_priority_rank"] = 999

    # Defaults para mãos sem leak detectado.
    enriched["study_leak_code"] = enriched["study_leak_code"].fillna("NONE")
    enriched["study_severity_score"] = pd.to_numeric(enriched["study_severity_score"], errors="coerce").fillna(0.0)
    enriched["study_confidence"] = pd.to_numeric(enriched["study_confidence"], errors="coerce").fillna(0.0)
    enriched["study_severity_label"] = enriched["study_severity_label"].fillna("BAIXO")
    enriched["study_priority_rank"] = pd.to_numeric(enriched["study_priority_rank"], errors="coerce").fillna(999).astype(int)

    summary = {
        "hands_total": int(len(df_hands)),
        "hands_enriched": int(len(enriched)),
        "leaks_detected": int(len(detections)),
        "contexts": int(len(df_contexts)),
        "severity_items": int(len(scores)),
        "study_recommendations": int(len(recommendations)),
        "critical_or_high": int(sum(1 for s in scores if getattr(s, "severity_label", "") in {"CRÍTICO", "ALTO"})),
    }
    return enriched, summary


def _build_weekly_missions(df_enriched: pd.DataFrame, top_n: int = 3) -> list[dict[str, Any]]:
    """Gera missões semanais ordenadas por prioridade de estudo com EV loss acumulado."""
    if df_enriched.empty or "study_leak_code" not in df_enriched.columns:
        return []

    df = df_enriched.copy()
    df["study_priority_rank"] = pd.to_numeric(df.get("study_priority_rank", 999), errors="coerce").fillna(999)
    df["study_severity_score"] = pd.to_numeric(df.get("study_severity_score", 0.0), errors="coerce").fillna(0.0)
    df["study_confidence"] = pd.to_numeric(df.get("study_confidence", 0.0), errors="coerce").fillna(0.0)

    amount_won = pd.to_numeric(df.get("hero_amount_won", 0.0), errors="coerce").fillna(0.0)
    bb = pd.to_numeric(df.get("big_blind", 0.0), errors="coerce").replace(0, pd.NA)
    loss_bb = ((-amount_won).clip(lower=0.0) / bb).fillna(0.0)
    df["ev_loss_bb"] = loss_bb

    valid = df[df["study_leak_code"].notna() & (df["study_leak_code"] != "NONE")]
    if valid.empty:
        return []

    grouped = (
        valid.groupby("study_leak_code", as_index=False)
        .agg(
            study_priority_rank=("study_priority_rank", "min"),
            study_severity_score=("study_severity_score", "max"),
            study_confidence=("study_confidence", "mean"),
            ev_loss_accumulated_bb=("ev_loss_bb", "sum"),
            occurrences=("study_leak_code", "size"),
        )
        .sort_values(["study_priority_rank", "ev_loss_accumulated_bb"], ascending=[True, False])
    )

    missions = []
    for _, row in grouped.head(top_n).iterrows():
        missions.append(
            {
                "topic": str(row["study_leak_code"]),
                "study_priority_rank": int(row["study_priority_rank"]),
                "study_severity_score": round(float(row["study_severity_score"]), 3),
                "study_confidence": round(float(row["study_confidence"]), 3),
                "ev_loss_accumulated_bb": round(float(row["ev_loss_accumulated_bb"]), 3),
                "occurrences": int(row["occurrences"]),
            }
        )
    return missions


def train_from_study_dataset(df_enriched: pd.DataFrame) -> dict[str, Any]:
    """Compatibilidade: não treina modelo. O estudo é heurístico por contrato."""
    return {
        "status": "skipped",
        "reason": "study_is_heuristic_only",
        "hands": int(len(df_enriched)),
    }


def run_study_collection(
    limit: int = 50000,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
    train: bool = False,
) -> dict[str, Any]:
    """Executa coleta+enriquecimento (+ treino opcional) e persiste resumo JSON."""
    init_db()

    df_hands = load_hands_from_postgres(limit=limit)
    if df_hands.empty:
        result = {"error": "no_hands"}
        atomic_write_json(output_json, result)
        return result

    df_enriched, summary = build_study_enriched_dataset(df_hands)
    if df_enriched.empty:
        result = {"error": "enrichment_failed", "study_summary": summary}
        atomic_write_json(output_json, result)
        return result

    out_csv = Path(output_csv)
    df_enriched.to_csv(out_csv, index=False)

    result: dict[str, Any] = {
        "study_summary": summary,
        "csv": str(out_csv),
        "weekly_missions": _build_weekly_missions(df_enriched, top_n=3),
    }

    if train:
        result["ml_training"] = train_from_study_dataset(df_enriched)

    atomic_write_json(output_json, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Coletor ML baseado no módulo de estudos")
    parser.add_argument("--limit", type=int, default=50000, help="Número máximo de mãos a carregar")
    parser.add_argument("--output-csv", type=str, default="hands_study_enriched.csv", help="CSV do dataset enriquecido")
    parser.add_argument("--output-json", type=str, default="study_ml_summary.json", help="Resumo JSON")
    parser.add_argument("--train", action="store_true", help="Treina a ML com base no dataset enriquecido")
    args = parser.parse_args()

    result = run_study_collection(
        limit=args.limit,
        output_csv=args.output_csv,
        output_json=args.output_json,
        train=args.train,
    )
    if result.get("error"):
        log.error("Falha no coletor: %s", result.get("error"))
        return
    log.info("Resumo salvo em %s", args.output_json)


if __name__ == "__main__":
    main()
