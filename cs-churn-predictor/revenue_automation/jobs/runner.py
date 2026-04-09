"""
revenue_automation/jobs/runner.py
===================================
Jobs reutilizáveis: daily_analysis, periodic_report.

Cada job é uma função simples — pode ser chamada pelo CLI, scheduler,
cron externo, ou qualquer orquestrador.

Uso direto:
    python -m revenue_automation.jobs.runner daily
    python -m revenue_automation.jobs.runner weekly
    python -m revenue_automation.jobs.runner monthly
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Daily analysis ────────────────────────────────────────────────────────────

def run_daily(
    dry_run:      bool = False,
    output_dir:   str  = "reports",
    accounts_csv: str  = "data/train_dataset.csv",
) -> Dict[str, Any]:
    """
    Job diário:
      1. Roda InterventionEngine em batch
      2. Loga todas as intervenções
      3. Verifica drift
      4. Salva relatório diário JSON
    """
    from revenue_automation.engine import InterventionEngine
    from analytics.drift_monitor import DriftDetector
    import pandas as pd

    log.info("=== JOB DAILY ===")

    engine  = InterventionEngine.from_env(
        dry_run      = dry_run,
        accounts_csv = accounts_csv,
    )
    records = engine.intervene_batch()
    summary = engine.batch_summary(records)

    # Drift check
    drift_info: Dict[str, Any] = {}
    if Path(accounts_csv).exists():
        df = pd.read_csv(accounts_csv)
        try:
            detector = DriftDetector(df)
            report   = detector.check_drift(df)
            drift_info = {
                "should_retrain":    report.should_retrain,
                "critical_features": report.critical_features,
            }
            if report.should_retrain:
                log.warning("DRIFT DETECTADO: %s", report.summary_line())
        except Exception as e:
            log.error("Drift check falhou: %s", e)

    result = {
        "job":        "daily",
        "date":       date.today().isoformat(),
        "summary":    summary,
        "drift":      drift_info,
        "dry_run":    dry_run,
    }
    _save_json(result, output_dir, f"daily_{date.today().isoformat()}.json")
    log.info("Daily job concluído: %d intervenções.", len(records))
    return result


# ── Periodic report ───────────────────────────────────────────────────────────

def run_report(
    period:        str  = "weekly",
    dry_run:       bool = False,
    output_dir:    str  = "reports",
    accounts_csv:  str  = "data/train_dataset.csv",
    formats:       List[str] = None,
) -> Dict[str, Any]:
    """
    Gera relatório para o período especificado (weekly / biweekly / monthly).
    Salva em Markdown + JSON (+ HTML se solicitado).
    """
    from revenue_automation.engine import InterventionEngine
    from revenue_automation.reports.builder import ReportBuilder
    from revenue_automation.reports.renderer import save_report
    from revenue_automation.schemas.models import ReportPeriod
    from revenue_automation.feedback.store import FeedbackStore

    formats = formats or ["markdown", "json"]
    log.info("=== JOB REPORT: %s ===", period.upper())

    # Determina período
    today  = date.today()
    period_map = {
        "weekly":   timedelta(days=7),
        "biweekly": timedelta(days=14),
        "monthly":  timedelta(days=30),
        "daily":    timedelta(days=1),
    }
    delta      = period_map.get(period, timedelta(days=7))
    start_date = today - delta
    period_end = today.isoformat()
    period_start = start_date.isoformat()

    # Intervenções
    engine = InterventionEngine.from_env(
        dry_run      = dry_run,
        accounts_csv = accounts_csv,
    )
    records = engine.intervene_batch()

    # Contextos
    import pandas as pd
    ctxs = []
    if Path(accounts_csv).exists():
        from decision_service.query_engine import ChurnQueryEngine
        from revenue_automation.schemas.models import InterventionContext
        qe   = ChurnQueryEngine.from_env()
        df   = pd.read_csv(accounts_csv)
        for _, row in df.iterrows():
            feat = row.to_dict()
            try:
                turn = qe.analyze(str(feat.get("account_id", "")), feat)
                ctxs.append(InterventionContext.from_prediction_turn(turn, feat))
            except Exception:
                pass

    # Feedback
    feedback_counts: Dict[str, int] = {}
    try:
        store           = FeedbackStore()
        feedback_counts = store.outcome_counts(since_days=delta.days)
    except Exception:
        pass

    # Build
    try:
        period_enum = ReportPeriod(period)
    except ValueError:
        period_enum = ReportPeriod.WEEKLY

    builder = ReportBuilder(
        period         = period_enum,
        period_start   = period_start,
        period_end     = period_end,
        contexts       = ctxs,
        records        = records,
        feedback_counts= feedback_counts,
    )
    report_data = builder.build()

    # Save
    saved = save_report(report_data, output_dir=output_dir, formats=formats)

    log.info("Relatório %s gerado: %s", period, saved)
    return {
        "job":         f"report_{period}",
        "period":      period,
        "period_start": period_start,
        "period_end":   period_end,
        "accounts":    report_data.total_accounts,
        "at_risk":     report_data.accounts_at_risk,
        "mrr_at_risk": report_data.total_mrr_at_risk,
        "files":       saved,
        "dry_run":     dry_run,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_json(data: Dict, output_dir: str, filename: str) -> str:
    import json
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(path)


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="CS Revenue Automation Jobs")
    parser.add_argument("job", choices=["daily", "weekly", "biweekly", "monthly"], help="Job a executar")
    parser.add_argument("--dry-run",  action="store_true", help="Não despacha alertas reais")
    parser.add_argument("--output",   default="reports",   help="Diretório de saída")
    parser.add_argument("--csv",      default="data/train_dataset.csv")
    parser.add_argument("--formats",  default="markdown,json", help="Formatos separados por vírgula")
    args = parser.parse_args()

    fmts = [f.strip() for f in args.formats.split(",")]

    if args.job == "daily":
        result = run_daily(dry_run=args.dry_run, output_dir=args.output, accounts_csv=args.csv)
    else:
        result = run_report(
            period       = args.job,
            dry_run      = args.dry_run,
            output_dir   = args.output,
            accounts_csv = args.csv,
            formats      = fmts,
        )

    import json
    print(json.dumps(result, indent=2, default=str))
