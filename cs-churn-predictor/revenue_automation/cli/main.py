"""
revenue_automation/cli/main.py
================================
CLI de intervenção autônoma.

Comandos:
    analyze   <account_id>     Analisa e intervém em uma conta
    batch     [--csv FILE]     Analisa e intervém em todas as contas
    report    [weekly|biweekly|monthly|daily]  Gera relatório do período
    feedback  <correlation_id> <outcome>       Registra feedback de outcome
    alerts    [--limit N]      Mostra histórico de alertas disparados
    test-alert <channel>       Dispara alerta de teste no canal

Uso:
    python -m revenue_automation.cli.main analyze ACC_000001
    python -m revenue_automation.cli.main batch --dry-run
    python -m revenue_automation.cli.main report weekly --formats markdown,html
    python -m revenue_automation.cli.main feedback abc123def renewal_happened
    python -m revenue_automation.cli.main alerts --limit 20
    python -m revenue_automation.cli.main test-alert slack
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level  = logging.INFO,
    format = "%(levelname)-8s %(message)s",
)
log = logging.getLogger("cs.cli")


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_analyze(args: argparse.Namespace) -> int:
    from revenue_automation.engine import InterventionEngine
    import pandas as pd

    engine = InterventionEngine.from_env(
        dry_run      = args.dry_run,
        accounts_csv = args.csv,
    )

    # Busca features da conta no CSV
    features: dict = {}
    if Path(args.csv).exists():
        df  = pd.read_csv(args.csv)
        row = df[df["account_id"] == args.account_id]
        if not row.empty:
            features = row.iloc[0].to_dict()

    if not features:
        # Se não achou no CSV, usa valores padrão para demo
        log.warning("Conta %s não encontrada em %s — usando valores demo.", args.account_id, args.csv)
        features = {
            "account_id":       args.account_id,
            "segment":          "MID_MARKET",
            "mrr":              2000.0,
            "max_users":        10,
            "engajamento_pct":  30.0,
            "nps_score":        4.5,
            "tickets_abertos":  5,
            "dias_no_contrato": 200,
            "engagement_trend": -0.3,
            "tickets_trend":    0.5,
            "nps_trend":        -0.5,
            "dias_sem_interacao": 10,
        }

    record = engine.intervene_one(args.account_id, features)
    _print_record(record)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from revenue_automation.engine import InterventionEngine

    engine  = InterventionEngine.from_env(dry_run=args.dry_run, accounts_csv=args.csv)
    records = engine.intervene_batch()
    summary = engine.batch_summary(records)

    print(f"\n{'='*55}")
    print(f"  BATCH COMPLETO — {len(records)} contas processadas")
    print(f"{'='*55}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from revenue_automation.jobs.runner import run_report

    fmts   = [f.strip() for f in args.formats.split(",")]
    result = run_report(
        period       = args.period,
        dry_run      = args.dry_run,
        output_dir   = args.output,
        accounts_csv = args.csv,
        formats      = fmts,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    from revenue_automation.feedback.store import FeedbackStore
    from revenue_automation.schemas.models import FeedbackOutcome

    store = FeedbackStore()
    try:
        outcome = FeedbackOutcome(args.outcome)
    except ValueError:
        log.error("Outcome inválido: %s. Valores válidos: %s",
                  args.outcome, [o.value for o in FeedbackOutcome])
        return 1

    entry = store.record(
        correlation_id    = args.correlation_id,
        account_id        = args.account_id or "UNKNOWN",
        outcome           = outcome,
        notes             = args.notes or "",
        recorded_by       = "cli",
        churn_risk_before = args.risk_before,
        churn_risk_after  = args.risk_after,
    )
    print(f"Feedback registrado: {entry.feedback_id}")
    print(f"  correlation_id : {entry.correlation_id}")
    print(f"  outcome        : {entry.outcome}")
    return 0


def cmd_alerts(args: argparse.Namespace) -> int:
    """Mostra últimas entradas 'intervention' do audit trail."""
    from analytics.audit_trail import ChurnAuditTrail

    trail   = ChurnAuditTrail(log_path=Path("logs/audit.jsonl"))
    entries = [e for e in trail.entries if e.event_type == "intervention"]
    entries = entries[-args.limit:]

    if not entries:
        print("Nenhum alerta registrado ainda.")
        return 0

    print(f"\n{'='*60}")
    print(f"  Últimos {len(entries)} alertas")
    print(f"{'='*60}")
    for e in entries:
        actions = e.details.get("actions", [])
        risk    = e.details.get("churn_risk", "?")
        print(f"  {e.timestamp[:19]} | {e.account_id:<14} | risk={risk:.2%} | {', '.join(actions)}")
    return 0


def cmd_test_alert(args: argparse.Namespace) -> int:
    """Dispara alerta de teste num canal específico."""
    channel = args.channel.lower()
    payload = {
        "account_id":   "TEST_ACCOUNT",
        "account_name": "Conta de Teste",
        "segment":      "ENTERPRISE",
        "mrr":          10000.0,
        "churn_risk":   "85.0%",
        "mrr_at_risk":  "R$ 8.500,00",
        "urgency":      "CRITICAL",
        "csm":          "Test CSM",
    }

    from revenue_automation.dispatch.channels import console, file as file_ch, slack, email, api_hook
    from revenue_automation.schemas.models import DispatchChannel

    ch_map = {
        "console":  lambda: console.send("TEST_ALERT", payload),
        "file":     lambda: file_ch.send("TEST_ALERT", payload),
        "slack":    lambda: slack.send("TEST_ALERT", payload),
        "email":    lambda: email.send("TEST_ALERT", payload),
        "api_hook": lambda: api_hook.send("TEST_ALERT", payload),
    }

    fn = ch_map.get(channel)
    if fn is None:
        log.error("Canal inválido: %s. Opções: %s", channel, list(ch_map.keys()))
        return 1

    result = fn()
    status = "OK" if result.success else f"FALHOU: {result.error}"
    print(f"Teste canal '{channel}': {status}")
    if result.message:
        print(f"  Mensagem: {result.message}")
    return 0 if result.success else 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_record(record) -> None:
    print(f"\n{'='*55}")
    print(f"  INTERVENCAO: {record.account_id}")
    print(f"{'='*55}")
    print(f"  correlation_id : {record.correlation_id}")
    print(f"  churn_risk     : {record.churn_risk:.1%}")
    print(f"  risk_level     : {record.risk_level}")
    print(f"  acoes          : {', '.join(record.actions_taken)}")
    print(f"  canais usados  : {', '.join(record.channels_used)}")
    print(f"  razoes         : {'; '.join(record.reasons)}")
    print(f"{'='*55}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "cs-intervene",
        description = "CS Revenue Automation CLI",
    )
    parser.add_argument("--dry-run", action="store_true", help="Nao despacha alertas reais")
    parser.add_argument("--csv",     default="data/train_dataset.csv", help="CSV de contas")

    subs = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subs.add_parser("analyze", help="Analisa e intervem em uma conta")
    p_analyze.add_argument("account_id")

    # batch
    subs.add_parser("batch", help="Processa todas as contas em batch")

    # report
    p_report = subs.add_parser("report", help="Gera relatorio do periodo")
    p_report.add_argument("period", choices=["daily", "weekly", "biweekly", "monthly"], default="weekly", nargs="?")
    p_report.add_argument("--output",  default="reports")
    p_report.add_argument("--formats", default="markdown,json")

    # feedback
    p_fb = subs.add_parser("feedback", help="Registra feedback de outcome")
    p_fb.add_argument("correlation_id")
    p_fb.add_argument("outcome")
    p_fb.add_argument("--account-id",   default="UNKNOWN", dest="account_id")
    p_fb.add_argument("--notes",        default="")
    p_fb.add_argument("--risk-before",  type=float, default=None, dest="risk_before")
    p_fb.add_argument("--risk-after",   type=float, default=None, dest="risk_after")

    # alerts
    p_alerts = subs.add_parser("alerts", help="Historico de alertas")
    p_alerts.add_argument("--limit", type=int, default=20)

    # test-alert
    p_ta = subs.add_parser("test-alert", help="Alerta de teste num canal")
    p_ta.add_argument("channel", choices=["console", "file", "slack", "email", "api_hook"])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    dispatch = {
        "analyze":    cmd_analyze,
        "batch":      cmd_batch,
        "report":     cmd_report,
        "feedback":   cmd_feedback,
        "alerts":     cmd_alerts,
        "test-alert": cmd_test_alert,
    }
    fn = dispatch.get(args.command)
    return fn(args) if fn else 1


if __name__ == "__main__":
    sys.exit(main())
