"""
app.py — FastAPI para inference de churn e expansão.
====================================================

Endpoints:
  GET  /api/health
  GET  /api/setup
  POST /api/predict-churn/{account_id}
  POST /api/predict-manual
  GET  /api/accounts
  POST /api/batch-predict
  GET  /api/audit
  GET  /api/performance
  POST /api/drift-check

  -- Revenue Automation --
  POST /api/interventions/run/{account_id}
  POST /api/interventions/batch
  GET  /api/interventions/summary
  GET  /api/reports/{period}
  GET  /api/alerts/history

Run:
    uvicorn app:app --reload --port 8001
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Fix encoding no Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config.settings import settings
from decision_service.inference import ChurnPredictor, ExpansionPredictor, batch_predict
from decision_service.memory import get_memory_store
from decision_service.query_engine import ChurnQueryEngine
from analytics.audit_trail import ChurnAuditTrail
from analytics.drift_monitor import DriftDetector
from analytics.performance_monitor import PerformanceMonitor
from study_service.recommendations import gerar_recomendacoes

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("churn_api")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "CS Churn Predictor API",
    description = "Predição de churn e expansão — com memória semântica, audit trail e drift detection",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ── Singletons (lazy) ─────────────────────────────────────────────────────────

_engine:         ChurnQueryEngine | None  = None
_accounts_df:    pd.DataFrame | None      = None
_drift_detector: DriftDetector | None     = None
_audit_trail     = ChurnAuditTrail(log_path=settings.logs_dir / "audit.jsonl")
_perf_monitor    = PerformanceMonitor(log_path=settings.perf_log_path)


def _models_available() -> bool:
    return (settings.models_dir / "churn_model.pkl").exists() and \
           (settings.models_dir / "churn_scaler.pkl").exists()


def get_engine() -> ChurnQueryEngine:
    global _engine
    if _engine is None:
        if not _models_available():
            raise HTTPException(
                status_code=503,
                detail="Modelos não treinados. Execute 02_training.ipynb.",
            )
        _engine = ChurnQueryEngine.from_env(models_dir=str(settings.models_dir))
    return _engine


def get_accounts() -> pd.DataFrame:
    global _accounts_df
    if _accounts_df is None:
        csv = settings.data_dir / "train_dataset.csv"
        _accounts_df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    return _accounts_df


def get_drift_detector() -> DriftDetector | None:
    global _drift_detector
    if _drift_detector is None:
        df = get_accounts()
        if not df.empty:
            _drift_detector = DriftDetector(df)
    return _drift_detector


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    from config.setup import run_churn_setup
    report = run_churn_setup()
    log.info("\n" + report.as_markdown())
    if not report.ready:
        log.warning("Setup incompleto — alguns endpoints estarão indisponíveis.")

    # Pre-aquece engine se modelos existem
    if _models_available():
        try:
            get_engine()
            log.info("ChurnQueryEngine pronto.")
        except Exception as e:
            log.warning(f"Engine não inicializado: {e}")


# ── Schemas ───────────────────────────────────────────────────────────────────

class FeaturesInput(BaseModel):
    engajamento_pct:    float = 50.0
    nps_score:          float = 7.0
    tickets_abertos:    int   = 2
    dias_no_contrato:   int   = 365
    engagement_trend:   float = 0.0
    tickets_trend:      float = 0.0
    nps_trend:          float = 0.0
    dias_sem_interacao: int   = 5
    mrr:                float = 1000.0
    max_users:          int   = 10
    segment:            str   = "MID_MARKET"


class PredictionResponse(BaseModel):
    account_id:          str
    session_id:          str
    churn_risk:          float
    retention_prob:      float
    risk_level:          str
    mrr_at_risk:         float
    upsell_probability:  Optional[float] = None
    upsell_signal:       Optional[str]   = None
    recommended_actions: List[Dict[str, Any]]
    memory_hits:         int   = 0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status":           "ok",
        "models_available": _models_available(),
        "accounts_loaded":  len(get_accounts()),
        "audit_entries":    len(_audit_trail.entries),
        "perf_records":     len(_perf_monitor.metrics),
    }


@app.get("/api/setup")
def setup_report():
    """Verifica todos os componentes do sistema."""
    from config.setup import run_churn_setup
    report = run_churn_setup()
    return {
        "ready":          report.ready,
        "python_version": report.python_version,
        "platform":       report.platform_name,
        "checks": [
            {"name": c.name, "ok": c.ok, "detail": c.detail}
            for c in report.checks
        ],
    }


@app.post("/api/predict-churn/{account_id}", response_model=PredictionResponse)
async def predict_churn(account_id: str):
    """Predição completa com memória semântica e audit trail."""
    df = get_accounts()
    if df.empty:
        raise HTTPException(status_code=503, detail="Dataset não carregado.")

    row = df[df["account_id"] == account_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Conta {account_id} não encontrada.")

    features = row.iloc[0].to_dict()
    engine   = get_engine()
    turn     = engine.analyze(account_id, features)

    # Audit
    _audit_trail.log_from_turn(turn)
    _perf_monitor.record_single(turn)

    return PredictionResponse(
        account_id          = account_id,
        session_id          = turn.session_id,
        churn_risk          = turn.churn_risk,
        retention_prob      = turn.retention_prob,
        risk_level          = turn.risk_level,
        mrr_at_risk         = turn.mrr_at_risk,
        upsell_probability  = turn.upsell_probability,
        upsell_signal       = turn.upsell_signal,
        recommended_actions = list(turn.recommended_actions),
        memory_hits         = len(turn.memory_hits),
    )


@app.post("/api/predict-manual", response_model=PredictionResponse)
async def predict_manual(features: FeaturesInput, account_id: str = "MANUAL"):
    """Predição passando features diretamente no body."""
    engine   = get_engine()
    feat_dict = features.model_dump()
    turn     = engine.analyze(account_id, feat_dict)

    _audit_trail.log_from_turn(turn)

    return PredictionResponse(
        account_id          = account_id,
        session_id          = turn.session_id,
        churn_risk          = turn.churn_risk,
        retention_prob      = turn.retention_prob,
        risk_level          = turn.risk_level,
        mrr_at_risk         = turn.mrr_at_risk,
        upsell_probability  = turn.upsell_probability,
        upsell_signal       = turn.upsell_signal,
        recommended_actions = list(turn.recommended_actions),
        memory_hits         = len(turn.memory_hits),
    )


@app.get("/api/accounts")
def list_accounts(
    risk_level: Optional[str] = None,
    segment:    Optional[str] = None,
    limit:      int = 50,
):
    """Lista contas com scores de risco."""
    df = get_accounts()
    if df.empty:
        return {"accounts": [], "total": 0}

    engine  = get_engine()
    session = uuid4().hex

    turns = engine.analyze_batch(
        [row.to_dict() for _, row in df.head(limit * 3).iterrows()]
    )

    results = [
        {
            "account_id":  t.account_id,
            "segment":     t.recommended_actions[0].get("code", "") if t.recommended_actions else "",
            "churn_risk":  t.churn_risk,
            "risk_level":  t.risk_level,
            "mrr":         t.mrr,
            "mrr_at_risk": t.mrr_at_risk,
        }
        for t in turns
    ]

    if risk_level:
        results = [r for r in results if r["risk_level"] == risk_level.upper()]
    if segment:
        df_seg = df.set_index("account_id")["segment"].to_dict()
        results = [r for r in results if df_seg.get(r["account_id"]) == segment.upper()]

    results.sort(key=lambda x: x["churn_risk"], reverse=True)
    return {"accounts": results[:limit], "total": len(results)}


@app.post("/api/batch-predict")
async def batch_predict_endpoint(save_csv: bool = True):
    """Predição em lote + drift check + métricas de performance."""
    df = get_accounts()
    if df.empty:
        raise HTTPException(status_code=503, detail="train_dataset.csv não encontrado.")

    engine  = get_engine()
    session = uuid4().hex
    turns   = engine.analyze_batch(
        [row.to_dict() for _, row in df.iterrows()]
    )

    # Audit e performance
    for t in turns:
        _audit_trail.log_from_turn(t)
    metric = _perf_monitor.record_batch(turns, session_id=session)

    # Drift check
    drift_info: Dict[str, Any] = {}
    detector = get_drift_detector()
    if detector:
        report = detector.check_drift(df)
        drift_info = {
            "drift_detected":    any(d.drift_detected for d in report.drifts),
            "should_retrain":    report.should_retrain,
            "critical_features": report.critical_features,
        }
        if report.should_retrain:
            log.warning(f"DRIFT: {report.summary_line()}")
            _audit_trail.log_drift_event(
                session_id       = session,
                drifted_features = report.critical_features,
                should_retrain   = True,
            )

    # Salva CSV
    output_path: str | None = None
    if save_csv:
        output_path = str(settings.data_dir / "test_predictions.csv")
        pd.DataFrame([
            {
                "account_id":  t.account_id,
                "churn_risk":  t.churn_risk,
                "risk_level":  t.risk_level,
                "mrr_at_risk": t.mrr_at_risk,
            }
            for t in turns
        ]).to_csv(output_path, index=False)

    summary = engine.risk_summary(turns)

    return {
        **summary,
        "session_id":   session,
        "performance":  {
            "avg_churn_risk":  metric.avg_churn_risk,
            "total_mrr_at_risk": metric.total_mrr_at_risk,
        },
        "drift":        drift_info,
        "saved_to":     output_path,
    }


@app.get("/api/audit")
def audit_log(account_id: Optional[str] = None, last_n: int = 50, fmt: str = "json"):
    """Retorna audit trail — filtrável por conta."""
    entries = (
        _audit_trail.for_account(account_id)
        if account_id
        else _audit_trail.last_n(last_n)
    )
    if fmt == "csv":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(_audit_trail.to_csv(), media_type="text/csv")
    return {
        "total":   len(_audit_trail.entries),
        "entries": [
            {"timestamp": e.timestamp, "account_id": e.account_id,
             "event_type": e.event_type, "details": e.details}
            for e in entries
        ],
        "summary": _audit_trail.summary(),
    }


@app.get("/api/performance")
def performance_summary(last_n: int = 10):
    """Tendência de performance das últimas N rodadas."""
    return {
        "trend":   _perf_monitor.trend_summary(last_n),
        "latest":  vars(_perf_monitor.latest()) if _perf_monitor.latest() else None,
        "alert":   _perf_monitor.should_alert(settings.mrr_alert_threshold),
    }


@app.post("/api/drift-check")
async def drift_check():
    """Executa drift check sob demanda."""
    df = get_accounts()
    if df.empty:
        raise HTTPException(status_code=503, detail="Dataset não disponível.")
    detector = get_drift_detector()
    if not detector:
        raise HTTPException(status_code=503, detail="DriftDetector não inicializado.")
    report = detector.check_drift(df)
    return {
        "timestamp":        report.timestamp,
        "should_retrain":   report.should_retrain,
        "critical_features": report.critical_features,
        "summary":          report.summary_line(),
        "features": [
            {
                "name":           d.feature_name,
                "drift_detected": d.drift_detected,
                "ks_pvalue":      d.ks_pvalue,
                "psi":            d.psi,
                "baseline_mean":  d.baseline_mean,
                "current_mean":   d.current_mean,
            }
            for d in report.drifts
        ],
    }


# ── Revenue Automation endpoints ──────────────────────────────────────────────

_intervention_engine = None


def _get_intervention_engine():
    """Lazy-init do InterventionEngine."""
    global _intervention_engine
    if _intervention_engine is None:
        from revenue_automation.engine import InterventionEngine
        _intervention_engine = InterventionEngine.from_env(
            dry_run      = False,
            accounts_csv = str(settings.data_dir / "train_dataset.csv"),
        )
    return _intervention_engine


@app.post("/api/interventions/run/{account_id}")
async def run_intervention(account_id: str, dry_run: bool = False):
    """
    Executa o ciclo completo de intervenção para uma conta.
    Retorna o InterventionRecord com ações tomadas e canais acionados.
    """
    from revenue_automation.engine import InterventionEngine
    from dataclasses import asdict

    df = get_accounts()
    features: Dict[str, Any] = {}
    if not df.empty:
        row = df[df["account_id"] == account_id]
        if not row.empty:
            features = row.iloc[0].to_dict()

    if not features:
        raise HTTPException(status_code=404, detail=f"Conta {account_id} não encontrada no dataset.")

    engine = InterventionEngine.from_env(
        dry_run      = dry_run,
        accounts_csv = str(settings.data_dir / "train_dataset.csv"),
    )
    record = engine.intervene_one(account_id, features)
    return asdict(record)


@app.post("/api/interventions/batch")
async def run_intervention_batch(dry_run: bool = True):
    """
    Executa intervenção em lote para todas as contas do dataset.
    Use dry_run=True para simular sem enviar alertas reais.
    """
    from revenue_automation.engine import InterventionEngine
    from dataclasses import asdict

    engine = InterventionEngine.from_env(
        dry_run      = dry_run,
        accounts_csv = str(settings.data_dir / "train_dataset.csv"),
    )
    records = engine.intervene_batch()
    summary = engine.batch_summary(records)
    return {
        "dry_run":  dry_run,
        "summary":  summary,
        "records":  [asdict(r) for r in records],
    }


@app.get("/api/interventions/summary")
async def intervention_summary():
    """
    Retorna sumário do histórico de intervenções registradas no audit trail.
    """
    entries = [e for e in _audit_trail.entries if e.event_type == "intervention"]
    from collections import Counter
    actions: Counter = Counter()
    channels: Counter = Counter()
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for e in entries:
        for a in e.details.get("actions", []):
            actions[a] += 1
        for c in e.details.get("channels", []):
            channels[c] += 1
        level = e.details.get("risk_level", "")
        if level in risk_counts:
            risk_counts[level] += 1

    return {
        "total_interventions": len(entries),
        "risk_distribution":   risk_counts,
        "top_actions":         dict(actions.most_common(5)),
        "top_channels":        dict(channels.most_common()),
    }


@app.get("/api/reports/{period}")
async def generate_report(
    period:  str,
    dry_run: bool = True,
    formats: str  = "json",
):
    """
    Gera relatório de intervenção para o período: daily / weekly / biweekly / monthly.
    Retorna metadados e caminhos dos arquivos salvos.
    """
    if period not in ("daily", "weekly", "biweekly", "monthly"):
        raise HTTPException(status_code=400, detail=f"Período inválido: {period}")

    from revenue_automation.jobs.runner import run_report
    fmts = [f.strip() for f in formats.split(",")]
    result = run_report(
        period       = period,
        dry_run      = dry_run,
        output_dir   = "reports",
        accounts_csv = str(settings.data_dir / "train_dataset.csv"),
        formats      = fmts,
    )
    return result


@app.get("/api/alerts/history")
async def alerts_history(limit: int = 20):
    """
    Retorna histórico dos últimos alertas de intervenção disparados.
    Filtra pelo event_type='intervention' no audit trail.
    """
    entries = [e for e in _audit_trail.entries if e.event_type == "intervention"]
    entries = entries[-limit:]
    return {
        "total":  len(_audit_trail.entries),
        "alerts": [
            {
                "timestamp":  e.timestamp,
                "account_id": e.account_id,
                "actions":    e.details.get("actions", []),
                "channels":   e.details.get("channels", []),
                "churn_risk": e.details.get("churn_risk"),
                "risk_level": e.details.get("risk_level"),
                "mrr_at_risk": e.details.get("mrr_at_risk"),
            }
            for e in entries
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.api_host, port=settings.api_port, reload=True)
