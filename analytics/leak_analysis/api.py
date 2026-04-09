"""
leak_analysis/api.py
API FastAPI para exposição do módulo de análise de leaks.

Endpoints:
  GET /health
  GET /leaks/summary
  GET /leaks/top10
  GET /leaks/by-position/{position}
  GET /leaks/by-stack/{stack_bucket}
  POST /leaks/analyze (trigg análise completa)
  GET /study-plan
  GET /report/json
  GET /report/html
"""

import json
import logging
import threading
import time
import os
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd

try:
    from .modules import (
        init_db,
        execute_query,
        DataQualityValidator,
        LeakDetector,
        LeakDetection,
        ContextAnalyzer,
        SeverityScorer,
        StudyPlanner,
        ReportGenerator,
    )
    from .modules.analysis_utils import deduplicate_hands
except ImportError:  # pragma: no cover - execução direta como script
    from modules import (
        init_db,
        execute_query,
        DataQualityValidator,
        LeakDetector,
        LeakDetection,
        ContextAnalyzer,
        SeverityScorer,
        StudyPlanner,
        ReportGenerator,
    )
    from modules.analysis_utils import deduplicate_hands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [LEAK-API]  %(message)s",
)
log = logging.getLogger(__name__)


def _detect_chunk(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        return []
    df = pd.DataFrame(records)
    detector = LeakDetector(df)
    detector.detect_all()
    out = detector.to_dataframe()
    if out.empty:
        return []
    return out.to_dict(orient="records")

app = FastAPI(
    title="Poker Leak Analysis API",
    description="API para análise de leaks e plano de estudo",
    version="1.0.0",
)

_cache: dict[str, Any] = {
    "report": None,
    "signature": None,
    "params": None,
    "last_analysis_at": None,
}

_cache_lock = threading.RLock()
_analysis_seq = 0

# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Requisição de análise."""
    limit_hands: int = 5000
    min_hands_per_context: int = 5

class LeakResponse(BaseModel):
    """Resposta de leak."""
    leak_code: str
    position: str
    stack_bucket: str
    stack_depth_bucket: str = ""
    severity: str
    score: float
    potential_gain: float
    sample_size: int = 0
    confidence: float = 0.0
    justification: str = ""
    why_i_lost: str = ""
    shap_like_values: Dict[str, float] = Field(default_factory=dict)

class StudyRecommendationResponse(BaseModel):
    """Resposta de recomendação de estudo."""
    priority: int
    leak_code: str
    severity: str
    hours: float
    expected_gain: float
    recommendation: str
    spot: str = "UNKNOWN"
    bayesian_bb100: float = 0.0
    sample_size: int = 0
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)
    representative_hand_ids: List[Any] = Field(default_factory=list)
    hand_references: List[str] = Field(default_factory=list)

class ExecutiveSummaryResponse(BaseModel):
    """Resumo executivo."""
    total_hands: int
    leaks_detected: int
    total_potential_gain: float
    key_message: str
    top_leaks: List[LeakResponse]
    analysis_warnings: List[str] = Field(default_factory=list)


def _dataset_signature(limit_hands: Optional[int] = None, min_hands_per_context: Optional[int] = None) -> str:
    query = "SELECT COUNT(*) AS total_hands, MAX(date_utc) AS latest_hand, MAX(ingested_at) AS latest_ingested FROM hands"
    rows = execute_query(query, fetch="one") or {}
    latest = rows.get("latest_hand") or rows.get("latest_ingested") or ""
    total = rows.get("total_hands", 0)
    return f"{total}:{latest}:{limit_hands or 'all'}:{min_hands_per_context or 'default'}"


def _invalidate_cache() -> None:
    with _cache_lock:
        _cache["report"] = None
        _cache["signature"] = None
        _cache["params"] = None
        _cache["last_analysis_at"] = None


def _ensure_cache_fresh(limit_hands: Optional[int] = None, min_hands_per_context: Optional[int] = None) -> None:
    with _cache_lock:
        report = _cache["report"]
        params = dict(_cache.get("params") or {})
        signature = _cache.get("signature")
    if report is None:
        return
    if params:
        if limit_hands is None:
            limit_hands = params.get("limit_hands")
        if min_hands_per_context is None:
            min_hands_per_context = params.get("min_hands_per_context")
    current_signature = _dataset_signature(limit_hands, min_hands_per_context)
    if signature != current_signature:
        _invalidate_cache()


def _get_cached_report_snapshot() -> Optional[Dict[str, Any]]:
    with _cache_lock:
        if _cache["report"] is None:
            return None
        return dict(_cache["report"])

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "service": "leak-analysis"}

@app.post("/analyze")
def analyze(req: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Dispara análise de leaks em background.
    
    Retorna status e ETA.
    """
    log.info("Análise disparada pelo usuário")
    _invalidate_cache()
    global _analysis_seq
    with _cache_lock:
        _analysis_seq += 1
        current_job_id = _analysis_seq
    background_tasks.add_task(_run_analysis_background, req.limit_hands, req.min_hands_per_context, current_job_id)
    
    return {
        "status": "queued",
        "message": "Análise iniciada. Verifique o endpoint /leaks/summary para progresso",
    }

@app.get("/leaks/summary", response_model=ExecutiveSummaryResponse)
def get_summary(limit_hands: Optional[int] = None, min_hands_per_context: Optional[int] = None):
    """Retorna resumo executivo da última análise."""
    _ensure_cache_fresh(limit_hands, min_hands_per_context)
    snapshot = _get_cached_report_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Nenhuma análise disponível. Execute POST /analyze primeiro.")
    report = dict(snapshot["executive"])
    
    return ExecutiveSummaryResponse(
        total_hands=report["total_hands_analyzed"],
        leaks_detected=report["leaks_detected"],
        total_potential_gain=report["total_potential_gain"],
        key_message=report["key_message"],
        analysis_warnings=report.get("analysis_warnings", []),
        top_leaks=[
            LeakResponse(
                leak_code=leak["leak"],
                position=leak["position"],
                stack_bucket=leak["stack"],
                stack_depth_bucket=leak.get("stack_depth_bucket", leak["stack"]),
                severity=leak["severity"],
                score=leak["score"],
                potential_gain=leak["potential_gain"],
                sample_size=leak.get("sample_size", 0),
                confidence=leak.get("confidence", 0.0),
                justification=leak.get("justification", ""),
                why_i_lost=leak.get("why_i_lost", ""),
                shap_like_values=leak.get("shap_like_values", {}),
            )
            for leak in report["top_3_leaks"]
        ],
    )

@app.get("/leaks/top10")
def get_top_10_leaks(limit_hands: Optional[int] = None, min_hands_per_context: Optional[int] = None):
    """Retorna top 10 leaks por severidade."""
    _ensure_cache_fresh(limit_hands, min_hands_per_context)
    snapshot = _get_cached_report_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Nenhuma análise disponível.")
    return list(snapshot["top_10_leaks"])

@app.get("/leaks/by-position/{position}")
def get_leaks_by_position(position: str, limit_hands: Optional[int] = None, min_hands_per_context: Optional[int] = None):
    """Filtra leaks por posição."""
    _ensure_cache_fresh(limit_hands, min_hands_per_context)
    snapshot = _get_cached_report_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Nenhuma análise disponível.")

    top_leaks = list(snapshot["top_10_leaks"])

    leaks = [
        leak for leak in top_leaks
        if leak["position"] == position.upper()
    ]
    
    return {
        "position": position.upper(),
        "leaks": leaks,
        "total_cost": sum(leak["potential_gain"] for leak in leaks),
    }

@app.get("/leaks/by-stack/{stack_bucket}")
def get_leaks_by_stack(stack_bucket: str, limit_hands: Optional[int] = None, min_hands_per_context: Optional[int] = None):
    """Filtra leaks por stack bucket."""
    _ensure_cache_fresh(limit_hands, min_hands_per_context)
    snapshot = _get_cached_report_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Nenhuma análise disponível.")
    
    stack = stack_bucket.upper()
    valid_known = {"LOW", "MID", "HIGH", "0-10BB", "10-15BB", "15-20BB", "20-25BB", "25-35BB", "35-50BB", "50BB+"}
    if stack not in valid_known:
        raise HTTPException(status_code=400, detail="Stack bucket inválido")
    
    top_leaks = list(snapshot["top_10_leaks"])

    leaks = [
        leak for leak in top_leaks
        if leak["stack"] == stack or leak["stack"].upper() == stack
    ]
    
    return {
        "stack_bucket": stack,
        "leaks": leaks,
        "total_cost": sum(leak["potential_gain"] for leak in leaks),
    }

@app.get("/study-plan", response_model=List[StudyRecommendationResponse])
def get_study_plan(limit_hands: Optional[int] = None, min_hands_per_context: Optional[int] = None):
    """Retorna plano de estudo priorizado."""
    _ensure_cache_fresh(limit_hands, min_hands_per_context)
    snapshot = _get_cached_report_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Nenhuma análise disponível.")

    plan = list(snapshot["study_plan"])

    return [
        StudyRecommendationResponse(
            priority=item["priority"],
            leak_code=item["leak"],
            severity=item["severity"],
            hours=item["hours"],
            expected_gain=item["expected_gain"],
            recommendation=item["recommendation"],
            spot=item.get("spot", "UNKNOWN"),
            bayesian_bb100=item.get("bayesian_bb100", 0.0),
            sample_size=item.get("sample_size", 0),
            confidence=item.get("confidence", 0.0),
            evidence=item.get("evidence", []),
            representative_hand_ids=item.get("representative_hand_ids", []),
            hand_references=item.get("hand_references", []),
        )
        for item in plan[:5]
    ]

@app.get("/report/json")
def get_report_json(limit_hands: Optional[int] = None, min_hands_per_context: Optional[int] = None):
    """Retorna relatório completo em JSON."""
    _ensure_cache_fresh(limit_hands, min_hands_per_context)
    snapshot = _get_cached_report_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Nenhuma análise disponível.")

    return snapshot

@app.get("/report/html", response_class=HTMLResponse)
def get_report_html(limit_hands: Optional[int] = None, min_hands_per_context: Optional[int] = None):
    """Retorna relatório em HTML."""
    _ensure_cache_fresh(limit_hands, min_hands_per_context)
    snapshot = _get_cached_report_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Nenhuma análise disponível.")

    report = snapshot
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Análise de Leaks</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
            header {{ background: #1a1a1a; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            h1 {{ font-size: 2em; margin-bottom: 10px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 30px; }}
            .metric {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric-label {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
            .metric-value {{ font-size: 1.8em; font-weight: bold; color: #1a1a1a; }}
            .section {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h2 {{ font-size: 1.5em; margin-bottom: 15px; color: #1a1a1a; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background: #f9f9f9; font-weight: 600; }}
            tr:hover {{ background: #f5f5f5; }}
            .severity-critical {{ color: #d32f2f; font-weight: bold; }}
            .severity-high {{ color: #f57c00; font-weight: bold; }}
            .severity-medium {{ color: #fbc02d; font-weight: bold; }}
            .severity-low {{ color: #388e3c; font-weight: bold; }}
            .message {{ background: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; margin-bottom: 20px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Análise de Leaks - Relatório Completo</h1>
                <p>Gerado em {report['metadata']['generated_at']}</p>
            </header>
            
            <div class="message">
                <strong>Mensagem Principal:</strong> {report['executive_summary']['key_message']}
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Mãos Analisadas</div>
                    <div class="metric-value">{report['executive_summary']['total_hands_analyzed']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Leaks Detectados</div>
                    <div class="metric-value">{report['executive_summary']['leaks_detected']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Ganho Potencial</div>
                    <div class="metric-value">{report['executive_summary']['total_potential_gain']:.1f} bb</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Horas de Estudo</div>
                    <div class="metric-value">{report['metrics']['estimated_study_hours']:.1f}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Top 10 Leaks</h2>
                <table>
                    <tr><th>#</th><th>Leak</th><th>Posição</th><th>Stack</th><th>Severidade</th><th>Score</th><th>Ganho</th></tr>
                    {"".join(f'<tr><td>{i+1}</td><td>{l["leak"]}</td><td>{l["position"]}</td><td>{l["stack"]}</td><td class="severity-{l["severity"].lower()}">{l["severity"]}</td><td>{l["score"]:.1f}</td><td>{l["potential_gain"]:.1f} bb</td></tr>' for i, l in enumerate(report['top_10_leaks'][:10]))}
                </table>
            </div>
            
            <div class="section">
                <h2>Breakdown por Severidade</h2>
                <table>
                    <tr><th>Nível</th><th>Quantidade</th></tr>
                    <tr><td class="severity-critical">Crítico</td><td>{report['breakdowns']['by_severity'].get('CRÍTICO', 0)}</td></tr>
                    <tr><td class="severity-high">Alto</td><td>{report['breakdowns']['by_severity'].get('ALTO', 0)}</td></tr>
                    <tr><td class="severity-medium">Médio</td><td>{report['breakdowns']['by_severity'].get('MÉDIO', 0)}</td></tr>
                    <tr><td class="severity-low">Baixo</td><td>{report['breakdowns']['by_severity'].get('BAIXO', 0)}</td></tr>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND TASKS
# ─────────────────────────────────────────────────────────────────────────────

def _run_analysis_background(limit_hands: int = 5000, min_hands_per_context: int = 5, job_id: Optional[int] = None) -> None:
    """Executa análise em background."""
    try:
        log.info("Iniciando análise em background...")
        init_db()
        
        # Carregar dados
        query = f"""
        SELECT
            h.id, h.hand_id, h.session_id,
            h.hero_position, h.hero_stack_start, h.big_blind, h.small_blind,
            h.hero_cards, h.hero_vpip, h.hero_pfr, h.hero_aggressor,
            h.hero_action_preflop, h.hero_action_flop, h.hero_action_turn, h.hero_action_river,
            h.board_flop, h.board_turn, h.board_river,
            h.hero_result, h.hero_amount_won, h.went_to_showdown,
            h.m_ratio, h.date_utc,
            s.tournament_name
        FROM hands h
        LEFT JOIN sessions s ON h.session_id = s.session_id
        ORDER BY h.date_utc DESC
        LIMIT {limit_hands}
        """
        
        result = execute_query(query, fetch="all")
        if not result:
            log.warning("Nenhuma mão encontrada")
            return
        
        df_hands = pd.DataFrame(result)
        df_hands = deduplicate_hands(df_hands)
        log.info(f"Carregadas {len(df_hands)} mãos")

        validator = DataQualityValidator(min_rows=20)
        validation_report = validator.validate(df_hands)
        validator.log_report(validation_report)
        if not validation_report.ok:
            log.error("Validação falhou no background; análise abortada.")
            return
        
        timings: Dict[str, float] = {}

        # Análise
        t0 = time.perf_counter()
        if len(df_hands) >= 100000:
            workers = max(2, min(6, (os.cpu_count() or 2) - 1))
            chunk_size = 50000
            chunks = [
                df_hands.iloc[i:i + chunk_size].to_dict(orient="records")
                for i in range(0, len(df_hands), chunk_size)
            ]
            records: List[Dict[str, Any]] = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for partial in ex.map(_detect_chunk, chunks):
                    records.extend(partial)
            detections = [LeakDetection(**d) for d in records]
        else:
            detector = LeakDetector(df_hands)
            detections = detector.detect_all()
        timings["leak_detector_seconds"] = round(time.perf_counter() - t0, 3)
        
        t0 = time.perf_counter()
        ctx_analyzer = ContextAnalyzer(df_hands)
        contexts = ctx_analyzer.analyze(detections)
        df_contexts = ctx_analyzer.aggregate_by_context()
        timings["context_analyzer_seconds"] = round(time.perf_counter() - t0, 3)
        
        if "sample_size" in df_contexts.columns:
            df_contexts = df_contexts[df_contexts["sample_size"] >= int(max(1, min_hands_per_context))].reset_index(drop=True)

        t0 = time.perf_counter()
        scorer = SeverityScorer(df_contexts)
        severity_scores = scorer.score_all()
        timings["severity_scorer_seconds"] = round(time.perf_counter() - t0, 3)
        
        t0 = time.perf_counter()
        planner = StudyPlanner(severity_scores, detections)
        recommendations = planner.plan_study()
        timings["study_planner_seconds"] = round(time.perf_counter() - t0, 3)
        
        t0 = time.perf_counter()
        reporter = ReportGenerator(
            leak_detections=detections,
            contexts=df_contexts,
            severity_scores=severity_scores,
            study_recommendations=recommendations,
            df_hands=df_hands,
        )
        
        report = reporter.generate_full_report()
        timings["report_generator_seconds"] = round(time.perf_counter() - t0, 3)
        report.setdefault("metrics", {})["module_timings"] = timings
        signature = _dataset_signature(limit_hands, min_hands_per_context)
        with _cache_lock:
            if job_id is not None and job_id != _analysis_seq:
                log.warning("Resultado descartado por stale job (job_id=%s, active=%s)", job_id, _analysis_seq)
                return
            _cache["report"] = report
            _cache["signature"] = signature
            _cache["params"] = {
                "limit_hands": limit_hands,
                "min_hands_per_context": min_hands_per_context,
            }
            _cache["last_analysis_at"] = report["metadata"]["generated_at"]
        log.info("Análise concluída com sucesso")
    
    except Exception as e:
        log.error(f"Erro na análise: {e}")

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
