"""
leak_analysis/analyzer.py
Orquestrador principal do pipeline de análise de leaks.

Executa:
1. Carregamento de histórico de mãos do PostgreSQL
2. Detecção de leaks
3. Análise de contexto
4. Cálculo de severidade
5. Planejamento de estudo
6. Geração de relatórios

Uso:
    python analyzer.py --output-json report.json --output-html report.html
    python analyzer.py --top-10  # só mostra top 10 leaks
"""

import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

try:
    from .modules import (
        init_db,
        execute_query,
        apply_schema_leak_analysis,
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
        apply_schema_leak_analysis,
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
    format="%(asctime)s  %(levelname)-8s  [LEAK-ANALYSIS]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _detect_chunk(records: list[dict]) -> list[dict]:
    """Executa leak detection em chunk isolado para processamento paralelo."""
    if not records:
        return []
    df = pd.DataFrame(records)
    detector = LeakDetector(df)
    detector.detect_all()
    out = detector.to_dataframe()
    if out.empty:
        return []
    return out.to_dict(orient="records")

def load_hands_from_db() -> pd.DataFrame:
    """Carrega histórico de mãos do PostgreSQL."""
    log.info("Carregando histórico de mãos do banco de dados...")
    
    query = """
    SELECT
        h.id, h.hand_id, h.session_id,
        h.hero_position, h.hero_stack_start, h.big_blind, h.small_blind,
        h.hero_cards, h.hero_vpip, h.hero_pfr, h.hero_aggressor, h.hero_went_allin,
        h.hero_action_preflop, h.hero_action_flop, h.hero_action_turn, h.hero_action_river,
        h.board_flop, h.board_turn, h.board_river,
        h.hero_result, h.hero_amount_won, h.went_to_showdown,
        h.pot_final, h.ante, h.level, h.num_players, h.m_ratio, h.date_utc,
        s.tournament_name
    FROM hands h
    LEFT JOIN sessions s ON h.session_id = s.session_id
    ORDER BY h.date_utc DESC
    LIMIT 10000
    """
    
    try:
        result = execute_query(query, fetch="all")
        if not result:
            log.warning("Nenhuma mão encontrada no banco de dados")
            return pd.DataFrame()
        
        df = pd.DataFrame(result)
        before = len(df)
        df = deduplicate_hands(df)
        dropped = before - len(df)
        if dropped > 0:
            log.warning("Deduplicação removeu %s mãos duplicadas", dropped)
        log.info(f"Carregadas {len(df)} mãos do banco de dados")
        return df
    
    except Exception as e:
        log.error(f"Erro ao carregar dados do DB: {e}")
        return pd.DataFrame()

def run_analysis(df_hands: pd.DataFrame, args) -> Dict:
    """Executa análise completa."""
    
    if df_hands.empty:
        log.error("Sem dados para análise")
        return {}

    validator = DataQualityValidator(min_rows=20)
    validation_report = validator.validate(df_hands)
    validator.log_report(validation_report)
    if not validation_report.ok:
        log.error("Validação estrutural falhou. Corrija os dados antes de confiar nos relatórios.")
        return {
            "error": "validation_failed",
            "validation_report": validation_report.to_dict(),
        }
    
    timings: Dict[str, float] = {}

    # 1. Detecção de leaks
    log.info("Executando LeakDetector...")
    t0 = time.perf_counter()
    if len(df_hands) >= int(getattr(args, "parallel_threshold", 100000)) and int(getattr(args, "workers", 1)) > 1:
        chunk_size = int(max(10000, getattr(args, "chunk_size", 50000)))
        workers = int(max(1, getattr(args, "workers", 2)))
        chunks = [
            df_hands.iloc[i:i + chunk_size].to_dict(orient="records")
            for i in range(0, len(df_hands), chunk_size)
        ]
        log.info("LeakDetector paralelo: %s chunks | workers=%s", len(chunks), workers)
        records: list[dict] = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for partial in ex.map(_detect_chunk, chunks):
                records.extend(partial)
        leak_detections = [LeakDetection(**r) for r in records]
    else:
        detector = LeakDetector(df_hands)
        leak_detections = detector.detect_all()
    timings["leak_detector_seconds"] = round(time.perf_counter() - t0, 3)
    
    # 2. Análise de contexto
    log.info("Executando ContextAnalyzer...")
    t0 = time.perf_counter()
    ctx_analyzer = ContextAnalyzer(df_hands)
    contexts = ctx_analyzer.analyze(leak_detections)
    df_contexts = ctx_analyzer.aggregate_by_context()
    timings["context_analyzer_seconds"] = round(time.perf_counter() - t0, 3)
    
    # 3. Cálculo de severidade
    log.info("Executando SeverityScorer...")
    t0 = time.perf_counter()
    scorer = SeverityScorer(df_contexts)
    severity_scores = scorer.score_all()
    timings["severity_scorer_seconds"] = round(time.perf_counter() - t0, 3)
    
    # 4. Planejamento de estudo
    log.info("Executando StudyPlanner...")
    t0 = time.perf_counter()
    planner = StudyPlanner(severity_scores, leak_detections)
    study_recommendations = planner.plan_study()
    timings["study_planner_seconds"] = round(time.perf_counter() - t0, 3)
    
    # 5. Geração de relatórios
    log.info("Gerando relatórios...")
    t0 = time.perf_counter()
    reporter = ReportGenerator(
        leak_detections=leak_detections,
        contexts=df_contexts,
        severity_scores=severity_scores,
        study_recommendations=study_recommendations,
        df_hands=df_hands,
    )
    
    # Executivo
    executive = reporter.generate_executive_summary()
    timings["report_generator_seconds"] = round(time.perf_counter() - t0, 3)
    log.info(f"\n{'-' * 60}")
    log.info(f"RESUMO EXECUTIVO")
    log.info(f"{'-' * 60}")
    log.info(f"Mãos analisadas: {executive['total_hands_analyzed']}")
    log.info(f"Leaks detectados: {executive['leaks_detected']}")
    log.info(f"Mensagem principal: {executive['key_message']}")
    log.info(f"{'-' * 60}\n")
    
    # Salvar relatórios conforme solicitado
    if args.output_json:
        reporter.to_json(args.output_json)
    
    if args.output_html:
        reporter.to_html(args.output_html)
    
    # Show top leaks
    if args.top_10 or args.top:
        log.info("\nTOP 10 LEAKS POR SEVERIDADE")
        log.info(f"{'-' * 60}")
        for score in severity_scores[:10]:
            log.info(
                f"{score.priority_rank}. {score.leak_code:25s} | "
                f"Score: {score.severity_score:5.1f} ({score.severity_label:8s}) | "
                f"Ganho: {score.potential_roi_gain:.2f} bb"
            )
        log.info(f"{'-' * 60}\n")
    
    return {
        "executive": executive,
        "detections": leak_detections,
        "contexts": df_contexts,
        "severity_scores": severity_scores,
        "study_plan": study_recommendations,
        "validation_report": validation_report.to_dict(),
        "timings": timings,
    }

def main():
    parser = argparse.ArgumentParser(description="Análise de Leaks em Poker")
    parser.add_argument("--output-json", help="Salvar relatório JSON")
    parser.add_argument("--output-html", help="Salvar relatório HTML")
    parser.add_argument(
        "--top-10", "--top", action="store_true", help="Mostrar top 10 leaks"
    )
    parser.add_argument(
        "--init-schema", action="store_true", help="Aplicar schema de leak analysis"
    )
    parser.add_argument("--workers", type=int, default=1, help="Workers para leak detection paralela")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Tamanho do chunk para processamento paralelo")
    parser.add_argument("--parallel-threshold", type=int, default=100000, help="Ativa paralelo acima deste volume")
    
    args = parser.parse_args()
    
    # Inicializar DB
    log.info("Inicializando conexão com banco de dados...")
    init_db()
    
    # Aplicar schema se solicitado
    if args.init_schema:
        log.info("Aplicando schema de leak analysis...")
        apply_schema_leak_analysis()
    
    # Carregar dados
    df_hands = load_hands_from_db()
    if df_hands.empty:
        log.error("Falha ao carregar dados. Saindo.")
        sys.exit(1)
    
    # Executar análise
    results = run_analysis(df_hands, args)
    if not results or results.get("error"):
        log.error("Análise encerrada com falha de validação ou dados insuficientes.")
        sys.exit(2)
    
    log.info("Análise concluída com sucesso")

if __name__ == "__main__":
    main()
