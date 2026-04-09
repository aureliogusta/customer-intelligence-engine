"""
leak_analysis/__init__.py
Módulo de análise de leaks para estudo de poker.

Expõe a interface principal do sistema de detecção de leaks,
agregação de contexto, cálculo de severidade e geração de plano de estudo.
"""

from .modules.db import init_db, get_db_connection, close_db_connection
from .modules.validation import DataQualityValidator, ValidationIssue, ValidationReport
from .modules.leak_detector import LeakDetector
from .modules.context_analyzer import ContextAnalyzer
from .modules.severity_scorer import SeverityScorer
from .modules.study_planner import StudyPlanner
from .modules.report_generator import ReportGenerator

__version__ = "1.0.0"
__all__ = [
    "init_db",
    "get_db_connection",
    "close_db_connection",
    "DataQualityValidator",
    "ValidationIssue",
    "ValidationReport",
    "LeakDetector",
    "ContextAnalyzer",
    "SeverityScorer",
    "StudyPlanner",
    "ReportGenerator",
]
