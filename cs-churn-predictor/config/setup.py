"""
config/setup.py
================
Verificação de ambiente na inicialização.

Portado do padrão WorkspaceSetup do claw-code-main/src/setup.py:
  - Checa todos os componentes antes de servir requests
  - Retorna SetupReport estruturado (não lança exceções desnecessárias)
  - Imprime status legível no startup da API

Uso:
    report = run_churn_setup()
    print(report.as_markdown())
    if not report.ready:
        sys.exit(1)
"""

from __future__ import annotations

import platform
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    from .settings import settings
except ImportError:
    from config.settings import settings


@dataclass(frozen=True)
class CheckResult:
    name:    str
    ok:      bool
    detail:  str


@dataclass
class SetupReport:
    python_version:  str
    platform_name:   str
    checks:          List[CheckResult] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True se todos os checks obrigatórios passaram."""
        required = {"models", "dataset"}
        return all(c.ok for c in self.checks if c.name in required)

    @property
    def warnings(self) -> List[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.ok]

    def as_markdown(self) -> str:
        lines = [
            "# CS Churn Predictor — Setup Report",
            f"Python  : {self.python_version}",
            f"Platform: {self.platform_name}",
            "",
            "## Checks",
        ]
        for c in self.checks:
            icon = "OK" if c.ok else "FAIL"
            lines.append(f"  [{icon}] {c.name:<20} {c.detail.encode('ascii', errors='replace').decode('ascii')}")
        lines.append("")
        lines.append(f"Ready: {self.ready}")
        return "\n".join(lines)


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_models() -> CheckResult:
    required = ["churn_model.pkl", "churn_scaler.pkl"]
    missing  = [f for f in required if not (settings.models_dir / f).exists()]
    if missing:
        return CheckResult("models", False, f"Faltam: {missing}. Execute 02_training.ipynb.")
    count = len(list(settings.models_dir.glob("*.pkl")))
    return CheckResult("models", True, f"{count} artefatos em {settings.models_dir}")


def _check_dataset() -> CheckResult:
    train = settings.data_dir / "train_dataset.csv"
    if not train.exists():
        return CheckResult("dataset", False, "train_dataset.csv ausente. Execute generate_training_data.py.")
    try:
        import pandas as pd
        df = pd.read_csv(train, nrows=5)
        return CheckResult("dataset", True, f"train_dataset.csv — {len(df.columns)} colunas")
    except Exception as e:
        return CheckResult("dataset", False, f"Erro ao ler CSV: {e}")


def _check_pgvector() -> CheckResult:
    if not settings.pgvector_dsn:
        return CheckResult("pgvector", False, "PGVECTOR_DSN não configurado (opcional).")
    try:
        import importlib
        psycopg = importlib.import_module("psycopg")
        conn    = psycopg.connect(settings.pgvector_dsn)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return CheckResult("pgvector", True, f"Conectado: {settings.pgvector_dsn[:40]}...")
    except ImportError:
        return CheckResult("pgvector", False, "psycopg não instalado: pip install psycopg[binary]")
    except Exception as e:
        return CheckResult("pgvector", False, f"Falha de conexão: {e}")


def _check_ollama() -> CheckResult:
    try:
        req = urllib.request.urlopen(f"{settings.ollama_url}/api/tags", timeout=3)
        if req.status == 200:
            return CheckResult("ollama", True, f"{settings.ollama_url} → {settings.ollama_model}")
        return CheckResult("ollama", False, f"HTTP {req.status}")
    except Exception as e:
        return CheckResult("ollama", False, f"Offline ({e}). LLM agent não disponível.")


def _check_sklearn() -> CheckResult:
    try:
        import sklearn
        return CheckResult("sklearn", True, f"v{sklearn.__version__}")
    except ImportError:
        return CheckResult("sklearn", False, "pip install scikit-learn")


def _check_fastapi() -> CheckResult:
    try:
        import fastapi
        return CheckResult("fastapi", True, f"v{fastapi.__version__}")
    except ImportError:
        return CheckResult("fastapi", False, "pip install fastapi uvicorn")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_churn_setup() -> SetupReport:
    report = SetupReport(
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform_name  = platform.platform(),
        checks         = [
            _check_sklearn(),
            _check_fastapi(),
            _check_models(),
            _check_dataset(),
            _check_pgvector(),
            _check_ollama(),
        ],
    )
    return report


if __name__ == "__main__":
    r = run_churn_setup()
    print(r.as_markdown())
    sys.exit(0 if r.ready else 1)
