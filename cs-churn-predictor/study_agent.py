"""
study_agent.py — Agente LLM para análise de contas CS.
=======================================================

Pipeline:
  features da conta + score de churn (ML) → Ollama → análise em português

Uso:
  python study_agent.py
  python study_agent.py --account ACC_000042
  python study_agent.py --csv data/train_dataset.csv --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any

import pandas as pd

# Fix encoding no Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "mistral:7b"          # alternativa: llama3.2:3b  (mais leve)
OLLAMA_TIMEOUT = 120                 # segundos

ROOT      = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "ml" / "models"
DATA_DIR  = ROOT / "data"


# ── LLM (Ollama) ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Você é um especialista em Customer Success com foco em retenção e expansão de receita.
Sua tarefa é:
1. Analisar os dados de saúde de um cliente
2. Identificar os principais problemas e riscos
3. Recomendar ações concretas com prioridade clara

Responda sempre em português do Brasil.
Inclua prioridade (ALTA, MÉDIA, BAIXA) e estimativa de impacto financeiro.
Seja objetivo — máximo 300 palavras.
"""


def _check_ollama() -> bool:
    try:
        req = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        return req.status == 200
    except Exception:
        return False


def _stream_ollama(prompt: str) -> str:
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )

    output = ""
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    output += token
                    print(token, end="", flush=True)
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except urllib.error.URLError as e:
        print(f"\n[Ollama] Erro de conexão: {e}")

    print()  # newline final
    return output


# ── Agent ─────────────────────────────────────────────────────────────────────

class CSAnalysisAgent:
    """
    Agente de análise de CS que combina dados de saúde + score ML → recomendações LLM.
    """

    def __init__(self, ollama_base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.ollama_url = ollama_base_url
        self.model      = model

    def _build_context(self, account: Dict[str, Any]) -> str:
        risk_pct   = account.get("churn_risk", 0)
        risk_label = account.get("risk_level", "N/A")
        upsell_pct = account.get("upsell_probability", 0)

        return f"""
DADOS DA CONTA:
- Nome             : {account.get('name', account.get('account_id', 'N/A'))}
- ID               : {account.get('account_id', 'N/A')}
- Segmento         : {account.get('segment', 'N/A')}
- MRR              : R$ {account.get('mrr', 0):,.2f}
- Dias no contrato : {account.get('dias_no_contrato', 0)} dias

INDICADORES DE SAÚDE (últimos 30 dias):
- Engajamento      : {account.get('engajamento_pct', 0):.1f}%
- NPS              : {account.get('nps_score', 0):.1f}/10
- Tickets abertos  : {account.get('tickets_abertos', 0)}
- Dias sem login   : {account.get('dias_sem_interacao', 0)}

TENDÊNCIAS (30d vs 60d):
- Engajamento      : {_trend_label(account.get('engagement_trend', 0))}
- Tickets          : {_trend_label(account.get('tickets_trend', 0), invert=True)}
- NPS              : {_trend_label(account.get('nps_trend', 0))}

PREDIÇÃO ML:
- Risco de churn   : {risk_pct:.1%}  [{risk_label}]
- Prob. de upsell  : {upsell_pct:.1%}
"""

    def analisar(self, account: Dict[str, Any]) -> str:
        context = self._build_context(account)
        prompt  = f"{SYSTEM_PROMPT}\n\n{context}\n\nAnálise e recomendações:"
        return _stream_ollama(prompt)

    def analisar_batch(self, accounts: list[Dict[str, Any]]) -> None:
        for acc in accounts:
            name = acc.get("name", acc.get("account_id", "?"))
            mrr  = acc.get("mrr", 0)
            risk = acc.get("churn_risk", 0)
            print(f"\n{'='*60}")
            print(f"  CONTA: {name}  |  MRR: R$ {mrr:,.0f}  |  RISCO: {risk:.1%}")
            print(f"{'='*60}\n")
            self.analisar(acc)


def _trend_label(val: float, invert: bool = False) -> str:
    """Converte valor numérico de trend em label legível."""
    if invert:
        val = -val
    if val > 0.10:
        return f"↑ subindo ({val:+.0%})"
    elif val < -0.10:
        return f"↓ caindo ({val:+.0%})"
    else:
        return f"→ estável ({val:+.0%})"


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _load_predictions() -> pd.DataFrame | None:
    pred_path = DATA_DIR / "test_predictions.csv"
    if pred_path.exists():
        return pd.read_csv(pred_path)
    return None


def _mock_account() -> Dict[str, Any]:
    """Conta de demonstração para testes sem DB/CSV."""
    return {
        "account_id":           "ACC_000042",
        "name":                 "Acme Corp",
        "segment":              "MID_MARKET",
        "mrr":                  5000.0,
        "dias_no_contrato":     180,
        "engajamento_pct":      35.0,
        "nps_score":            4.0,
        "tickets_abertos":      8,
        "engagement_trend":     -0.45,
        "tickets_trend":         0.80,
        "nps_trend":            -1.2,
        "dias_sem_interacao":   21,
        "churn_risk":           0.82,
        "risk_level":           "HIGH",
        "upsell_probability":   0.03,
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CS Analysis Agent")
    parser.add_argument("--account", help="ID da conta (ex: ACC_000042)")
    parser.add_argument("--csv",     help="CSV com predições (data/test_predictions.csv)")
    parser.add_argument("--top",     type=int, default=3, help="Analisar top N contas em risco")
    parser.add_argument("--demo",    action="store_true",  help="Usar conta demo")
    args = parser.parse_args()

    if not _check_ollama():
        print(f"[AVISO] Ollama não respondeu em {OLLAMA_URL}")
        print("  Inicie com: ollama serve")
        print(f"  Baixe o modelo com: ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    agent = CSAnalysisAgent()

    if args.demo or (not args.account and not args.csv):
        # Modo demo
        account = _mock_account()
        print(f"\n=== ANÁLISE DE {account['name']} (DEMO) ===\n")
        agent.analisar(account)
        return

    # Carregar CSV de predições
    csv_path = Path(args.csv) if args.csv else DATA_DIR / "test_predictions.csv"
    if not csv_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {csv_path}")
        print("  Execute primeiro: python data/generate_training_data.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if args.account:
        subset = df[df["account_id"] == args.account]
        if subset.empty:
            print(f"[ERRO] Conta {args.account} não encontrada.")
            sys.exit(1)
        accounts = [subset.iloc[0].to_dict()]
    else:
        # Top N em risco
        if "churn_risk" not in df.columns:
            print("[ERRO] CSV não tem coluna 'churn_risk'. Execute inference primeiro.")
            sys.exit(1)
        top_df   = df.nlargest(args.top, "churn_risk")
        accounts = [row.to_dict() for _, row in top_df.iterrows()]

    agent.analisar_batch(accounts)


if __name__ == "__main__":
    main()
