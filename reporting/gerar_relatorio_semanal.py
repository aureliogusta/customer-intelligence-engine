"""
Gera um snapshot semanal em PDF para acompanhamento de performance.

Conteúdo:
- Curva semanal de bb/100 por sessão
- Top 3 leaks da semana
- Nota de disciplina de tilt

Uso:
    .venv\\Scripts\\python.exe gerar_relatorio_semanal.py
    .venv\\Scripts\\python.exe gerar_relatorio_semanal.py --force
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

ROOT = Path(__file__).resolve().parent
SESSION_HISTORY = ROOT / "session_history.csv"
DASHBOARD_FEED = ROOT / "dashboard_live_feed.json"
OUTPUT_PDF = ROOT / "relatorio_semanal_poker.pdf"


def _is_sunday() -> bool:
    return datetime.now().weekday() == 6


def _load_session_history() -> pd.DataFrame:
    if not SESSION_HISTORY.exists():
        return pd.DataFrame()
    df = pd.read_csv(SESSION_HISTORY)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    now = datetime.now()
    start = now - timedelta(days=7)
    if "date" in df.columns:
        df = df[df["date"] >= start]
    return df


def _load_feed() -> dict:
    if not DASHBOARD_FEED.exists():
        return {}
    try:
        return json.loads(DASHBOARD_FEED.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tilt_note(df: pd.DataFrame) -> str:
    if df.empty or "tilt_risk_pct" not in df.columns:
        return "Sem dados suficientes para nota de disciplina de tilt."
    avg_tilt = float(pd.to_numeric(df["tilt_risk_pct"], errors="coerce").fillna(0.0).mean())
    if avg_tilt < 15:
        return f"Disciplina de Tilt: Excelente (media {avg_tilt:.1f}%)."
    if avg_tilt < 30:
        return f"Disciplina de Tilt: Moderada (media {avg_tilt:.1f}%)."
    return f"Disciplina de Tilt: Critica (media {avg_tilt:.1f}%)."


def generate_weekly_pdf(output_path: Path = OUTPUT_PDF) -> Path:
    df = _load_session_history()
    feed = _load_feed()

    top3 = feed.get("weekly_study_topics") or []
    if not top3 and isinstance(feed.get("study_missions"), list):
        top3 = feed["study_missions"][:3]

    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        if not df.empty and "bb100_final" in df.columns:
            y = pd.to_numeric(df["bb100_final"], errors="coerce").fillna(0.0)
            x = pd.to_datetime(df.get("date"), errors="coerce")
            ax.plot(x, y, marker="o", linewidth=2)
            ax.set_title("Curva Semanal de bb/100")
            ax.set_ylabel("bb/100")
            ax.set_xlabel("Data")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Sem dados de sessao para a semana.", ha="center", va="center")
            ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 7))
        ax2.axis("off")
        lines = [
            "Resumo Semanal de Intervencao",
            "",
            f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            _tilt_note(df),
            "",
            "Top 3 Leaks da Semana:",
        ]
        if top3:
            for i, t in enumerate(top3[:3], start=1):
                topic = t.get("topic") or t.get("leak_code") or "UNKNOWN"
                ev = t.get("ev_loss_accumulated_bb", t.get("bayesian_bb100", 0.0))
                lines.append(f"{i}. {topic} | EV Loss: {float(ev):.2f}")
        else:
            lines.append("1. Sem topicos suficientes na semana.")

        lines += ["", "Observacao:", "Mantenha foco no Top 1 leak e execute o checklist tecnico diariamente."]
        ax2.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=12)
        pdf.savefig(fig2)
        plt.close(fig2)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera PDF semanal de performance e estudo")
    parser.add_argument("--force", action="store_true", help="Gera mesmo se nao for domingo")
    args = parser.parse_args()

    if not args.force and not _is_sunday():
        print("[INFO] Hoje nao e domingo. Use --force para gerar manualmente.")
        return

    out = generate_weekly_pdf()
    print(f"[INFO] Relatorio semanal gerado: {out}")


if __name__ == "__main__":
    main()
