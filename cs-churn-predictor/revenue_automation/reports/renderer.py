"""
revenue_automation/reports/renderer.py
========================================
Renderiza ReportData em Markdown, JSON ou HTML.

Uso:
    report = builder.build()
    md   = render_markdown(report)
    js   = render_json(report)
    html = render_html(report)   # minimalista, sem dependência externa
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ..schemas.models import ReportData


def render_markdown(report: ReportData) -> str:
    d = report
    lines = [
        f"# Relatório CS Intelligence — {d.period.upper()}",
        f"**Período:** {d.period_start} → {d.period_end}",
        f"**Gerado em:** {d.generated_at}",
        "",
        "---",
        "",
        "## Resumo Executivo",
        "",
        d.executive_summary,
        "",
        "---",
        "",
        "## Visão Geral da Carteira",
        "",
        f"| Indicador | Valor |",
        f"|-----------|-------|",
        f"| Total de contas analisadas | {d.total_accounts} |",
        f"| Contas em risco alto (HIGH) | {d.accounts_at_risk} |",
        f"| Contas críticas (HIGH + MRR > 5k) | {d.critical_accounts} |",
        f"| Contas em risco médio (MEDIUM) | {d.recoverable_accounts} |",
        f"| Contas saudáveis (LOW) | {d.safe_accounts} |",
        f"| MRR total analisado | R$ {d.total_mrr_analyzed:,.2f} |",
        f"| MRR em risco | R$ {d.total_mrr_at_risk:,.2f} |",
        f"| MRR potencialmente preservado | R$ {d.estimated_mrr_preserved:,.2f} |",
        f"| Drift detectado | {'Sim' if d.drift_detected else 'Nao'} |",
        "",
        "---",
        "",
        "## Top Contas em Risco",
        "",
        "| Conta | Segmento | MRR | Churn Risk | MRR em Risco | Acao |",
        "|-------|----------|-----|-----------|-------------|------|",
    ]
    for a in d.top_risk_accounts:
        lines.append(
            f"| {a.account_name} | {a.segment} | R$ {a.mrr:,.0f} "
            f"| {a.churn_risk:.1%} | R$ {a.mrr_at_risk:,.0f} | {a.top_action} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Breakdown por Segmento",
        "",
        "| Segmento | Total | Em Risco Alto | MRR | MRR em Risco |",
        "|----------|-------|--------------|-----|-------------|",
    ]
    for seg, info in d.segment_breakdown.items():
        lines.append(
            f"| {seg} | {info['total']} | {info['high_risk']} "
            f"| R$ {info['mrr']:,.0f} | R$ {info['mrr_at_risk']:,.0f} |"
        )

    if d.top_actions_taken:
        lines += [
            "",
            "---",
            "",
            "## Ações Executadas",
            "",
            "| Acao | Qtd |",
            "|------|-----|",
        ]
        for action, count in d.top_actions_taken.items():
            lines.append(f"| {action} | {count} |")

    if d.channels_used:
        lines += [
            "",
            "## Canais Utilizados",
            "",
            "| Canal | Qtd |",
            "|-------|-----|",
        ]
        for ch, count in d.channels_used.items():
            lines.append(f"| {ch} | {count} |")

    if d.feedback_outcomes:
        lines += [
            "",
            "---",
            "",
            "## Feedback de Outcomes",
            "",
            "| Outcome | Qtd |",
            "|---------|-----|",
        ]
        for outcome, count in d.feedback_outcomes.items():
            lines.append(f"| {outcome} | {count} |")

    if d.drift_summary:
        lines += [
            "",
            "---",
            "",
            "## Drift de Dados",
            "",
            f"> {d.drift_summary}",
        ]

    lines += ["", "---", "", "*Gerado automaticamente pelo CS Intelligence Platform.*"]
    return "\n".join(lines)


def render_json(report: ReportData) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str)


def render_html(report: ReportData) -> str:
    """HTML minimalista — sem dependências externas."""
    md = render_markdown(report)
    # Conversão básica Markdown → HTML (suficiente para visualização)
    import re
    html = md
    html = re.sub(r"^# (.+)$",   r"<h1>\1</h1>",   html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$",  r"<h2>\1</h2>",   html, flags=re.MULTILINE)
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>",   html, flags=re.MULTILINE)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"> (.+)",     r"<blockquote>\1</blockquote>", html)
    html = re.sub(r"^---$",      r"<hr>",            html, flags=re.MULTILINE)
    # Tables: basic passthrough wrapped in <pre> for now
    rows = html.split("\n")
    out  = []
    in_table = False
    for row in rows:
        if row.startswith("|"):
            if not in_table:
                out.append('<table border="1" cellpadding="4" cellspacing="0"><tbody>')
                in_table = True
            cells = [c.strip() for c in row.strip("|").split("|")]
            out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        else:
            if in_table:
                out.append("</tbody></table>")
                in_table = False
            out.append(row)
    if in_table:
        out.append("</tbody></table>")

    body = "\n".join(out)
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>CS Report — {report.period}</title>
  <style>
    body {{ font-family: sans-serif; max-width: 960px; margin: 2em auto; color: #333; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
    td, th {{ border: 1px solid #ccc; padding: 6px 10px; }}
    h1 {{ color: #1a1a2e; }} h2 {{ color: #16213e; border-bottom: 1px solid #eee; }}
    blockquote {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 8px 16px; }}
    hr {{ border: none; border-top: 1px solid #ddd; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def save_report(report: ReportData, output_dir: str, formats: list[str] | None = None) -> dict[str, str]:
    """Salva o relatório nos formatos especificados. Retorna dict {formato: caminho}."""
    import os
    from pathlib import Path

    formats = formats or ["markdown", "json"]
    os.makedirs(output_dir, exist_ok=True)
    slug     = f"{report.period}_{report.period_end[:10].replace('-', '')}"
    saved: dict[str, str] = {}

    for fmt in formats:
        if fmt in ("markdown", "md"):
            path = Path(output_dir) / f"report_{slug}.md"
            path.write_text(render_markdown(report), encoding="utf-8")
            saved["markdown"] = str(path)
        elif fmt == "json":
            path = Path(output_dir) / f"report_{slug}.json"
            path.write_text(render_json(report), encoding="utf-8")
            saved["json"] = str(path)
        elif fmt == "html":
            path = Path(output_dir) / f"report_{slug}.html"
            path.write_text(render_html(report), encoding="utf-8")
            saved["html"] = str(path)

    return saved
