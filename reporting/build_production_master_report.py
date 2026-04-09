from __future__ import annotations

import json
from pathlib import Path

from analysis_service.ex_post import analyze_ex_post, load_ex_post_hands

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "production_master_report.json"


def main() -> None:
    # Thin wrapper over the official ex-post/study services.
    df_hands = load_ex_post_hands(limit_hands=50000)
    analysis = analyze_ex_post(df_hands, limit_hands=50000, min_hands_per_context=5)

    out = {
        "generated_from": "analysis_service.ex_post",
        "loaded_rows": int(len(df_hands)),
        "status": analysis.get("status", "unknown"),
        "behavioral_risk_score": analysis.get("behavioral_risk_score", {}),
        "top_leaks": (analysis.get("report", {}) or {}).get("top_10_leaks", [])[:10],
        "study_plan_next_48h": analysis.get("study_plan", [])[:8],
        "contexts": analysis.get("contexts", []),
        "severity_scores": analysis.get("severity_scores", []),
    }

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Relatório mestre gerado em {OUT_PATH.name}")


if __name__ == "__main__":
    main()
