from __future__ import annotations

from typing import Any

import pandas as pd

from analysis_service.ex_post import analyze_ex_post, load_ex_post_hands


def build_study_recommendations(df_hands: pd.DataFrame, limit_hands: int = 5000, min_hands_per_context: int = 5) -> list[dict[str, Any]]:
    analysis = analyze_ex_post(df_hands, limit_hands=limit_hands, min_hands_per_context=min_hands_per_context)
    return list(analysis.get("study_plan", []))


def build_study_recommendations_from_db(
    limit_hands: int = 5000,
    min_hands_per_context: int = 5,
    hero_name: str | None = None,
    session_id: str | None = None,
    source_file_like: str | None = None,
) -> list[dict[str, Any]]:
    df_hands = load_ex_post_hands(
        limit_hands=limit_hands,
        hero_name=hero_name,
        session_id=session_id,
        source_file_like=source_file_like,
    )
    return build_study_recommendations(
        df_hands,
        limit_hands=limit_hands,
        min_hands_per_context=min_hands_per_context,
    )
