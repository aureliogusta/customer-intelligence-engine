from __future__ import annotations

from typing import Any

import pandas as pd

from feature_store.contracts import ExAnteFeatureContract, ExAnteObservation


ACTION_MAP = {
    "FOLD": "FOLD",
    "CALL": "CALL",
    "CHECK": "CALL",
    "BET": "RAISE",
    "RAISE": "RAISE",
    "RFI": "RAISE",
    "3-BET": "RAISE",
    "3BET": "RAISE",
    "4-BET": "RAISE",
    "4BET": "RAISE",
    "ALL-IN": "RAISE",
}


def canonical_action(action: str) -> str | None:
    text = str(action or "").upper().strip()
    if not text:
        return None
    if "ALL" in text and "IN" in text:
        return "RAISE"
    if text.startswith("FOLD"):
        return "FOLD"
    if text.startswith("CALL") or text.startswith("CHECK"):
        return "CALL"
    if any(token in text for token in ("RAISE", "BET", "RFI", "3BET", "3-BET", "4BET", "4-BET")):
        return "RAISE"
    return ACTION_MAP.get(text)


def _session_order(df: pd.DataFrame) -> list[str]:
    if df.empty or "session_id" not in df.columns:
        return []
    sort_col = "date_utc" if "date_utc" in df.columns else ("ingested_at" if "ingested_at" in df.columns else None)
    grouped = df.groupby("session_id", dropna=False)
    if sort_col:
        ordered = grouped[sort_col].min().sort_values(kind="stable")
        return [str(idx) for idx in ordered.index.tolist()]
    return [str(idx) for idx in grouped.size().index.tolist()]


def walk_forward_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {"train": df.copy(), "val": df.copy(), "test": df.copy()}

    ordered_sessions = _session_order(df)
    if not ordered_sessions:
        return {"train": df.copy(), "val": df.iloc[0:0].copy(), "test": df.iloc[0:0].copy()}

    n_sessions = len(ordered_sessions)
    train_end = max(1, int(n_sessions * 0.6))
    val_end = max(train_end + 1, int(n_sessions * 0.8))
    train_sessions = set(ordered_sessions[:train_end])
    val_sessions = set(ordered_sessions[train_end:val_end])
    test_sessions = set(ordered_sessions[val_end:])

    session_col = df["session_id"].astype(str)
    return {
        "train": df[session_col.isin(train_sessions)].copy(),
        "val": df[session_col.isin(val_sessions)].copy(),
        "test": df[session_col.isin(test_sessions)].copy(),
    }


def build_ex_ante_training_frame(df_hands: pd.DataFrame) -> pd.DataFrame:
    contract = ExAnteFeatureContract()
    rows: list[dict[str, Any]] = []
    if df_hands is None or df_hands.empty:
        return pd.DataFrame()

    for _, row in df_hands.iterrows():
        label = canonical_action(row.get("hero_action_preflop", ""))
        if label is None:
            continue

        bb = float(row.get("big_blind", 0) or 0)
        stack_start = float(row.get("hero_stack_start", 0) or 0)
        stack_bb = stack_start / bb if bb > 0 else 0.0
        ante = float(row.get("ante", 0) or 0)
        ante_bb = ante / bb if bb > 0 else 0.0

        obs = ExAnteObservation(
            hand=str(row.get("hero_cards", "") or ""),
            position=str(row.get("hero_position", "") or ""),
            stack_bb=stack_bb,
            pot_bb_before=0.0,
            num_players=int(row.get("num_players", 0) or 0),
            limpers=int(row.get("limpers", 0) or 0),
            open_size_bb=float(row.get("open_size_bb", 0.0) or 0.0),
            street="PREFLOP",
            board_cards=tuple(),
            ante_bb=ante_bb,
            bb_chips=bb if bb > 0 else 100.0,
            is_3bet_spot=bool(row.get("is_3bet_spot", False)),
            session_id=str(row.get("session_id", "") or ""),
            source_file=str(row.get("source_file", "") or ""),
        )

        try:
            features = contract.build_row(obs)
        except ValueError:
            continue

        features["label_action"] = label
        features["date_utc"] = row.get("date_utc")
        features["session_id"] = str(row.get("session_id", "") or "")
        rows.append(features)

    return pd.DataFrame(rows)
